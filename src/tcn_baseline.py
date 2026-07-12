"""Matched-information 1D-TCN endpoint forecaster."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

import dataset as D
from interfaces import ForecastInputs, as_numpy, inputs_from_bundle


SCHEMA_VERSION = 2
ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"


@dataclass
class TCNConfig:
    channels: int = 22
    levels: int = 3
    kernel_size: int = 2
    head_hidden: int = 20
    dropout: float = 0.10
    lr: float = 5e-3
    weight_decay: float = 1e-5
    epochs: int = 1200
    patience: int = 120
    grad_clip: float = 1.0
    window_s: float = 60.0
    prefix_min_s: float = 20.0
    wr_beta: float = 0.0
    seed: int = 0


class Chomp1d(nn.Module):
    def __init__(self, amount: int):
        super().__init__()
        self.amount = amount

    def forward(self, x: Tensor) -> Tensor:
        return x[..., :-self.amount].contiguous() if self.amount else x


class TemporalBlock(nn.Module):
    def __init__(self, n_in: int, n_out: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(n_in, n_out, kernel, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(n_out, n_out, kernel, padding=padding, dilation=dilation),
            Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(self.net(x) + self.skip(x))


class TCNForecaster(nn.Module):
    def __init__(self, cfg: Optional[TCNConfig] = None):
        super().__init__()
        self.cfg = cfg or TCNConfig()
        blocks: List[nn.Module] = []
        for level in range(self.cfg.levels):
            blocks.append(TemporalBlock(
                4 if level == 0 else self.cfg.channels,
                self.cfg.channels,
                self.cfg.kernel_size,
                2 ** level,
                self.cfg.dropout,
            ))
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(self.cfg.channels, self.cfg.head_hidden), nn.ReLU(),
            nn.Dropout(self.cfg.dropout), nn.Linear(self.cfg.head_hidden, 1),
        )
        self.register_buffer("x_mean", torch.zeros(4))
        self.register_buffer("x_std", torch.ones(4))
        self.register_buffer("y_mean", torch.tensor(0.0))
        self.register_buffer("y_std", torch.tensor(1.0))

    def raw_features(self, inputs: ForecastInputs) -> Tensor:
        n, w = inputs.I_early.shape
        time = (inputs.t / inputs.t[-1].clamp_min(1.0)).view(1, w).expand(n, -1)
        temp = inputs.T_ambient.view(n, 1).expand(-1, w)
        rh = inputs.RH.view(n, 1).expand(-1, w)
        return torch.stack([inputs.I_early, time, temp, rh], dim=1)

    def standardized(self, inputs: ForecastInputs) -> Tensor:
        raw = self.raw_features(inputs)
        return (raw - self.x_mean.view(1, 4, 1)) / self.x_std.view(1, 4, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.tcn(x)[..., -1]).squeeze(-1)

    @torch.no_grad()
    def predict(self, inputs: ForecastInputs, t_end: float = 900.0) -> np.ndarray:
        del t_end
        self.eval()
        return as_numpy(self(self.standardized(inputs)) * self.y_std + self.y_mean)


def count_trainable(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


@torch.no_grad()
def fit_normalization(model: TCNForecaster, bundle: D.TraceBundle) -> None:
    inputs = inputs_from_bundle(bundle, model.cfg.window_s)
    raw = model.raw_features(inputs)
    model.x_mean.copy_(raw.mean(dim=(0, 2)))
    model.x_std.copy_(raw.std(dim=(0, 2)).clamp_min(1e-6))
    model.y_mean.copy_(bundle.true_I_900.mean())
    model.y_std.copy_(bundle.true_I_900.std().clamp_min(1e-6))


def _prefix(bundle: D.TraceBundle, cfg: TCNConfig, epoch: int) -> ForecastInputs:
    choices = [x for x in (20.0, 30.0, 40.0, 50.0, 60.0) if cfg.prefix_min_s <= x <= cfg.window_s]
    return inputs_from_bundle(bundle, choices[epoch % len(choices)] if choices else cfg.window_s)


@torch.no_grad()
def rmse(model: TCNForecaster, bundle: D.TraceBundle) -> float:
    prediction = model.predict(inputs_from_bundle(bundle, model.cfg.window_s))
    return float(np.sqrt(np.mean((prediction - as_numpy(bundle.true_I_900)) ** 2)))


def train_tcn(
    train_bundle: D.TraceBundle,
    validation_bundle: D.TraceBundle,
    cfg: Optional[TCNConfig] = None,
    wr_reference: Optional[D.TraceBundle] = None,
) -> Dict:
    cfg = cfg or TCNConfig()
    torch.manual_seed(cfg.seed)
    model = TCNForecaster(cfg)
    fit_normalization(model, train_bundle)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = float("inf")
    best_state: Optional[Dict[str, Tensor]] = None
    best_epoch = -1
    stale = 0
    history = {"train": [], "validation_rmse": []}
    resamples = None
    if cfg.wr_beta > 0.0:
        if wr_reference is None:
            raise ValueError("wr_beta>0 requires a disjoint WR-reference bundle")
        from train import _wr_resamples
        resamples = _wr_resamples(train_bundle, wr_reference, cfg.seed)
    for epoch in range(cfg.epochs):
        model.train()
        inputs = _prefix(train_bundle, cfg, epoch)
        optimizer.zero_grad()
        prediction = model(model.standardized(inputs))
        target = (train_bundle.true_I_900 - model.y_mean) / model.y_std
        loss = torch.mean((prediction - target) ** 2)
        if cfg.wr_beta > 0.0:
            assert wr_reference is not None and resamples is not None
            reference_inputs = _prefix(wr_reference, cfg, epoch)
            reference_prediction = model(model.standardized(reference_inputs))
            reference_target = (wr_reference.true_I_900 - model.y_mean) / model.y_std
            source_scores = torch.abs(prediction - target)
            reference_scores = torch.abs(reference_prediction - reference_target)
            distances = []
            for group, ref_index in resamples.items():
                source = torch.sort(source_scores[train_bundle.group_id == group]).values
                weighted_reference = torch.sort(reference_scores[ref_index]).values
                distances.append(torch.mean(torch.abs(source - weighted_reference)))
            loss = loss + cfg.wr_beta * torch.stack(distances).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        value = rmse(model, validation_bundle)
        history["train"].append(float(loss.detach()))
        history["validation_rmse"].append(value)
        if value < best - 1e-8:
            best, best_epoch = value, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is None:
        raise RuntimeError("TCN training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return {
        "model": model,
        "cfg": cfg,
        "best_validation_rmse": best,
        "best_epoch": best_epoch,
        "epochs_run": len(history["train"]),
        "history": history,
    }


def save_checkpoint(result: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema_version": SCHEMA_VERSION,
        "cfg": asdict(result["cfg"]),
        "state_dict": result["model"].state_dict(),
        "metrics": {
            "best_validation_rmse": result["best_validation_rmse"],
            "best_epoch": result["best_epoch"],
            "epochs_run": result["epochs_run"],
        },
    }, path)


def load_checkpoint(path: Path) -> Tuple[TCNForecaster, TCNConfig, Dict]:
    checkpoint = torch.load(path, weights_only=False)
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("legacy TCN checkpoint rejected")
    cfg = TCNConfig(**checkpoint["cfg"])
    model = TCNForecaster(cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, cfg, checkpoint.get("metrics", {})


if __name__ == "__main__":
    from train import LatentODEForecaster, TrainConfig, count_trainable as count_latent

    tcn = TCNForecaster()
    gray = LatentODEForecaster(TrainConfig())
    print(f"TCN trainable={count_trainable(tcn)} gray-box trainable={count_latent(gray)}")
    print("Both accept only ForecastInputs(t, I_early, T, RH).")
