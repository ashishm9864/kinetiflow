"""Training for deployable latent-ODE forecasters.

The recognition encoder reads the observed kinetic prefix and infers the
unknown initial free concentration.  True synthetic ``C_f0`` is never accepted
by the model API or used by the primary loss; it is retained only for held-out
diagnostics in the evaluator.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torchdiffeq import odeint, odeint_adjoint

import dataset as D
from interfaces import ForecastInputs, as_numpy, inputs_from_bundle
from mechanistic_ode import MechanisticODE, MeasurementModel
import priors as P


ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"
FIG_DIR = ROOT / "figures"
SCHEMA_VERSION = 2


@dataclass
class TrainConfig:
    dynamics: str = "graybox"             # physics_only | residual_only | graybox
    epochs: int = 240
    patience: int = 50
    lr: float = 3e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    endpoint_weight: float = 1.0
    reconstruction_weight: float = 0.20
    method: str = "dopri5"
    rtol: float = 1e-5
    atol: float = 1e-7
    adjoint_rtol: float = 1e-5
    adjoint_atol: float = 1e-7
    window_s: float = 60.0
    prefix_min_s: float = 20.0
    seed: int = 0
    encoder_hidden: int = 16
    tag: str = "graybox"

    @property
    def physics_only(self) -> bool:
        return self.dynamics == "physics_only"


class LatentStateEncoder(nn.Module):
    """Reverse-time GRU recognition model for ``C_f(0)``.

    ``L(0)=5 mm`` and ``C_b(0)=0`` are fixed by the experimental definition;
    only the unknown initial free concentration is inferred.
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.gru = nn.GRU(input_size=4, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -0.2)
        self.register_buffer("I_mean", torch.tensor(0.0))
        self.register_buffer("I_std", torch.tensor(1.0))
        self.register_buffer("T_mean", torch.tensor(30.0))
        self.register_buffer("T_std", torch.tensor(1.0))
        self.register_buffer("RH_mean", torch.tensor(55.0))
        self.register_buffer("RH_std", torch.tensor(1.0))
        self.register_buffer("cf_scale", torch.tensor(10.0))

    @torch.no_grad()
    def fit_normalization(self, inputs: ForecastInputs) -> None:
        self.I_mean.copy_(inputs.I_early.mean())
        self.I_std.copy_(inputs.I_early.std().clamp_min(1e-6))
        self.T_mean.copy_(inputs.T_ambient.mean())
        self.T_std.copy_(inputs.T_ambient.std().clamp_min(1e-6))
        self.RH_mean.copy_(inputs.RH.mean())
        self.RH_std.copy_(inputs.RH.std().clamp_min(1e-6))

    def sequence(self, inputs: ForecastInputs) -> Tensor:
        n, w = inputs.I_early.shape
        intensity = (inputs.I_early - self.I_mean) / self.I_std
        t_scale = inputs.t[-1].clamp_min(1.0)
        time = (inputs.t / t_scale).view(1, w).expand(n, -1)
        temp = ((inputs.T_ambient - self.T_mean) / self.T_std).view(n, 1).expand(-1, w)
        rh = ((inputs.RH - self.RH_mean) / self.RH_std).view(n, 1).expand(-1, w)
        return torch.stack([intensity, time, temp, rh], dim=-1)

    def forward(self, inputs: ForecastInputs) -> Tensor:
        # Chen-style recognition consumes observations in reverse temporal order.
        seq = torch.flip(self.sequence(inputs), dims=[1])
        _, h = self.gru(seq)
        return torch.nn.functional.softplus(self.head(h[-1]).squeeze(-1)) * self.cf_scale


class LatentODEForecaster(nn.Module):
    """Recognition encoder + conservative latent ODE + separate optics."""

    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = LatentStateEncoder(cfg.encoder_hidden)
        self.core = MechanisticODE.identifiable(dynamics=cfg.dynamics)
        self.measurement = MeasurementModel(alpha=2.0, beta=25.0)
        self.register_buffer("target_mean", torch.tensor(0.0))
        self.register_buffer("target_std", torch.tensor(1.0))

    def initial_state(self, inputs: ForecastInputs) -> Tensor:
        cf0 = self.encoder(inputs)
        return torch.stack(
            [torch.full_like(cf0, P.L0_MM), cf0, torch.zeros_like(cf0)], dim=-1
        )

    def integrate(
        self,
        inputs: ForecastInputs,
        t_eval: Tensor,
        *,
        use_adjoint: bool,
    ) -> Tuple[Tensor, Tensor]:
        z0 = self.initial_state(inputs)
        self.core.set_covariate_tensors(inputs.T_ambient, inputs.RH)
        solver = odeint_adjoint if use_adjoint else odeint
        kwargs = dict(
            method=self.cfg.method,
            rtol=self.cfg.rtol,
            atol=self.cfg.atol,
        )
        if use_adjoint:
            kwargs.update(
                adjoint_rtol=self.cfg.adjoint_rtol,
                adjoint_atol=self.cfg.adjoint_atol,
            )
        z = solver(self.core, z0, t_eval, **kwargs)
        return z, self.measurement(z[..., 2])

    def forward(self, inputs: ForecastInputs, t_eval: Tensor, use_adjoint: bool = True) -> Tensor:
        return self.integrate(inputs, t_eval, use_adjoint=use_adjoint)[1]

    @torch.no_grad()
    def predict(self, inputs: ForecastInputs, t_end: float = 900.0) -> np.ndarray:
        self.eval()
        t_eval = torch.tensor([0.0, float(t_end)], dtype=inputs.t.dtype)
        _, intensity = self.integrate(inputs, t_eval, use_adjoint=False)
        return as_numpy(intensity[-1])

    @torch.no_grad()
    def infer_cf0(self, inputs: ForecastInputs) -> np.ndarray:
        self.eval()
        return as_numpy(self.encoder(inputs))

    def validate_science(self) -> None:
        self.core.validate_frozen_priors()
        if self.core.cfg.dynamics == "physics_only":
            if any(p.requires_grad for p in self.core.residual.parameters()):
                raise RuntimeError("physics-only residual is trainable")


def count_trainable(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


@torch.no_grad()
def fit_normalization(model: LatentODEForecaster, train_bundle: D.TraceBundle, cfg: TrainConfig) -> None:
    inputs = inputs_from_bundle(train_bundle, cfg.window_s)
    model.encoder.fit_normalization(inputs)
    target = train_bundle.true_I_900.float()
    model.target_mean.copy_(target.mean())
    model.target_std.copy_(target.std().clamp_min(1e-6))
    model.measurement.beta.copy_(inputs.I_early[:, 0].mean())


def _prefix_for_epoch(bundle: D.TraceBundle, cfg: TrainConfig, epoch: int) -> ForecastInputs:
    choices = [x for x in (20.0, 30.0, 40.0, 50.0, 60.0) if cfg.prefix_min_s <= x <= cfg.window_s]
    window = choices[epoch % len(choices)] if choices else cfg.window_s
    return inputs_from_bundle(bundle, window)


def _loss(
    model: LatentODEForecaster,
    bundle: D.TraceBundle,
    inputs: ForecastInputs,
    *,
    use_adjoint: bool,
) -> Tuple[Tensor, Dict[str, float]]:
    t_eval = torch.cat([inputs.t, torch.tensor([900.0], dtype=inputs.t.dtype)])
    _, pred = model.integrate(inputs, t_eval, use_adjoint=use_adjoint)
    endpoint = torch.mean(((pred[-1] - bundle.true_I_900) / model.target_std) ** 2)
    reconstruction = torch.mean(((pred[:-1].transpose(0, 1) - inputs.I_early) / model.encoder.I_std) ** 2)
    total = model.cfg.endpoint_weight * endpoint + model.cfg.reconstruction_weight * reconstruction
    return total, {
        "endpoint": float(endpoint.detach()),
        "reconstruction": float(reconstruction.detach()),
    }


@torch.no_grad()
def evaluate_loss(model: LatentODEForecaster, bundle: D.TraceBundle, window_s: float = 60.0) -> float:
    inputs = inputs_from_bundle(bundle, window_s)
    loss, _ = _loss(model, bundle, inputs, use_adjoint=False)
    return float(loss)


def train(train_bundle: D.TraceBundle, validation_bundle: D.TraceBundle, cfg: TrainConfig) -> Dict:
    """Fit on ``train`` and select only on a disjoint validation split."""
    torch.manual_seed(cfg.seed)
    model = LatentODEForecaster(cfg)
    fit_normalization(model, train_bundle, cfg)
    model.validate_science()

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, min_lr=1e-5)
    best_value = float("inf")
    best_state: Optional[Dict[str, Tensor]] = None
    best_epoch = -1
    stale = 0
    history = {"train": [], "validation": [], "grad_norm": []}
    all_finite = True

    for epoch in range(cfg.epochs):
        model.train()
        inputs = _prefix_for_epoch(train_bundle, cfg, epoch)
        optimizer.zero_grad()
        loss, _ = _loss(model, train_bundle, inputs, use_adjoint=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, cfg.grad_clip)
        if not torch.isfinite(norm):
            all_finite = False
            raise FloatingPointError(f"non-finite gradient at epoch {epoch}")
        optimizer.step()
        model.validate_science()

        validation = evaluate_loss(model, validation_bundle, cfg.window_s)
        scheduler.step(validation)
        history["train"].append(float(loss.detach()))
        history["validation"].append(validation)
        history["grad_norm"].append(float(norm.detach()))
        if validation < best_value - 1e-8:
            best_value = validation
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.validate_science()
    return {
        "model": model,
        "cfg": cfg,
        "history": history,
        "best_validation": best_value,
        "best_epoch": best_epoch,
        "epochs_run": len(history["train"]),
        "grads_all_finite": all_finite,
    }


def save_checkpoint(result: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model: LatentODEForecaster = result["model"]
    model.validate_science()
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "prior_signature": P.prior_signature(),
            "cfg": asdict(result["cfg"]),
            "state_dict": model.state_dict(),
            "metrics": {
                "best_validation": result["best_validation"],
                "best_epoch": result["best_epoch"],
                "epochs_run": result["epochs_run"],
            },
        },
        path,
    )


def load_checkpoint(path: Path) -> Tuple[LatentODEForecaster, TrainConfig, Dict]:
    checkpoint = torch.load(path, weights_only=False)
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("legacy/no-encoder checkpoint rejected")
    if checkpoint.get("prior_signature") != P.prior_signature():
        raise RuntimeError("checkpoint prior signature mismatch")
    cfg = TrainConfig(**checkpoint["cfg"])
    model = LatentODEForecaster(cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.validate_science()
    model.eval()
    return model, cfg, checkpoint.get("metrics", {})


def information_invariance(model: LatentODEForecaster, bundle: D.TraceBundle) -> Dict[str, float]:
    """Runtime proof that metadata/post-window values cannot affect predictions."""
    inputs = inputs_from_bundle(bundle, model.cfg.window_s)
    baseline = model.predict(inputs)
    # The model never receives a bundle, so metadata mutation is structurally absent.
    late_mutated = copy.copy(bundle)
    late_mutated.I_obs = bundle.I_obs.clone()
    late_mutated.I_obs[:, bundle.t > model.cfg.window_s] += 10_000.0
    after_late = model.predict(inputs_from_bundle(late_mutated, model.cfg.window_s))
    early = ForecastInputs(inputs.t, torch.flip(inputs.I_early, dims=[1]), inputs.T_ambient, inputs.RH)
    after_early = model.predict(early)
    return {
        "late_window_delta": float(np.max(np.abs(baseline - after_late))),
        "early_window_delta": float(np.max(np.abs(baseline - after_early))),
    }


if __name__ == "__main__":
    model = LatentODEForecaster(TrainConfig())
    model.validate_science()
    print(f"latent-ODE trainable parameters: {count_trainable(model)}")
    print("Input contract: ForecastInputs(t, I_early, T_ambient, RH); no concentration metadata.")
