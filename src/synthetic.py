"""Honest synthetic pipeline-validation study for KinetiFlow-CP.

The simulator is not evidence that the planted non-ideality exists in real hCG
strips.  It provides controlled null, in-family, and out-of-family cases for
testing whether the software can recover or reject known mismatches.

Primary concentration levels remain in the clinical range (0--125 mIU/mL).
Hook-effect simulations are intentionally excluded because the documented hook
bracket begins hundreds of times higher; hook claims require a separate high-dose
dilution study on hardware.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torchdiffeq import odeint

import priors as P


@dataclass
class SyntheticConfig:
    nominal_mIU_per_mL: float = 25.0
    conc_levels: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0)
    n_groups: int = 20
    reps_per_level: int = 3
    group_id_offset: int = 0
    T_nominal: float = 30.0
    T_between_sd: float = 1.5
    T_within_sd: float = 0.5
    RH_nominal: float = 55.0
    RH_between_sd: float = 4.0
    RH_within_sd: float = 2.0
    group_effect_scale: float = 1.0
    mismatch_scale: float = 1.0
    mismatch_mechanism: str = "combined"  # none | combined | supply
    alpha_true: float = 20.0              # DN per nmol/m^2
    beta_true: float = 25.0
    noise_std_dn: float = 0.40
    t_end: float = 900.0
    n_t: int = 91
    early_window_s: float = 60.0
    seed: int = 12345


class TruthODE(nn.Module):
    """Independent virtual-assay dynamics in the production state units."""

    def __init__(self, cfg: SyntheticConfig):
        super().__init__()
        values = P.production_values()
        self.cfg = cfg
        self.k_wash = values["k_wash"]
        self.k_on = values["k_on"]
        self.k_off = values["k_off"]
        self.B_max = values["B_max"]
        self.area = values["capture_area_m2"]
        self.volume0 = values["volume_at_L0_mL"]
        self.cross_section = values["pore_cross_section_mm2"]
        self.T = cfg.T_nominal
        self.RH = cfg.RH_nominal
        self.lot_on = 1.0
        self.lot_wash = 1.0
        self.feed_cf = 0.0

    def set_context(
        self, T: float, RH: float, lot_on: float, lot_wash: float, feed_cf: float
    ) -> None:
        self.T, self.RH = T, RH
        self.lot_on, self.lot_wash = lot_on, lot_wash
        self.feed_cf = feed_cf

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        L, C_f, C_b = z.unbind(dim=-1)
        dL = (self.k_wash * self.lot_wash) / L
        volume = self.volume0 + self.cross_section * 1e-3 * (L - P.L0_MM).clamp_min(0.0)
        dvolume = self.cross_section * 1e-3 * dL
        cf = C_f.clamp_min(0.0)
        occupancy = (C_b / self.B_max).clamp(0.0, 1.0)
        cb = occupancy * self.B_max

        scale = self.cfg.mismatch_scale
        if self.cfg.mismatch_mechanism == "none":
            scale = 0.0
        # Prespecified in-family mismatch.  It is intentionally smooth and
        # learnable; results can validate only pipeline recovery, not biology.
        log_on = scale * (0.60 * occupancy + 0.15 * (self.T - 30.0) / 10.0)
        log_off = scale * (-0.30 * occupancy)
        j_on = self.k_on * self.lot_on * cf * (self.B_max - cb) * torch.exp(log_on)
        j_off = self.k_off * cb * torch.exp(log_off)
        flux = j_on - j_off
        coupling = self.area * P.MW_HCG_G_PER_MOL / volume
        dC_f = -(dvolume / volume) * C_f - coupling * flux
        dC_b = flux

        if self.cfg.mismatch_mechanism == "supply" and scale != 0.0:
            # Out-of-family replenishment: a feed term absent from the learner.
            # This declared open-system flux is not incorrectly called conserved.
            rate = scale * 2e-3 * torch.exp(-t / 180.0)
            dC_f = dC_f + rate * (self.feed_cf - C_f)
        return torch.stack([dL, dC_f, dC_b], dim=-1)


def _group_effects(cfg: SyntheticConfig, generator: torch.Generator, group: int) -> Dict[str, float]:
    del group
    scale = cfg.group_effect_scale
    normal = lambda sd: float(torch.randn((), generator=generator)) * sd * scale
    return {
        "T_mean": cfg.T_nominal + normal(cfg.T_between_sd),
        "RH_mean": cfg.RH_nominal + normal(cfg.RH_between_sd),
        "alpha_mult": float(torch.exp(torch.tensor(normal(0.05)))),
        "beta_offset": normal(0.8),
        "lot_on": float(torch.exp(torch.tensor(normal(0.10)))),
        "lot_wash": float(torch.exp(torch.tensor(normal(0.08)))),
        "noise_mult": float(torch.exp(torch.tensor(normal(0.12)))),
    }


def generate(cfg: SyntheticConfig | None = None) -> Dict[str, Tensor]:
    cfg = cfg or SyntheticConfig()
    if cfg.mismatch_mechanism not in {"none", "combined", "supply"}:
        raise ValueError(f"unknown mismatch mechanism: {cfg.mismatch_mechanism}")
    generator = torch.Generator().manual_seed(cfg.seed)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        truth = TruthODE(cfg)
        time = torch.linspace(0.0, cfg.t_end, cfg.n_t)
        traces: List[Tensor] = []
        clean: List[Tensor] = []
        fields: Dict[str, list] = {
            key: [] for key in (
                "C_f0", "concentration_level", "day", "lot", "group_id",
                "T_ambient", "RH", "true_I_900", "alpha_group", "beta_group",
            )
        }

        for local_group in range(cfg.n_groups):
            group = cfg.group_id_offset + local_group
            effects = _group_effects(cfg, generator, group)
            for level in cfg.conc_levels:
                cf0 = P.mIU_per_mL_to_ng_per_mL(level * cfg.nominal_mIU_per_mL)
                for _ in range(cfg.reps_per_level):
                    T = effects["T_mean"] + float(torch.randn((), generator=generator)) * cfg.T_within_sd
                    RH = effects["RH_mean"] + float(torch.randn((), generator=generator)) * cfg.RH_within_sd
                    T = max(18.0, min(42.0, T))
                    RH = max(25.0, min(85.0, RH))
                    truth.set_context(T, RH, effects["lot_on"], effects["lot_wash"], cf0)
                    z0 = torch.tensor([P.L0_MM, cf0, 0.0])
                    z = odeint(truth, z0, time, method="dopri5", rtol=1e-8, atol=1e-10)
                    alpha = cfg.alpha_true * effects["alpha_mult"]
                    beta = cfg.beta_true + effects["beta_offset"]
                    signal = alpha * z[:, 2] + beta
                    noise = cfg.noise_std_dn * effects["noise_mult"] * torch.randn(
                        signal.shape, generator=generator
                    )
                    traces.append(signal + noise)
                    clean.append(signal)
                    fields["C_f0"].append(cf0)
                    fields["concentration_level"].append(level)
                    fields["day"].append(group)       # unique day/lot block
                    fields["lot"].append(group)
                    fields["group_id"].append(group)
                    fields["T_ambient"].append(T)
                    fields["RH"].append(RH)
                    fields["true_I_900"].append(float(signal[-1]))
                    fields["alpha_group"].append(alpha)
                    fields["beta_group"].append(beta)

        data: Dict[str, Tensor] = {
            "t": time.float(),
            "I_obs": torch.stack(traces).float(),
            "I_clean": torch.stack(clean).float(),
            "noise_std": torch.tensor(cfg.noise_std_dn, dtype=torch.float32),
        }
        for key, values in fields.items():
            dtype = torch.long if key in {"day", "lot", "group_id"} else torch.float32
            data[key] = torch.tensor(values, dtype=dtype)
        return data
    finally:
        torch.set_default_dtype(previous_dtype)


def save(data: Dict[str, Tensor], out_dir: str | Path, cfg: SyntheticConfig | None = None) -> Path:
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "traces.pt"
    torch.save(data, path)
    columns = [
        "concentration_level", "day", "lot", "group_id", "T_ambient", "RH",
        "C_f0", "true_I_900", "alpha_group", "beta_group",
    ]
    lines = ["idx," + ",".join(columns)]
    for i in range(data["I_obs"].shape[0]):
        lines.append(str(i) + "," + ",".join(f"{float(data[c][i]):.8g}" for c in columns))
    (out / "metadata.csv").write_text("\n".join(lines) + "\n")
    if cfg is not None:
        import json
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    return path


def early_window_separation(data: Dict[str, Tensor], window_s: float = 60.0) -> Tuple[float, float, float]:
    mask = data["t"] <= window_s + 1e-6
    means = data["I_obs"][:, mask].mean(dim=1)
    levels = data["concentration_level"]
    group_means, within = [], []
    for level in torch.unique(levels):
        values = means[levels == level]
        group_means.append(values.mean())
        within.append(values.std())
    between = float(torch.stack(group_means).std())
    within_mean = float(torch.stack(within).mean())
    return between, within_mean, between / max(within_mean, 1e-12)


if __name__ == "__main__":
    config = SyntheticConfig()
    generated = generate(config)
    save(generated, Path(__file__).resolve().parents[1] / "data" / "synthetic", config)
    between, within, ratio = early_window_separation(generated)
    print(f"generated {len(generated['I_obs'])} traces in {config.n_groups} day/lot blocks")
    print(f"early separation: between={between:.3f}, within={within:.3f}, ratio={ratio:.2f}")
    print("SYNTHETIC PIPELINE ONLY — no real-assay scientific claim")
