"""
synthetic.py
============
Synthetic lateral-flow-assay (LFA) data generator for KinetiFlow-CP v2.

Ground truth is the KNOWN-physics model in mechanistic_ode.py with the neural
residual set to zero (pure Lucas-Washburn + Langmuir), read out through the
separate optical MeasurementModel:

    I_obs(t) = alpha * C_b(t) + beta + Gaussian noise

On top of the known physics we inject a SMALL, KNOWN nonlinear perturbation into
the chemistry channels (C_f, C_b) only — a cooperative binding term, a high-dose
hook-like suppression at high analyte, and a mild temperature modulation of the
on-rate. This is deliberate MODEL MISMATCH: the hook term acts differently across
concentrations, so a pure-physics fit (Baseline A) cannot reproduce it by
rescaling {alpha, beta}, and the neural residual has a real, systematic signal to
learn. Mass is conserved (whatever is added to dC_b is removed from dC_f),
exactly like the true physics.

Experimental structure (needed for leakage-safe grouped splits in dataset.py):
  * 5 concentration levels: 0x, 0.5x, 1x, 2x, 5x of a nominal mIU/mL threshold.
  * days and lots are laid out in three DISJOINT blocks so that no day and no lot
    is shared between the train / calibration / test partitions dataset.py will
    later recover. Each trace carries (concentration_level, day, lot, T_ambient,
    RH, true 900 s equilibrium intensity).

Output (under data/synthetic/):
  * traces.pt  : a dict of tensors (t grid, I_obs, clean signal, metadata).
  * metadata.csv : the same per-trace metadata in tidy tabular form.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torchdiffeq import odeint

import priors as P
from mechanistic_ode import MechanisticODE, MeasurementModel


# --------------------------------------------------------------------------- #
#  Configuration                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class SyntheticConfig:
    # --- concentrations ---
    nominal_mIU_per_mL: float = 25.0                 # clinical hCG threshold scale
    conc_levels: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0)

    # --- optics (true values the trainer must recover; larger alpha than the
    #     identifiability nominal so the line spans a realistic DN range) ---
    alpha_true: float = 2000.0                       # DN per ug/cm^2
    beta_true: float = 25.0                          # baseline DN
    noise_frac: float = 0.013                        # noise std as frac of dyn. range

    # --- known nonlinear model-mismatch perturbation (chemistry only) ---
    #  extra_bind = base_bind * (temp_fac + coop_gain*C_b - hook_gain*C_f): a
    #  bounded modulation of the physical binding rate. It multiplies base_bind
    #  (which vanishes at C_b -> B_max), so C_b stays admissible. The hook term
    #  -hook_gain*C_f SUPPRESSES binding at high analyte (a high-dose-hook-like
    #  effect); because it acts differently across concentrations, a single
    #  {alpha, beta} rescale of pure physics CANNOT absorb it -> the residual has
    #  a real, non-trivial signal to learn.
    coop_gain: float = 4.0         # cooperative boost, per ug/cm^2 of C_b
    hook_gain: float = 0.06        # high-dose hook suppression, per ng/mL of C_f
    temp_coeff: float = 0.10       # fractional on-rate change per +10 C above 30 C

    # --- environmental covariates (sampled per trace) ---
    T_nominal: float = 30.0
    T_std: float = 2.0
    RH_nominal: float = 55.0
    RH_std: float = 6.0

    # --- integration grid ---
    t_end: float = 900.0
    n_t: int = 91                                    # 10 s resolution
    early_window_s: float = 60.0
    L0: float = 5.0                                  # front at read window (>0!)

    # --- experimental layout: DISJOINT day/lot blocks per intended split ---
    #  (dataset.py rediscovers these blocks from the data; it does not trust them)
    train_days: Tuple[int, ...] = (0, 1, 2, 3)
    cal_days: Tuple[int, ...] = (4, 5)
    test_days: Tuple[int, ...] = (6, 7)
    train_lots: Tuple[int, ...] = (0, 1, 2)
    cal_lots: Tuple[int, ...] = (3,)
    test_lots: Tuple[int, ...] = (4,)
    train_reps: int = 2            # reps per (day, lot, level) in the train block
    cal_reps: int = 4
    test_reps: int = 4

    seed: int = 12345


# --------------------------------------------------------------------------- #
#  Ground-truth dynamics = known physics + known perturbation                 #
# --------------------------------------------------------------------------- #
class TruthODE(nn.Module):
    """Wraps a frozen, residual-free MechanisticODE (the known physics) and adds a
    small KNOWN nonlinear perturbation to the chemistry channels only."""

    def __init__(self, core: MechanisticODE, coop_gain: float, hook_gain: float,
                 temp_coeff: float):
        super().__init__()
        self.core = core
        self.coop_gain = float(coop_gain)
        self.hook_gain = float(hook_gain)
        self.temp_coeff = float(temp_coeff)

    def set_covariates(self, T_ambient: float, RH: float) -> None:
        self.core.set_covariates(T_ambient, RH)

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        d = self.core(t, z)                       # pure physics (residual == 0)
        L, C_f, C_b = z.unbind(dim=-1)

        # Known model-mismatch, applied as extra (de)binding so mass is conserved.
        # Modulates the physical binding rate; vanishes as C_b -> B_max so the
        # trajectory stays admissible.
        base_bind = self.core.k_on * C_f * (self.core.B_max - C_b)
        temp_fac = self.temp_coeff * (self.core.T_ambient - 30.0) / 10.0
        coop = self.coop_gain * C_b               # cooperative acceleration
        hook = self.hook_gain * C_f               # high-dose hook suppression
        extra_bind = base_bind * (temp_fac + coop - hook)

        dL = d[..., 0]
        dC_f = d[..., 1] - extra_bind
        dC_b = d[..., 2] + extra_bind
        return torch.stack([dL, dC_f, dC_b], dim=-1)


def build_truth() -> Tuple[TruthODE, MeasurementModel, SyntheticConfig]:
    """Convenience builder used by tests: (truth dynamics, optics, default cfg)."""
    cfg = SyntheticConfig()
    core = _residual_free_core()
    truth = TruthODE(core, cfg.coop_gain, cfg.hook_gain, cfg.temp_coeff)
    meas = MeasurementModel(alpha=cfg.alpha_true, beta=cfg.beta_true)
    return truth, meas, cfg


def _residual_free_core() -> MechanisticODE:
    """Known-physics core: identifiable-mode model with the residual zeroed out."""
    core = MechanisticODE.identifiable()
    with torch.no_grad():
        for p in core.residual.parameters():
            p.zero_()
    core.freeze_residual(True)
    return core


# --------------------------------------------------------------------------- #
#  Generation                                                                 #
# --------------------------------------------------------------------------- #
def _iter_layout(cfg: SyntheticConfig):
    """Yield (day, lot, reps) for each disjoint block. Days and lots never cross
    blocks, guaranteeing dataset.py can split with zero day/lot leakage."""
    blocks = [
        (cfg.train_days, cfg.train_lots, cfg.train_reps),
        (cfg.cal_days, cfg.cal_lots, cfg.cal_reps),
        (cfg.test_days, cfg.test_lots, cfg.test_reps),
    ]
    for days, lots, reps in blocks:
        for day in days:
            for lot in lots:
                yield day, lot, reps


def generate(cfg: SyntheticConfig | None = None) -> Dict[str, Tensor]:
    """Generate the full synthetic dataset. Returns a dict of tensors."""
    cfg = cfg or SyntheticConfig()
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)          # accurate ground-truth solve
    try:
        gen = torch.Generator().manual_seed(cfg.seed)
        core = _residual_free_core()
        truth = TruthODE(core, cfg.coop_gain, cfg.hook_gain, cfg.temp_coeff)
        meas = MeasurementModel(alpha=cfg.alpha_true, beta=cfg.beta_true)
        for p in meas.parameters():
            p.requires_grad_(False)                 # data-gen: no gradients needed

        t = torch.linspace(0.0, cfg.t_end, cfg.n_t)

        # Dynamic range: 0x baseline vs the strongest level at equilibrium.
        dyn_range = _dynamic_range(truth, meas, cfg, t)
        noise_std = cfg.noise_frac * dyn_range

        I_list: List[Tensor] = []
        Iclean_list: List[Tensor] = []
        cf0_list, lvl_list, day_list, lot_list = [], [], [], []
        T_list, RH_list, y900_list = [], [], []

        for day, lot, reps in _iter_layout(cfg):
            for lvl in cfg.conc_levels:
                mIU = lvl * cfg.nominal_mIU_per_mL
                cf0 = P.mIU_per_mL_to_ng_per_mL(mIU)
                for _ in range(reps):
                    T_amb = cfg.T_nominal + cfg.T_std * torch.randn((), generator=gen)
                    RH = cfg.RH_nominal + cfg.RH_std * torch.randn((), generator=gen)
                    T_amb = float(T_amb.clamp(24.0, 36.0))
                    RH = float(RH.clamp(35.0, 75.0))

                    truth.set_covariates(T_amb, RH)
                    z0 = torch.tensor([cfg.L0, cf0, 0.0])
                    z = odeint(truth, z0, t, method="dopri5", rtol=1e-7, atol=1e-9)
                    I_clean = meas(z[..., 2])
                    noise = noise_std * torch.randn(I_clean.shape, generator=gen)
                    I_obs = I_clean + noise

                    I_list.append(I_obs)
                    Iclean_list.append(I_clean)
                    cf0_list.append(cf0)
                    lvl_list.append(lvl)
                    day_list.append(day)
                    lot_list.append(lot)
                    T_list.append(T_amb)
                    RH_list.append(RH)
                    y900_list.append(float(I_clean[-1]))

        data = {
            "t": t.to(torch.float32),
            "I_obs": torch.stack(I_list).to(torch.float32),          # [N, T]
            "I_clean": torch.stack(Iclean_list).to(torch.float32),   # [N, T]
            "C_f0": torch.tensor(cf0_list, dtype=torch.float32),
            "concentration_level": torch.tensor(lvl_list, dtype=torch.float32),
            "day": torch.tensor(day_list, dtype=torch.long),
            "lot": torch.tensor(lot_list, dtype=torch.long),
            "T_ambient": torch.tensor(T_list, dtype=torch.float32),
            "RH": torch.tensor(RH_list, dtype=torch.float32),
            "true_I_900": torch.tensor(y900_list, dtype=torch.float32),
            "noise_std": torch.tensor(float(noise_std), dtype=torch.float32),
            "dyn_range": torch.tensor(float(dyn_range), dtype=torch.float32),
            "alpha_true": torch.tensor(cfg.alpha_true, dtype=torch.float32),
            "beta_true": torch.tensor(cfg.beta_true, dtype=torch.float32),
        }
        return data
    finally:
        torch.set_default_dtype(prev_dtype)


def _dynamic_range(truth: TruthODE, meas: MeasurementModel,
                   cfg: SyntheticConfig, t: Tensor) -> float:
    """Equilibrium DN span between the 0x and the strongest level (nominal covs)."""
    truth.set_covariates(cfg.T_nominal, cfg.RH_nominal)
    top = max(cfg.conc_levels)
    spans = []
    for lvl in (0.0, top):
        cf0 = P.mIU_per_mL_to_ng_per_mL(lvl * cfg.nominal_mIU_per_mL)
        z0 = torch.tensor([cfg.L0, cf0, 0.0])
        z = odeint(truth, z0, t, method="dopri5", rtol=1e-7, atol=1e-9)
        spans.append(float(meas(z[-1, 2])))
    return abs(spans[1] - spans[0])


# --------------------------------------------------------------------------- #
#  Persistence                                                                #
# --------------------------------------------------------------------------- #
def save(data: Dict[str, Tensor], out_dir: str | Path) -> Path:
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    pt_path = out / "traces.pt"
    torch.save(data, pt_path)

    # Tidy per-trace metadata CSV (no trace columns; those live in traces.pt).
    csv_path = out / "metadata.csv"
    cols = ["concentration_level", "day", "lot", "T_ambient", "RH",
            "C_f0", "true_I_900"]
    N = data["I_obs"].shape[0]
    lines = ["idx," + ",".join(cols)]
    for i in range(N):
        row = [str(i)] + [f"{float(data[c][i]):.6g}" for c in cols]
        lines.append(",".join(row))
    csv_path.write_text("\n".join(lines) + "\n")
    return pt_path


# --------------------------------------------------------------------------- #
#  Sanity check: early-window concentration separation                        #
# --------------------------------------------------------------------------- #
def early_window_separation(data: Dict[str, Tensor], early_window_s: float = 60.0):
    """Per-trace mean of I_obs over 0..early_window_s, grouped by concentration.

    Returns (between_level_spread, within_level_noise, ratio, per_level_means).
    between_level_spread = std of the per-level mean early-window intensities.
    within_level_noise   = mean over levels of the within-level std.
    """
    t = data["t"]
    mask = t <= early_window_s
    early_mean = data["I_obs"][:, mask].mean(dim=1)          # [N]
    levels = data["concentration_level"]

    uniq = sorted(set(float(x) for x in levels.tolist()))
    level_means, within_stds = [], []
    for lv in uniq:
        sel = levels == lv
        vals = early_mean[sel]
        level_means.append(float(vals.mean()))
        within_stds.append(float(vals.std(unbiased=False)))

    between = float(torch.tensor(level_means).std(unbiased=False))
    within = float(torch.tensor(within_stds).mean())
    ratio = between / within if within > 0 else float("inf")
    return between, within, ratio, dict(zip(uniq, level_means))


# --------------------------------------------------------------------------- #
#  Main: generate, save, and print the pilot separation criterion             #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    cfg = SyntheticConfig()
    data = generate(cfg)

    out_dir = Path(__file__).resolve().parents[1] / "data" / "synthetic"
    pt_path = save(data, out_dir)

    N = data["I_obs"].shape[0]
    n_days = len(set(data["day"].tolist()))
    n_lots = len(set(data["lot"].tolist()))
    print("== synthetic LFA dataset ==")
    print(f"traces           : {N}")
    print(f"timepoints       : {data['t'].numel()} over 0..{cfg.t_end:.0f} s")
    print(f"days / lots       : {n_days} / {n_lots}")
    print(f"dynamic range    : {float(data['dyn_range']):.2f} DN "
          f"(alpha={cfg.alpha_true:.0f}, beta={cfg.beta_true:.0f})")
    print(f"noise std        : {float(data['noise_std']):.3f} DN "
          f"({cfg.noise_frac*100:.0f}% of dyn. range)")
    print(f"saved            : {pt_path}")

    between, within, ratio, means = early_window_separation(data, cfg.early_window_s)
    print("\n== early-window (0-60 s) concentration separation ==")
    for lv, m in means.items():
        print(f"   level {lv:>4}x : mean early I = {m:8.3f} DN")
    print(f"\nbetween-level spread : {between:.3f} DN")
    print(f"within-level noise   : {within:.3f} DN")
    print(f"ratio (want > 3x)    : {ratio:.2f}x")
    print("SEPARATION:", "PASS" if ratio > 3.0 else "FAIL")
