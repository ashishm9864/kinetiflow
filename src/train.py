"""
train.py
========
Minimal, correct training loop for the KinetiFlow-CP v2 gray-box UDE on the
synthetic dataset.

The model is built in IDENTIFIABLE TRAINING MODE (mechanistic_ode.py): the
kinetic constants k_wash/k_on/k_off/B_max are FROZEN at the priors.py values and
only {alpha, beta (optics), the neural residual} are learned — this is the
Risk-1 action proven necessary in identifiability.py.

Two configurations are supported:
  * gray-box   : the residual is trainable (the full model).
  * physics_only: the residual is ALSO frozen (Baseline A, the pure-physics
    ceiling). Only the optics {alpha, beta} adapt, so any gap to the gray-box is
    attributable to the neural residual.

Training details: continuous-adjoint integration (odeint_adjoint), Adam with
per-group learning rates, gradient clipping, early stopping on the calibration
split, best checkpoint saved to checkpoints/, and a train/cal loss-curve PNG.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torchdiffeq import odeint, odeint_adjoint

from mechanistic_ode import MechanisticODE, MeasurementModel
from loss_functions import kinetiflow_loss
import dataset as D


ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"
FIG_DIR = ROOT / "figures"

# --------------------------------------------------------------------------- #
#  Configuration                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    lam: float = 1.0                 # physics-bounds weight in kinetiflow_loss
    epochs: int = 180
    lr_residual: float = 5e-3        # initial residual lr (decayed on plateau)
    lr_optics: float = 0.3           # optics start near a data-driven init
    grad_clip: float = 1.0           # NaN/blow-up guard on the residual gradient
    sched_factor: float = 0.5        # ReduceLROnPlateau decay factor
    sched_patience: int = 10         # epochs of no cal improvement before decay
    min_lr: float = 1e-4
    patience: int = 50               # early stopping on cal loss
    physics_only: bool = False       # True => Baseline A (freeze residual too)
    residual_scale: float = 8e-4     # authority of the residual on dC_f/dC_b. The
                                     # physical rates are ~1e-4; scaling lets the
                                     # MLP learn well-conditioned O(1) signals
                                     # instead of the tiny raw corrections that
                                     # otherwise make the stiff adjoint explode.
    L0: float = 5.0
    rtol: float = 1e-3
    atol: float = 1e-5
    method: str = "dopri5"
    seed: int = 0
    tag: str = "graybox"


# --------------------------------------------------------------------------- #
#  Residual-scaled gray-box wrapper                                           #
# --------------------------------------------------------------------------- #
class GrayBoxUDE(torch.nn.Module):
    """Wrap a MechanisticODE and rescale the neural residual's authority.

    The residual adds directly to the chemistry rates dC_f/dC_b, whose physical
    magnitude is ~1e-4. An MLP naturally emits O(1) outputs, so the RAW residual
    dwarfs the physics and makes the stiff 900 s adjoint blow up during training.
    We instead let the MLP learn a well-conditioned O(1) signal and multiply its
    contribution by `residual_scale` (~1e-4) inside the vector field:

        dz/dt = physics(z) + residual_scale * f_theta(z)

    Implemented as (full = physics + raw_delta from the core):
        physics + scale*delta = full - (1 - scale) * delta
    which reuses BOTH the core's physics and its residual net unchanged.
    """

    def __init__(self, core: MechanisticODE, residual_scale: float):
        super().__init__()
        self.core = core
        self.residual_scale = float(residual_scale)

    # pass-throughs used by the trainer / losses
    @property
    def B_max(self) -> Tensor:
        return self.core.B_max

    @property
    def residual(self) -> torch.nn.Module:
        return self.core.residual

    def set_covariate_tensors(self, T_ambient: Tensor, RH: Tensor) -> None:
        self.core.T_ambient = T_ambient.detach()
        self.core.RH = RH.detach()

    def _raw_delta(self, t: Tensor, z: Tensor) -> Tensor:
        _, C_f, C_b = z.unbind(dim=-1)
        ones = torch.ones_like(C_f)
        raw = torch.stack(
            [C_f, C_b, ones * t, ones * self.core.T_ambient, ones * self.core.RH],
            dim=-1,
        )
        res_in = (raw - self.core.res_mean) / self.core.res_scale
        return self.core.residual(res_in)                      # [..., 2]

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        full = self.core(t, z)                                 # physics + raw delta
        delta = self._raw_delta(t, z)
        back = (1.0 - self.residual_scale) * delta             # remove all but scale
        dL = full[..., 0]
        dC_f = full[..., 1] - back[..., 0]
        dC_b = full[..., 2] - back[..., 1]
        return torch.stack([dL, dC_f, dC_b], dim=-1)


# --------------------------------------------------------------------------- #
#  Batch assembly + integration                                               #
# --------------------------------------------------------------------------- #
def make_batch(bundle: "D.TraceBundle", L0: float
               ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return (z0[B,3], t[T], I_target[T,B], T_ambient[B], RH[B])."""
    B = len(bundle)
    z0 = torch.stack([
        torch.full((B,), float(L0)),
        bundle.C_f0.clone(),
        torch.zeros(B),
    ], dim=-1)
    I_target = bundle.I_obs.transpose(0, 1).contiguous()   # [T, B]
    return z0, bundle.t, I_target, bundle.T_ambient.clone(), bundle.RH.clone()


def integrate_batch(model: GrayBoxUDE, meas: MeasurementModel,
                    z0: Tensor, t: Tensor, T_amb: Tensor, RH: Tensor,
                    cfg: TrainConfig, use_adjoint: bool) -> Tuple[Tensor, Tensor]:
    """Solve the batched UDE with per-trace covariates. Returns (z_traj, I_pred).

    Per-trace environmental covariates are written into the model's (no-grad)
    covariate buffers so the residual sees the right T/RH for each strip; the
    whole batch integrates on the shared time grid.
    """
    model.set_covariate_tensors(T_amb, RH)      # per-trace [B] covariate buffers
    solver = odeint_adjoint if use_adjoint else odeint
    z_traj = solver(model, z0, t, method=cfg.method, rtol=cfg.rtol, atol=cfg.atol)
    I_pred = meas(z_traj[..., 2])        # [T, B]
    return z_traj, I_pred


# --------------------------------------------------------------------------- #
#  Data-driven optics initialization (keeps optimization well-conditioned)    #
# --------------------------------------------------------------------------- #
def init_optics(model: MechanisticODE, meas: MeasurementModel,
                bundle: "D.TraceBundle", cfg: TrainConfig) -> None:
    """Least-squares init of {alpha, beta} from the physics-predicted C_b so the
    optics start at the right SCALE (C_b ~ 0.05 but intensity ~ tens of DN). This
    only sets the starting point; both remain trainable."""
    z0, t, I_target, T_amb, RH = make_batch(bundle, cfg.L0)
    with torch.no_grad():
        z, _ = integrate_batch(model, meas, z0, t, T_amb, RH, cfg, use_adjoint=False)
        C_b = z[..., 2].reshape(-1).double()        # [T*B]
        y = I_target.reshape(-1).double()           # [T*B]
        # Ordinary least squares y ~ alpha * C_b + beta. Solve in float64: C_b is
        # tiny (~0.05) next to the ones column, so the design matrix is badly
        # conditioned and a float32 solve returns garbage.
        A = torch.stack([C_b, torch.ones_like(C_b)], dim=1)   # [M, 2]
        sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution
        alpha0, beta0 = float(sol[0]), float(sol[1])
        meas.alpha.fill_(alpha0)
        meas.beta.fill_(beta0)


# --------------------------------------------------------------------------- #
#  Operating-range-matched residual input normalization                       #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _set_residual_norm(model: "GrayBoxUDE", meas: MeasurementModel,
                       bundle: "D.TraceBundle", cfg: TrainConfig,
                       eps: float = 1e-6) -> None:
    """Center/scale the residual inputs (C_f, C_b, t, T, RH) to their ACTUAL
    operating range, measured from a pure-physics rollout of the TRAIN split.

    The ODEConfig default normalization assumes C_b ~ 0.5, but the assay operates
    at C_b ~ 0-0.06, so C_b collapses to a near-constant feature and the residual
    is blind to the cooperative-term mismatch. Deriving mean/scale from data (not
    hardcoded constants) makes this generalize to real strips. This must be called
    while the residual output is still zeroed, so the rollout is pure physics and
    independent of the normalization being set. The stats are written into the
    core.res_mean/res_scale buffers, which travel with the checkpoint so inference
    uses the identical normalization."""
    z0, t, _, T_amb, RH = make_batch(bundle, cfg.L0)
    z, _ = integrate_batch(model, meas, z0, t, T_amb, RH, cfg, use_adjoint=False)
    C_f, C_b = z[..., 1], z[..., 2]                      # [T, B]
    feats = torch.stack([
        C_f, C_b,
        t.view(-1, 1).expand_as(C_f),
        T_amb.view(1, -1).expand_as(C_f),
        RH.view(1, -1).expand_as(C_f),
    ], dim=-1).reshape(-1, 5)                            # [T*B, 5]
    model.core.res_mean.copy_(feats.mean(0))
    model.core.res_scale.copy_(feats.std(0).clamp_min(eps))


# --------------------------------------------------------------------------- #
#  Evaluation                                                                 #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_loss(model: MechanisticODE, meas: MeasurementModel,
              bundle: "D.TraceBundle", cfg: TrainConfig
              ) -> Tuple[float, Dict[str, float], Tensor]:
    z0, t, I_target, T_amb, RH = make_batch(bundle, cfg.L0)
    z, I_pred = integrate_batch(model, meas, z0, t, T_amb, RH, cfg, use_adjoint=False)
    total, parts = kinetiflow_loss(I_pred, I_target, z, model.B_max, lam=cfg.lam)
    return float(total), parts, I_pred.detach()


# --------------------------------------------------------------------------- #
#  Training                                                                   #
# --------------------------------------------------------------------------- #
def train(train_bundle: "D.TraceBundle", cal_bundle: "D.TraceBundle",
          cfg: TrainConfig) -> Dict:
    torch.manual_seed(cfg.seed)

    core = MechanisticODE.identifiable()           # frozen kinetics, learn residual
    _zero_residual_output(core)                    # start exactly at pure physics
    model = GrayBoxUDE(core, cfg.residual_scale)
    meas = MeasurementModel(alpha=100.0, beta=20.0)
    if cfg.physics_only:
        core.freeze_residual(True)                 # Baseline A: pure-physics ceiling

    _set_residual_norm(model, meas, train_bundle, cfg)  # operating-range-matched norm
    init_optics(model, meas, train_bundle, cfg)    # scale-correct optics start

    params = [{"params": [p for p in model.residual.parameters() if p.requires_grad],
               "lr": cfg.lr_residual},
              {"params": [meas.alpha, meas.beta], "lr": cfg.lr_optics}]
    params = [g for g in params if len(g["params"]) > 0]
    opt = torch.optim.Adam(params)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=cfg.sched_factor,
        patience=cfg.sched_patience, min_lr=cfg.min_lr)

    z0, t, I_target, T_amb, RH = make_batch(train_bundle, cfg.L0)

    history = {"train": [], "cal": [], "train_obs": [], "cal_obs": []}
    grads_all_finite = True
    best_cal = float("inf")
    best_state: Optional[Dict] = None
    best_epoch = -1
    since_improve = 0

    # Epoch 0 reference (before any optimizer step).
    cal0, cal0_parts, _ = eval_loss(model, meas, cal_bundle, cfg)

    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        z, I_pred = integrate_batch(model, meas, z0, t, T_amb, RH, cfg, use_adjoint=True)
        total, parts = kinetiflow_loss(I_pred, I_target, z, model.B_max, lam=cfg.lam)
        total.backward()

        # gradient health + clipping
        trainable = [p for g in params for p in g["params"]]
        for p in trainable:
            if p.grad is not None and not torch.isfinite(p.grad).all():
                grads_all_finite = False
        torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
        opt.step()

        cal_total, cal_parts, _ = eval_loss(model, meas, cal_bundle, cfg)
        sched.step(cal_total)                          # decay residual/optics lr on plateau
        history["train"].append(float(total.detach()))
        history["cal"].append(cal_total)
        history["train_obs"].append(parts["obs"])
        history["cal_obs"].append(cal_parts["obs"])

        if cal_total < best_cal - 1e-9:
            best_cal = cal_total
            best_epoch = epoch
            best_state = {
                "model": copy.deepcopy(_portable_state(model)),
                "meas": copy.deepcopy(meas.state_dict()),
            }
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= cfg.patience:
                break

    return {
        "model": model, "meas": meas, "cfg": cfg,
        "history": history,
        "cal0": cal0, "cal0_obs": cal0_parts["obs"],
        "best_cal": best_cal, "best_epoch": best_epoch,
        "best_state": best_state,
        "grads_all_finite": grads_all_finite,
        "epochs_run": len(history["train"]),
    }


def _zero_residual_output(model: MechanisticODE) -> None:
    """Zero the residual's final layer so it outputs 0 -> the UDE starts as PURE
    physics. In this low-rate regime the default small-random residual otherwise
    dominates dC_b, corrupting both the optics init and solver stiffness. The
    final-layer weights still receive gradient (their input is non-zero), so the
    residual learns from step 1."""
    last = [m for m in model.residual.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.zero_()


def _portable_state(model: GrayBoxUDE) -> Dict[str, Tensor]:
    """state_dict without the transient per-trace covariate buffers (their shape
    depends on the last batch, so they must not travel with the checkpoint)."""
    sd = model.state_dict()
    sd.pop("core.T_ambient", None)
    sd.pop("core.RH", None)
    return sd


# --------------------------------------------------------------------------- #
#  Checkpoint save / load                                                     #
# --------------------------------------------------------------------------- #
def build_from_state(state: Dict, residual_scale: float) -> Tuple[GrayBoxUDE, MeasurementModel]:
    core = MechanisticODE.identifiable()
    model = GrayBoxUDE(core, residual_scale)
    model.load_state_dict(state["model"], strict=False)   # covariate buffers set at inference
    meas = MeasurementModel()
    meas.load_state_dict(state["meas"])
    return model, meas


def save_checkpoint(result: Dict, path: Path, cal_bundle: "D.TraceBundle") -> Tensor:
    """Save the best model + optics and a reference cal prediction. Returns the
    reference prediction so the caller can verify reload reproduction."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = result["cfg"]

    model, meas = build_from_state(result["best_state"], cfg.residual_scale)
    _, _, cal_pred = eval_loss(model, meas, cal_bundle, cfg)

    torch.save({
        "model": result["best_state"]["model"],
        "meas": result["best_state"]["meas"],
        "cfg": cfg.__dict__,
        "cal_pred_ref": cal_pred,
        "best_cal": result["best_cal"],
        "best_epoch": result["best_epoch"],
    }, path)
    return cal_pred


def load_checkpoint(path: Path) -> Tuple[GrayBoxUDE, MeasurementModel, Dict, Tensor]:
    ckpt = torch.load(path, weights_only=False)
    scale = ckpt["cfg"].get("residual_scale", 3e-4)
    model, meas = build_from_state({"model": ckpt["model"], "meas": ckpt["meas"]}, scale)
    return model, meas, ckpt["cfg"], ckpt["cal_pred_ref"]


# --------------------------------------------------------------------------- #
#  Loss curve                                                                 #
# --------------------------------------------------------------------------- #
def save_loss_curve(gray: Dict, baseline: Dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    gh, bh = gray["history"], baseline["history"]

    ax[0].plot(gh["train"], label="gray-box train", color="C0")
    ax[0].plot(gh["cal"], label="gray-box cal", color="C0", ls="--")
    ax[0].plot(bh["cal"], label="physics-only cal (Baseline A)", color="C3", ls=":")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("total loss (log)")
    ax[0].set_title("Training / calibration loss"); ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].plot(gh["cal_obs"], label="gray-box cal (obs MSE)", color="C0")
    ax[1].plot(bh["cal_obs"], label="physics-only cal (obs MSE)", color="C3", ls=":")
    ax[1].axhline(baseline["best_cal"], color="C3", lw=0.8, alpha=0.5)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("observation MSE (log)")
    ax[1].set_title("Gray-box vs pure-physics ceiling"); ax[1].legend(); ax[1].grid(alpha=0.3)

    fig.suptitle("KinetiFlow-CP v2 — synthetic training", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  Main: train baseline + gray-box, run acceptance checks, save artifacts     #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    data = D.load()
    train_b, cal_b, test_b = D.grouped_split(data)
    print(f"data: train={len(train_b)}  cal={len(cal_b)}  test={len(test_b)}")

    print("\n[1/2] training physics-only baseline (Baseline A)...")
    base_cfg = TrainConfig(physics_only=True, tag="physics_only")
    base = train(train_b, cal_b, base_cfg)
    print(f"      baseline best cal loss = {base['best_cal']:.4f} "
          f"(epoch {base['best_epoch']}, {base['epochs_run']} run)")

    print("\n[2/2] training gray-box (residual on)...")
    gray_cfg = TrainConfig(physics_only=False, tag="graybox")
    gray = train(train_b, cal_b, gray_cfg)
    print(f"      gray-box best cal loss = {gray['best_cal']:.4f} "
          f"(epoch {gray['best_epoch']}, {gray['epochs_run']} run)")

    # ---- checkpoint + reload reproduction ---------------------------------
    ckpt_path = CKPT_DIR / "graybox_best.pt"
    ref_pred = save_checkpoint(gray, ckpt_path, cal_b)
    r_model, r_meas, _, saved_ref = load_checkpoint(ckpt_path)
    _, _, reload_pred = eval_loss(r_model, r_meas, cal_b, gray_cfg)
    repro_err = float((reload_pred - saved_ref).abs().max())

    # ---- acceptance metrics -----------------------------------------------
    cal0 = gray["cal0"]
    cal_best = gray["best_cal"]
    decrease = 100.0 * (cal0 - cal_best) / cal0
    beats = cal_best < base["best_cal"]
    improve = 100.0 * (base["best_cal"] - cal_best) / base["best_cal"]

    save_loss_curve(gray, base, FIG_DIR / "training_loss.png")

    print("\n" + "=" * 64)
    print("ACCEPTANCE (D) — synthetic training")
    print("=" * 64)
    print(f"gray-box cal loss: epoch0={cal0:.4f} -> best={cal_best:.4f}  "
          f"(-{decrease:.1f}%)   {'PASS' if decrease > 50 else 'FAIL'} (>50%)")
    print(f"all gradients finite: {gray['grads_all_finite']}  "
          f"{'PASS' if gray['grads_all_finite'] else 'FAIL'}")
    print(f"gray-box ({cal_best:.4f}) beats physics_only ({base['best_cal']:.4f}): "
          f"{beats}  (residual improves cal by {improve:.1f}%)  "
          f"{'PASS' if beats else 'FAIL'}")
    print(f"checkpoint reload max|Δpred| = {repro_err:.2e}  "
          f"{'PASS' if repro_err < 1e-5 else 'FAIL'} (<1e-5)")
    print(f"\nloss curve: {FIG_DIR / 'training_loss.png'}")
    print(f"checkpoint: {ckpt_path}")
