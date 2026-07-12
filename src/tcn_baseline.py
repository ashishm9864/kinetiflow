"""
tcn_baseline.py
===============
Baseline B — the BLACK-BOX discrete-time comparator for KinetiFlow-CP v2.

WHAT THIS IS
    A **1D Temporal Convolutional Network (1D TCN)** that forecasts the 15-minute
    equilibrium optical intensity I(900 s) [DN] of an hCG lateral-flow assay
    DIRECTLY from the first 60 s of observed intensity, with no physics. It is the
    black-box performance CEILING the physics-structured gray-box UDE must beat: if
    the gray-box (with the same information, splits, capacity, and conformal layer)
    matches or beats this TCN, the mechanistic structure has earned its keep.

    This is NOT a reproduction of TIMESAVER. Different analyte (hCG vs. the TIMESAVER
    target), different hardware, and a different architecture; it is an independent
    off-the-shelf temporal-CNN regressor used here purely as a fair discrete-time
    baseline. No weights, hyperparameters, or code are taken from that work.

WHY IT MUST BE FAIR (the comparison is meaningless otherwise)
    1. SAME INFORMATION. The TCN's per-trace inputs are exactly:
           * I_obs(t) over the 0-60 s window  (channel 0, a length-7 sequence),
           * T_ambient                        (channel 1, broadcast over the window),
           * RH                               (channel 2, broadcast over the window).
       It NEVER receives the true initial concentration C_f0, the concentration
       level, the day/lot labels, or the target true_I_900 as an input. `raw_features`
       is the single choke point that reads the bundle, and it touches only those
       three fields; `leakage_invariance_max_delta` PROVES the forecast is invariant
       to C_f0/level/day/lot/true_I_900 by scrambling them and asserting the
       predictions are bit-identical.

       HONEST ASYMMETRY (stated, not hidden): the trained gray-box's inference path
       (train.make_batch) initializes its ODE from the TRUE C_f0 and integrates
       forward, so it does not actually consume the I_obs window. Thus the two models'
       information sets are not literally identical -- the gray-box uses C_f0, the TCN
       uses the observed early window. Per project decision the TCN stays on the
       observed window: that is the DEPLOYABLE information set (C_f0 is unknown at test
       time). Reconciling the gray-box's C_f0 seeding is out of scope for this module
       (single-module job; do not modify other modules).

    2. SAME SPLITS. The grouped, leakage-safe (day AND lot) splits are imported from
       dataset.py -- never reimplemented -- and dataset.assert_no_leakage is asserted.

    3. MATCHED CAPACITY. Trainable parameters are tuned to the gray-box's trainable
       count (neural residual 4674 + alpha + beta = 4676). Channel width is the primary
       capacity knob (=20); a small readout-head width (=11) fine-tunes the count to
       4683 (delta +7, 0.15%). Both counts are printed side by side and asserted close.

    4. SAME NORMALIZATION DISCIPLINE. Input and target normalization statistics are
       computed from the TRAIN split ONLY and stored as buffers in the checkpoint, so
       inference uses identical normalization and nothing leaks from cal/test.

CRITICAL INTERFACE (so the SAME conformal layer wraps both models)
    `TCNForecaster.predict(bundle) -> (pred_900, true_900)` returns numpy float64
    arrays in the EXACT format of `conformal.predict_equilibrium(model, meas, bundle,
    cfg)`: the point forecast of I(900 s) in DN and the noise-free ground truth. Every
    conformal.py primitive (`nonconformity_scores`, `scp_quantile`, `scp_interval`,
    `coverage`, `abstain`, `wrcp_intervals`, ...) therefore wraps the TCN with no
    change to conformal.py. The scientifically decisive comparison is
    `gray-box + SCP` vs `TCN + SCP` under this identical conformal layer, which
    isolates the contribution of the physics.

Run the self-test:  python src/tcn_baseline.py
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

import dataset as D
import train  # imported per spec; we reuse its checkpoint dir + gray-box construction
from mechanistic_ode import MechanisticODE, MeasurementModel

# weight norm: prefer the non-deprecated parametrizations API (torch >= 1.12),
# fall back to the legacy functional wrapper on older torch.
try:
    from torch.nn.utils.parametrizations import weight_norm as _weight_norm
except ImportError:  # pragma: no cover
    from torch.nn.utils import weight_norm as _weight_norm


ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = train.CKPT_DIR                       # reuse train.py's checkpoint location
CKPT_PATH = CKPT_DIR / "tcn_baseline.pt"

EARLY_WINDOW_S = 60.0                           # the 0..60 s observation window
TARGET_DELTA = 0.10                             # for the conformal-wrap demonstration

# The ONLY bundle fields the model is allowed to read as inputs, and the fields it
# must never see (they would leak the answer). Enforced by `raw_features` (which reads
# only the allowed set) and PROVEN by `leakage_invariance_max_delta`.
ALLOWED_FEATURE_FIELDS = ("I_obs[0-60 s]", "T_ambient", "RH")
EXCLUDED_FIELDS = ("C_f0", "concentration_level", "day", "lot", "true_I_900 (as input)")


# --------------------------------------------------------------------------- #
#  Configuration                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class TCNConfig:
    # --- architecture (channels + head_hidden chosen to match gray-box capacity) ---
    channels: int = 20            # primary capacity knob (matched to gray-box 4676)
    levels: int = 3               # 3 residual blocks -> dilations 1, 2, 4
    kernel_size: int = 2          # receptive field 1 + (k-1)*sum(dil)*2 = 15 >= 7
    head_hidden: int = 11         # readout width; fine-tunes the trainable count
    dropout: float = 0.1
    # --- optimization ---
    lr: float = 1e-2
    epochs: int = 3000
    patience: int = 300           # early stopping on the CAL split (mirrors train.py)
    grad_clip: float = 1.0
    seed: int = 0
    window_s: float = EARLY_WINDOW_S


# --------------------------------------------------------------------------- #
#  TCN building blocks (Bai, Kolter & Koltun 2018 style)                       #
# --------------------------------------------------------------------------- #
class Chomp1d(nn.Module):
    """Trim the `chomp` right-most timesteps introduced by left-padding, so the
    dilated convolution is strictly CAUSAL (output t depends only on inputs <= t)."""

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = int(chomp)

    def forward(self, x: Tensor) -> Tensor:
        return x[..., : -self.chomp].contiguous() if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    """One residual block: two weight-normed dilated causal convolutions, each with
    ReLU + dropout, plus a residual connection (1x1 conv when channel counts differ)."""

    def __init__(self, n_in: int, n_out: int, kernel_size: int, dilation: int,
                 dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            _weight_norm(nn.Conv1d(n_in, n_out, kernel_size, padding=pad, dilation=dilation)),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            _weight_norm(nn.Conv1d(n_out, n_out, kernel_size, padding=pad, dilation=dilation)),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNForecaster(nn.Module):
    """1D TCN mapping the [B, 3, W] early-window input to a scalar I(900 s) forecast.

    The stacked residual blocks preserve the sequence length W; the LAST (causal)
    timestep aggregates the whole 0-60 s window and is fed to a small MLP readout
    head. Train-only normalization statistics live in buffers so they travel with the
    checkpoint and inference reuses them exactly.
    """

    def __init__(self, cfg: Optional[TCNConfig] = None, n_inputs: int = 3):
        super().__init__()
        self.cfg = cfg = cfg or TCNConfig()
        self.n_inputs = n_inputs

        blocks: List[nn.Module] = []
        for i in range(cfg.levels):
            dilation = 2 ** i                                  # 1, 2, 4
            in_ch = n_inputs if i == 0 else cfg.channels
            blocks.append(TemporalBlock(in_ch, cfg.channels, cfg.kernel_size,
                                        dilation, cfg.dropout))
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(cfg.channels, cfg.head_hidden), nn.ReLU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, 1),
        )

        # Train-only normalization (filled by fit_normalization); part of state_dict.
        self.register_buffer("x_mean", torch.zeros(n_inputs))
        self.register_buffer("x_std", torch.ones(n_inputs))
        self.register_buffer("y_mean", torch.zeros(()))
        self.register_buffer("y_std", torch.ones(()))
        self.register_buffer("window_s", torch.tensor(float(cfg.window_s)))

    def forward(self, x: Tensor) -> Tensor:
        """x : [B, n_inputs, W] standardized -> [B] standardized I(900 s) forecast."""
        h = self.tcn(x)                    # [B, channels, W]
        last = h[..., -1]                  # [B, channels]  causal summary at t = W-end
        return self.head(last).squeeze(-1)  # [B]

    # ---- inference interface (mirrors conformal.predict_equilibrium) --------- #
    @torch.no_grad()
    def predict(self, bundle: "D.TraceBundle") -> Tuple[np.ndarray, np.ndarray]:
        """Forecast I(900 s) for a TraceBundle.

        Returns (pred_900, true_900) as numpy float64 arrays -- the SAME format as
        conformal.predict_equilibrium, so every conformal.py primitive wraps the TCN
        with no modification. Predictions are de-standardized back to DN.
        """
        self.eval()
        x = self._standardize_inputs(bundle)
        yhat = self(x) * self.y_std + self.y_mean               # DN
        return (yhat.detach().cpu().numpy().astype(np.float64),
                _asnp(bundle.true_I_900))

    def _standardize_inputs(self, bundle: "D.TraceBundle") -> Tensor:
        raw = raw_features(bundle, float(self.window_s))         # [B, 3, W]
        return (raw - self.x_mean[None, :, None]) / self.x_std[None, :, None]


# --------------------------------------------------------------------------- #
#  Feature construction — the SINGLE choke point that reads a bundle           #
# --------------------------------------------------------------------------- #
def early_window_mask(t: Tensor, window_s: float = EARLY_WINDOW_S,
                      eps: float = 1e-6) -> Tensor:
    """Boolean mask selecting timepoints in [0, window_s] (the 0-60 s window)."""
    return t <= (window_s + eps)


def raw_features(bundle: "D.TraceBundle", window_s: float = EARLY_WINDOW_S) -> Tensor:
    """Build RAW model inputs [B, 3, W] from ONLY the allowed fields.

    channel 0 = I_obs over 0..window_s
    channel 1 = T_ambient (broadcast across the window)
    channel 2 = RH        (broadcast across the window)

    This function reads NOTHING else from the bundle -- so C_f0, concentration_level,
    day, lot, and true_I_900 cannot enter the forecast. (Enforced here, proven by
    leakage_invariance_max_delta.)
    """
    mask = early_window_mask(bundle.t, window_s)
    I_win = bundle.I_obs[:, mask].float()                        # [B, W]
    W = I_win.shape[1]
    T = bundle.T_ambient.reshape(-1, 1).expand(-1, W).float()     # [B, W]
    RH = bundle.RH.reshape(-1, 1).expand(-1, W).float()           # [B, W]
    return torch.stack([I_win, T, RH], dim=1)                     # [B, 3, W]


@torch.no_grad()
def fit_normalization(model: TCNForecaster, train_bundle: "D.TraceBundle") -> None:
    """Compute per-channel input stats and target stats from the TRAIN split ONLY,
    and write them into the model's buffers (so they travel with the checkpoint)."""
    raw = raw_features(train_bundle, float(model.window_s))       # [B, 3, W]
    model.x_mean.copy_(raw.mean(dim=(0, 2)))
    model.x_std.copy_(raw.std(dim=(0, 2)).clamp_min(1e-6))
    y = train_bundle.true_I_900.float()
    model.y_mean.copy_(y.mean())
    model.y_std.copy_(y.std().clamp_min(1e-6))


# --------------------------------------------------------------------------- #
#  Parameter accounting (matched-capacity gate)                                #
# --------------------------------------------------------------------------- #
def count_trainable(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def graybox_reference_trainable() -> int:
    """The gray-box's trainable-parameter count, computed LIVE from the same
    construction train.py uses: the neural residual (frozen kinetics excluded) plus
    the optics {alpha, beta}. Equals 4674 + 2 = 4676."""
    core = MechanisticODE.identifiable()             # kinetics frozen, residual trainable
    meas = MeasurementModel(alpha=100.0, beta=20.0)
    return (int(sum(p.numel() for p in core.residual.parameters() if p.requires_grad))
            + int(sum(p.numel() for p in meas.parameters() if p.requires_grad)))


# --------------------------------------------------------------------------- #
#  Metrics                                                                     #
# --------------------------------------------------------------------------- #
def _asnp(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


def rmse(pred, true) -> float:
    p, y = _asnp(pred), _asnp(true)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def mae(pred, true) -> float:
    p, y = _asnp(pred), _asnp(true)
    return float(np.mean(np.abs(p - y)))


# --------------------------------------------------------------------------- #
#  Information-set verification (proof by running)                             #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def leakage_invariance_max_delta(model: TCNForecaster, bundle: "D.TraceBundle",
                                 seed: int = 1234) -> float:
    """Scramble every answer-leaking field (C_f0, concentration_level, day, lot, and
    true_I_900) and return the max change in the TCN's point forecast. A value of 0
    PROVES the forecast is invariant to that metadata -- i.e. none of it leaks in."""
    g = torch.Generator().manual_seed(seed)
    n = len(bundle)
    scrambled = dataclasses.replace(
        bundle,
        C_f0=torch.rand(n, generator=g) * 1000.0,
        concentration_level=torch.randint(0, 9, (n,), generator=g).float(),
        day=torch.randint(0, 99, (n,), generator=g),
        lot=torch.randint(0, 99, (n,), generator=g),
        true_I_900=torch.rand(n, generator=g) * 1000.0,
    )
    pred_ref, _ = model.predict(bundle)
    pred_scr, _ = model.predict(scrambled)            # only pred is compared
    return float(np.max(np.abs(pred_ref - pred_scr)))


# --------------------------------------------------------------------------- #
#  Training (early stopping on the CAL split, mirroring train.py)              #
# --------------------------------------------------------------------------- #
def build_model(cfg: TCNConfig) -> TCNForecaster:
    """Seed then construct, so weight initialization is reproducible."""
    torch.manual_seed(cfg.seed)
    return TCNForecaster(cfg)


def train_tcn(model: TCNForecaster, train_bundle: "D.TraceBundle",
              cal_bundle: "D.TraceBundle", cfg: TCNConfig) -> Dict:
    torch.manual_seed(cfg.seed + 1)                  # reproducible dropout stream
    fit_normalization(model, train_bundle)           # TRAIN-only stats -> buffers

    x_tr = model._standardize_inputs(train_bundle)
    y_tr = (train_bundle.true_I_900.float() - model.y_mean) / model.y_std

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    history = {"train_rmse": [], "cal_rmse": []}
    best_cal = float("inf")
    best_state: Optional[Dict] = None
    best_epoch = -1
    since_improve = 0

    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        pred_std = model(x_tr)
        loss = loss_fn(pred_std, y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        # Monitor DN-scale RMSE on train + cal; select the best model on cal.
        tr_pred, tr_true = model.predict(train_bundle)
        cal_pred, cal_true = model.predict(cal_bundle)
        tr_rmse, cal_rmse = rmse(tr_pred, tr_true), rmse(cal_pred, cal_true)
        history["train_rmse"].append(tr_rmse)
        history["cal_rmse"].append(cal_rmse)

        if cal_rmse < best_cal - 1e-9:
            best_cal, best_epoch = cal_rmse, epoch
            best_state = copy.deepcopy(model.state_dict())
            since_improve = 0
        else:
            since_improve += 1
            if since_improve >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)            # restore best-on-cal weights
    return {"history": history, "best_cal_rmse": best_cal, "best_epoch": best_epoch,
            "epochs_run": len(history["train_rmse"])}


# --------------------------------------------------------------------------- #
#  Checkpoint I/O                                                              #
# --------------------------------------------------------------------------- #
def save_checkpoint(model: TCNForecaster, cfg: TCNConfig, path: Path,
                    metrics: Dict, cal_bundle: "D.TraceBundle") -> np.ndarray:
    """Save weights + normalization buffers + config + metrics, plus a reference cal
    prediction so a reload can be verified. Returns the reference prediction."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    cal_pred, _ = model.predict(cal_bundle)
    torch.save({
        "state_dict": model.state_dict(),            # includes normalization buffers
        "cfg": dataclasses.asdict(cfg),
        "n_trainable": count_trainable(model),
        "metrics": metrics,
        "cal_pred_ref": cal_pred,
    }, path)
    return cal_pred


def load_checkpoint(path: Path) -> Tuple[TCNForecaster, TCNConfig, Dict, np.ndarray]:
    ckpt = torch.load(path, weights_only=False)
    cfg = TCNConfig(**ckpt["cfg"])
    model = TCNForecaster(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg, ckpt.get("metrics", {}), np.asarray(ckpt["cal_pred_ref"])


# --------------------------------------------------------------------------- #
#  Self-test / demonstration                                                   #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("=" * 72)
    print("tcn_baseline.py  —  Baseline B: 1D-TCN black-box comparator")
    print("=" * 72)

    cfg = TCNConfig()

    # ---- (1) data + leakage-safe grouped splits (imported, not reimplemented) ---
    data = D.load()
    train_b, cal_b, test_b = D.grouped_split(data)
    D.assert_no_leakage(train_b, cal_b, test_b)      # raises on any day/lot overlap
    print(f"\n[1] splits (from dataset.py): train={len(train_b)}  cal={len(cal_b)}  "
          f"test={len(test_b)}   NO DAY/LOT LEAKAGE: PASS")

    # ---- (2) matched capacity: print both counts side by side -------------------
    model = build_model(cfg)
    n_tcn = count_trainable(model)
    n_gb = graybox_reference_trainable()
    print("\n[2] MATCHED CAPACITY (trainable parameters)")
    print(f"    gray-box (residual 4674 + alpha + beta) : {n_gb}")
    print(f"    TCN (channels={cfg.channels}, head={cfg.head_hidden})           "
          f": {n_tcn}   (delta {n_tcn - n_gb:+d})")
    assert abs(n_tcn - n_gb) <= 25, \
        f"capacity mismatch: TCN {n_tcn} vs gray-box {n_gb} (retune channels/head)"

    # ---- (3) SAME INFORMATION: state it, then PROVE no answer leakage -----------
    print("\n[3] INFORMATION SET (verified)")
    print(f"    TCN inputs (allowed)   : {', '.join(ALLOWED_FEATURE_FIELDS)}")
    print(f"    never seen (excluded)  : {', '.join(EXCLUDED_FIELDS)}")
    x_probe = raw_features(test_b, cfg.window_s)
    print(f"    input tensor shape     : {tuple(x_probe.shape)}  "
          f"(B, [I_obs,T,RH], 0-60 s window)")
    fit_normalization(model, train_b)                # need stats before predicting
    leak = leakage_invariance_max_delta(model, test_b)
    print(f"    scramble C_f0/level/day/lot/true_I_900 -> max|d pred| = {leak:.3e}  "
          f"{'PASS (no leakage)' if leak == 0.0 else 'FAIL (leak!)'}")
    assert leak == 0.0, "forecast changed when answer-leaking metadata was scrambled"

    # ---- (4) train + report test RMSE / MAE -------------------------------------
    print("\n[4] training 1D TCN (early stop on cal, target = clean true_I_900)...")
    model = build_model(cfg)                          # fresh, reproducible init
    res = train_tcn(model, train_b, cal_b, cfg)
    test_pred, test_true = model.predict(test_b)
    cal_pred, cal_true = model.predict(cal_b)
    naive = float(np.mean(_asnp(train_b.true_I_900)))  # predict-the-train-mean baseline
    print(f"    best epoch {res['best_epoch']} / {res['epochs_run']} run")
    print(f"    cal  RMSE = {rmse(cal_pred, cal_true):7.3f} DN")
    print(f"    TEST RMSE = {rmse(test_pred, test_true):7.3f} DN     "
          f"MAE = {mae(test_pred, test_true):7.3f} DN")
    print(f"    (reference: predict-train-mean test RMSE = "
          f"{rmse(np.full_like(test_true, naive), test_true):7.3f} DN)")

    # ---- (5) checkpoint + exact reload reproduction -----------------------------
    metrics = {"test_rmse": rmse(test_pred, test_true),
               "test_mae": mae(test_pred, test_true),
               "cal_rmse": rmse(cal_pred, cal_true)}
    ref = save_checkpoint(model, cfg, CKPT_PATH, metrics, cal_b)
    r_model, _, _, saved_ref = load_checkpoint(CKPT_PATH)
    reload_pred, _ = r_model.predict(cal_b)
    repro = float(np.max(np.abs(reload_pred - saved_ref)))
    print("\n[5] CHECKPOINT")
    print(f"    saved: {CKPT_PATH}")
    print(f"    reload max|d pred| = {repro:.2e}  "
          f"{'PASS' if repro < 1e-5 else 'FAIL'} (<1e-5)")
    assert repro < 1e-5, "checkpoint reload did not reproduce predictions"

    # ---- (6) the SAME conformal layer wraps the TCN IDENTICALLY -----------------
    import conformal
    scores_cal = conformal.nonconformity_scores(*model.predict(cal_b))
    q = conformal.scp_quantile(scores_cal, TARGET_DELTA)
    lo, hi = conformal.scp_interval(test_pred, q)
    cov = conformal.coverage(test_true, lo, hi)
    print("\n[6] TCN + SCP  (conformal.py wraps predict() with zero changes)")
    print(f"    SCP half-width q      : {q:.3f} DN")
    print(f"    clean-test coverage   : {cov:.3f}   (target {1 - TARGET_DELTA:.2f})")
    print(f"    mean interval width   : {conformal.mean_width(lo, hi):.3f} DN")

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"    capacity matched : TCN {n_tcn} vs gray-box {n_gb} (delta {n_tcn - n_gb:+d})")
    print(f"    no day/lot leakage : PASS      answer-metadata leakage : none ({leak:.0e})")
    print(f"    test RMSE / MAE  : {metrics['test_rmse']:.3f} / {metrics['test_mae']:.3f} DN")
    print(f"    TCN + SCP coverage : {cov:.3f}  -> ready for gray-box+SCP vs TCN+SCP")
    print("    NOTE: gray-box seeds z0 from true C_f0 at inference; the TCN is held to "
          "the\n          observed 0-60 s window (deployable info). Asymmetry documented, "
          "not hidden.")
