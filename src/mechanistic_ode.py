"""
mechanistic_ode.py
==================
Latent-state gray-box Universal Differential Equation (UDE) for lateral-flow
assay (LFA) kinetics — KinetiFlow-CP v2, Phase 1.

The ODE integrates THREE physical latent states (never image intensity):

    z(t) = [ L(t), C_f(t), C_b(t) ]
        L   : fluid-front position along nitrocellulose membrane   [mm]
        C_f : free labelled analyte conc. at the test-line zone     [ng/mL]
        C_b : bound antibody-antigen complex surface density        [ug/cm^2]

Known mechanistic physics (Lucas-Washburn 1921 + Qian & Bau 2003):

    dL/dt   = k_wash / L                                    (capillary advection)
    dC_f/dt = -(dL/dt / L) * C_f
              - k_on * C_f * (B_max - C_b) + k_off * C_b    (Langmuir, free)
    dC_b/dt =  k_on * C_f * (B_max - C_b) - k_off * C_b     (Langmuir, bound)

A neural residual f_theta corrects ONLY the chemical kinetics (C_f, C_b). It
never touches transport (L) and never touches the optical observation:

    dz/dt = KnownPhysics(z) + [ 0, f_theta_1, f_theta_2 ]

Optical measurement is a SEPARATE model applied to the solved C_b trajectory
(this is the v2 change that keeps chemistry and camera model identifiable):

    I_obs(t) = alpha * C_b(t) + beta                (+ optional sigmoidal upgrade)

------------------------------------------------------------------------------
NOTE 1 — Washburn exponent (flag this with your mentor)
    The outline writes dL/dt = sqrt(gamma_r / (8*eta*L)). That is inconsistent
    with Lucas-Washburn, which gives L ∝ sqrt(t)  <=>  dL/dt ∝ 1/L. Taken
    literally the sqrt form gives L ∝ t^(2/3). We implement the correct 1/L
    form with a single lumped constant k_wash = gamma*r*cos(theta)/(4*eta).

NOTE 2 — Rate-constant units / priors
    Literature k_on/k_off are usually molar (M^-1 s^-1 / s^-1). Here C_f is in
    ng/mL and C_b in ug/cm^2, so the defaults below are PLACEHOLDER priors that
    produce a stable, well-conditioned trajectory. Replace them with values
    from the Claude Science literature pass, and be explicit about the
    bulk->surface unit conversion (a judge will ask).

NOTE 3 — Positivity
    k_wash, k_on, k_off, B_max are log-parameterized (stored as logs, exp'd on
    read) so they stay strictly positive under gradient descent. Combine with
    the physics-bounds penalty in loss_functions.py for admissible states.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

try:
    from torchdiffeq import odeint, odeint_adjoint
except ImportError as exc:  # pragma: no cover
    raise ImportError("torchdiffeq is required: pip install torchdiffeq") from exc


# --------------------------------------------------------------------------- #
#  Configuration                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class ODEConfig:
    # --- Known-physics priors (replace with Claude Science literature values) ---
    k_wash: float = 2.0        # lumped Washburn constant           [mm^2 / s]
    k_on: float = 1.0e-4       # association rate         [mL / (ng * s)] (see NOTE 2)
    k_off: float = 1.0e-3      # dissociation rate                  [1 / s]
    B_max: float = 1.0         # capture-site surface density       [ug / cm^2]

    # --- Neural residual (3 Linear layers, 64 hidden, Tanh) ---
    residual_hidden: int = 64
    residual_depth: int = 3            # number of Linear layers
    residual_in_dim: int = 5           # (C_f, C_b, t, T_ambient, RH)
    residual_out_dim: int = 2          # (dC_f, dC_b) — never corrects L
    residual_init_scale: float = 1e-2  # near-zero start => UDE ~ pure physics at init

    # --- Residual input normalization (keeps the MLP well-conditioned) ---
    #  raw = [C_f (ng/mL), C_b (ug/cm^2), t (s), T (C), RH (%)]
    res_mean: Tuple[float, ...] = (25.0, 0.5, 450.0, 30.0, 55.0)
    res_scale: Tuple[float, ...] = (25.0, 0.5, 450.0, 8.0, 20.0)

    # --- Numerics ---
    eps: float = 1.0e-6                # floor on L to avoid 1/L singularity
    trainable_physics: bool = True     # False => freeze kinetics (Risk-1 fallback)


# --------------------------------------------------------------------------- #
#  Core dynamics                                                              #
# --------------------------------------------------------------------------- #
class MechanisticODE(nn.Module):
    """Gray-box vector field dz/dt = f(t, z). Compatible with torchdiffeq
    odeint / odeint_adjoint. Handles both single [3] and batched [B, 3] state.
    """

    def __init__(self, cfg: Optional[ODEConfig] = None):
        super().__init__()
        self.cfg = cfg = cfg or ODEConfig()

        # Positive physical parameters via log-parameterization.
        def _logp(value: float) -> nn.Parameter:
            return nn.Parameter(
                torch.log(torch.tensor(float(value))),
                requires_grad=cfg.trainable_physics,
            )

        self.log_k_wash = _logp(cfg.k_wash)
        self.log_k_on = _logp(cfg.k_on)
        self.log_k_off = _logp(cfg.k_off)
        self.log_B_max = _logp(cfg.B_max)

        # Neural residual: [5] -> 64 -> 64 -> [2], Tanh activations.
        layers: List[nn.Module] = []
        d_in = cfg.residual_in_dim
        for _ in range(cfg.residual_depth - 1):
            layers += [nn.Linear(d_in, cfg.residual_hidden), nn.Tanh()]
            d_in = cfg.residual_hidden
        final = nn.Linear(d_in, cfg.residual_out_dim)
        # Near-zero init: residual starts ~0 (model ~ pure physics) but every
        # layer still receives a finite, non-zero gradient (final layer != 0).
        with torch.no_grad():
            final.weight.mul_(cfg.residual_init_scale)
            final.bias.zero_()
        layers.append(final)
        self.residual = nn.Sequential(*layers)

        # Per-trajectory environmental covariates. Buffers (no grad) so the
        # adjoint parameter search doesn't try to differentiate them.
        self.register_buffer("T_ambient", torch.tensor(float(cfg.res_mean[3])))
        self.register_buffer("RH", torch.tensor(float(cfg.res_mean[4])))
        self.register_buffer("res_mean", torch.tensor(cfg.res_mean))
        self.register_buffer("res_scale", torch.tensor(cfg.res_scale))
        self.register_buffer("eps_t", torch.tensor(float(cfg.eps)))

    # -- positive-parameter accessors ------------------------------------- #
    @property
    def k_wash(self) -> Tensor:
        return torch.exp(self.log_k_wash)

    @property
    def k_on(self) -> Tensor:
        return torch.exp(self.log_k_on)

    @property
    def k_off(self) -> Tensor:
        return torch.exp(self.log_k_off)

    @property
    def B_max(self) -> Tensor:
        return torch.exp(self.log_B_max)

    def set_covariates(self, T_ambient: float, RH: float) -> None:
        """Set the environmental covariates for the strip about to be solved."""
        self.T_ambient.fill_(float(T_ambient))
        self.RH.fill_(float(RH))

    def freeze_physics(self, frozen: bool = True) -> None:
        """Risk-1 fallback: lock k_on/k_off/k_wash/B_max, learn only residual."""
        for p in (self.log_k_wash, self.log_k_on, self.log_k_off, self.log_B_max):
            p.requires_grad_(not frozen)

    def freeze_residual(self, frozen: bool = True) -> None:
        """Baseline A (pure-physics ceiling): lock the neural residual so only the
        frozen known-physics + optics remain. Combined with freeze_physics() this
        yields a fully non-learnable dynamics core."""
        for p in self.residual.parameters():
            p.requires_grad_(not frozen)

    # -- identifiable-training-mode factory ------------------------------- #
    @classmethod
    def identifiable(cls, **overrides) -> "MechanisticODE":
        """Build the model in IDENTIFIABLE TRAINING MODE.

        identifiability.py proved that from the single output I_obs, alpha and
        B_max are perfectly confounded and k_on/k_off/k_wash are not separately
        recoverable; so we FREEZE k_wash/k_on/k_off/B_max at the priors.py values
        and let only {alpha, beta (in MeasurementModel), neural residual} train.

        `overrides` are forwarded to priors.to_ode_config() (e.g. residual_hidden).
        The full-flexibility mode remains available via the normal constructor.
        """
        import priors as _P

        # trainable_physics=False => the four log-parameters are created frozen.
        cfg = _P.to_ode_config(trainable_physics=False, **overrides)
        return cls(cfg)

    # -- vector field ----------------------------------------------------- #
    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        L, C_f, C_b = z.unbind(dim=-1)
        L_safe = L.clamp_min(self.eps_t)          # guard 1/L singularity

        dL = self.k_wash / L_safe                                  # transport
        binding = self.k_on * C_f * (self.B_max - C_b)
        unbinding = self.k_off * C_b
        dC_f_phys = -(dL / L_safe) * C_f - binding + unbinding      # free
        dC_b_phys = binding - unbinding                            # bound

        # Neural residual on normalized (C_f, C_b, t, T, RH).
        ones = torch.ones_like(C_f)
        raw = torch.stack(
            [C_f, C_b, ones * t, ones * self.T_ambient, ones * self.RH], dim=-1
        )
        res_in = (raw - self.res_mean) / self.res_scale
        delta = self.residual(res_in)              # [..., 2] = (dC_f, dC_b)

        dC_f = dC_f_phys + delta[..., 0]
        dC_b = dC_b_phys + delta[..., 1]
        return torch.stack([dL, dC_f, dC_b], dim=-1)


# --------------------------------------------------------------------------- #
#  Optical measurement model (kept separate from the chemistry)               #
# --------------------------------------------------------------------------- #
class MeasurementModel(nn.Module):
    """I_obs = alpha * C_b + beta  (linear, default), or the sigmoidal upgrade
    I_obs = alpha * tanh(k_sig * C_b) + beta for high-concentration saturation.

    Sign convention: alpha may be negative if you record reflected intensity
    (line gets darker as C_b grows); positive if you record line 'darkness'.
    """

    def __init__(
        self,
        alpha: float = 100.0,
        beta: float = 20.0,
        sigmoidal: bool = False,
        k_sig: float = 1.0,
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.sigmoidal = sigmoidal
        if sigmoidal:
            self.k_sig = nn.Parameter(torch.tensor(float(k_sig)))

    def forward(self, C_b: Tensor) -> Tensor:
        if self.sigmoidal:
            return self.alpha * torch.tanh(self.k_sig * C_b) + self.beta
        return self.alpha * C_b + self.beta


# --------------------------------------------------------------------------- #
#  Integration helper                                                         #
# --------------------------------------------------------------------------- #
def integrate(
    model: MechanisticODE,
    measurement: MeasurementModel,
    z0: Tensor,
    t_eval: Tensor,
    *,
    method: str = "dopri5",
    use_adjoint: bool = True,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    options: Optional[dict] = None,
) -> Tuple[Tensor, Tensor]:
    """Solve the UDE over t_eval and map C_b -> observed intensity.

    Returns
    -------
    z_traj : Tensor, shape [len(t_eval), ..., 3]
    I_pred : Tensor, shape [len(t_eval), ...]

    Notes
    -----
    * `dopri5` (adaptive RK) is the robust default used for the stability /
      gradient-flow smoke test. For the stiff regime flagged in the risk
      register, switch to a fixed-step implicit method
      (method="implicit_adams", options={"step_size": ...}) or the
      scipy Radau fallback described in the outline.
    * `use_adjoint=True` uses the O(1)-memory continuous adjoint — this is the
      path used in training over long (15-min) integration windows.
    """
    solver = odeint_adjoint if use_adjoint else odeint
    kwargs = dict(method=method, atol=atol, rtol=rtol)
    if options is not None:
        kwargs["options"] = options
    z_traj = solver(model, z0, t_eval, **kwargs)
    I_pred = measurement(z_traj[..., 2])   # observe the bound-complex channel
    return z_traj, I_pred


# --------------------------------------------------------------------------- #
#  Gradient-flow verification (the Phase-1 acceptance test)                    #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)

    cfg = ODEConfig()
    model = MechanisticODE(cfg)
    meas = MeasurementModel(alpha=100.0, beta=20.0)
    model.set_covariates(T_ambient=30.0, RH=55.0)

    # Initial state: L0 set to the front already near the test line. Starting at
    # L0 ~ 0 makes the Washburn term k_wash/L and the advection term
    # (k_wash/L^2)*C_f explode, which is stiff and underflows the adjoint solver.
    # Start the clock once the front has wetted the read window (a few mm).
    z0 = torch.tensor([5.0, 20.0, 0.0])
    t_eval = torch.linspace(0.0, 900.0, 91)     # 15 minutes @ 10 s resolution

    # ---- 1) forward solve (stability) -------------------------------------
    z_traj, I_pred = integrate(model, meas, z0, t_eval, use_adjoint=True)
    finite = bool(torch.isfinite(z_traj).all() and torch.isfinite(I_pred).all())
    zf = z_traj.detach()
    print("z_traj shape :", tuple(z_traj.shape), "| I_pred shape:", tuple(I_pred.shape))
    print(
        "final state  : L=%.3f mm  C_f=%.3f ng/mL  C_b=%.4f ug/cm^2"
        % (zf[-1, 0], zf[-1, 1], zf[-1, 2])
    )
    print("monotone L   :", bool((z_traj[1:, 0] >= z_traj[:-1, 0]).all()))
    print("C_b <= B_max :", bool((z_traj[..., 2] <= model.B_max + 1e-4).all()))
    print("all finite   :", finite)

    # ---- 2) real training signal: observation + physics-bounds loss --------
    try:
        from loss_functions import kinetiflow_loss

        I_target = I_pred.detach() + 0.5 * torch.randn_like(I_pred)  # synthetic
        loss, parts = kinetiflow_loss(I_pred, I_target, z_traj, model.B_max, lam=1.0)
    except ImportError:  # self-contained fallback if run without the loss file
        I_target = I_pred.detach() + 0.5 * torch.randn_like(I_pred)
        C_f_t, C_b_t = z_traj[..., 1], z_traj[..., 2]
        L_obs = ((I_pred - I_target) ** 2).mean()
        L_phys = (
            torch.relu(-C_f_t) + torch.relu(-C_b_t) + torch.relu(C_b_t - model.B_max)
        ).mean()
        loss = L_obs + 1.0 * L_phys
        parts = {"obs": float(L_obs), "physics": float(L_phys), "total": float(loss)}

    # ---- 3) backprop through the adjoint solver ---------------------------
    loss.backward()
    print("\nloss = %.5f   parts = %s" % (loss.item(), parts))

    print("\n--- gradient-flow check (continuous adjoint) ---")
    named: List[Tuple[str, Tensor]] = list(model.named_parameters())
    named += [("meas." + n, p) for n, p in meas.named_parameters()]

    all_ok = True
    for name, p in named:
        g = p.grad
        if g is None or not torch.isfinite(g).all() or g.norm() == 0:
            all_ok = False
            shown = "None" if g is None else f"{g.norm().item():.3e}"
            print(f"  [FAIL] {name:26s} grad={shown}")
        else:
            print(f"  [ ok ] {name:26s} grad_norm={g.norm().item():.3e}")

    print("\nGRADIENT FLOW:", "PASS" if (all_ok and finite) else "FAIL")
