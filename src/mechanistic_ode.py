"""Dimensionally coherent latent-state LFA dynamics.

The three physical states are ``[L_mm, C_f_ng_mL, C_b_nmol_m2]``.  The
Lucas-Washburn front always obeys ``dL/dt = k_wash/L``.  Reversible capture is
represented by a surface flux ``J`` [nmol m^-2 s^-1], then coupled to free bulk
concentration through an explicit capture-area/effective-volume conversion.

The neural component may modify only the on/off chemistry fluxes.  It cannot
modify ``L`` or the optical mapping, and the same flux is applied
stoichiometrically to ``C_f`` and ``C_b``.  The optical model remains a separate
module applied after integration.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torchdiffeq import odeint, odeint_adjoint


@dataclass
class ODEConfig:
    k_wash: float = 0.25
    k_on: float = 2.7027027027e-5       # (ng/mL)^-1 s^-1
    k_off: float = 1e-4                 # s^-1
    B_max: float = 67.0                 # nmol/m^2 occupied sites
    capture_area_m2: float = 5e-6
    volume_at_L0_mL: float = 0.05
    pore_cross_section_mm2: float = 0.35
    residual_hidden: int = 64
    residual_depth: int = 3
    residual_in_dim: int = 5
    residual_out_dim: int = 2           # log on/off multipliers
    residual_log_scale: float = 1.0
    res_mean: Tuple[float, ...] = (6.7, 20.0, 450.0, 30.0, 55.0)
    res_scale: Tuple[float, ...] = (6.7, 20.0, 450.0, 8.0, 20.0)
    eps: float = 1e-8
    trainable_physics: bool = False
    dynamics: str = "graybox"           # physics_only | residual_only | graybox


class MechanisticODE(nn.Module):
    """Qian-Bau-inspired lumped transport/capture model for torchdiffeq."""

    def __init__(self, cfg: Optional[ODEConfig] = None):
        super().__init__()
        self.cfg = cfg = cfg or ODEConfig()
        if cfg.dynamics not in {"physics_only", "residual_only", "graybox"}:
            raise ValueError(f"unknown dynamics mode: {cfg.dynamics}")

        def log_parameter(value: float) -> nn.Parameter:
            return nn.Parameter(
                torch.log(torch.tensor(float(value))),
                requires_grad=cfg.trainable_physics,
            )

        self.log_k_wash = log_parameter(cfg.k_wash)
        self.log_k_on = log_parameter(cfg.k_on)
        self.log_k_off = log_parameter(cfg.k_off)
        self.log_B_max = log_parameter(cfg.B_max)

        layers: List[nn.Module] = []
        d_in = cfg.residual_in_dim
        for _ in range(cfg.residual_depth - 1):
            layers.extend([nn.Linear(d_in, cfg.residual_hidden), nn.Tanh()])
            d_in = cfg.residual_hidden
        final = nn.Linear(d_in, cfg.residual_out_dim)
        with torch.no_grad():
            final.weight.zero_()
            final.bias.zero_()
        layers.append(final)
        self.residual = nn.Sequential(*layers)
        if cfg.dynamics == "physics_only":
            self.freeze_residual(True)

        self.register_buffer("T_ambient", torch.tensor(float(cfg.res_mean[3])), persistent=False)
        self.register_buffer("RH", torch.tensor(float(cfg.res_mean[4])), persistent=False)
        self.register_buffer("res_mean", torch.tensor(cfg.res_mean, dtype=torch.float32))
        self.register_buffer("res_scale", torch.tensor(cfg.res_scale, dtype=torch.float32))
        self.register_buffer("capture_area_m2", torch.tensor(float(cfg.capture_area_m2)))
        self.register_buffer("volume_at_L0_mL", torch.tensor(float(cfg.volume_at_L0_mL)))
        self.register_buffer("pore_cross_section_mm2", torch.tensor(float(cfg.pore_cross_section_mm2)))
        self.register_buffer("L0_mm", torch.tensor(5.0))
        self.register_buffer("eps_t", torch.tensor(float(cfg.eps)))

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

    @classmethod
    def identifiable(cls, **overrides) -> "MechanisticODE":
        """Construct production mode with every physical prior frozen."""
        import priors as P

        overrides.setdefault("trainable_physics", False)
        model = cls(P.to_ode_config(**overrides))
        model.validate_frozen_priors()
        return model

    def freeze_physics(self, frozen: bool = True) -> None:
        for parameter in (self.log_k_wash, self.log_k_on, self.log_k_off, self.log_B_max):
            parameter.requires_grad_(not frozen)

    def freeze_residual(self, frozen: bool = True) -> None:
        for parameter in self.residual.parameters():
            parameter.requires_grad_(not frozen)

    def validate_frozen_priors(self, atol: float = 1e-8) -> None:
        """Fail closed if a frozen physical value differs from :mod:`priors`."""
        import priors as P

        expected = P.production_values()
        actual = {
            "k_wash": float(self.k_wash.detach()),
            "k_on": float(self.k_on.detach()),
            "k_off": float(self.k_off.detach()),
            "B_max": float(self.B_max.detach()),
            "capture_area_m2": float(self.capture_area_m2),
            "volume_at_L0_mL": float(self.volume_at_L0_mL),
            "pore_cross_section_mm2": float(self.pore_cross_section_mm2),
        }
        for name, wanted in expected.items():
            if name == "L0":
                continue
            got = actual[name]
            if not math.isclose(got, wanted, rel_tol=1e-6, abs_tol=atol):
                raise RuntimeError(f"frozen prior {name} changed: {got} != {wanted}")
        for name in ("log_k_wash", "log_k_on", "log_k_off", "log_B_max"):
            if getattr(self, name).requires_grad:
                raise RuntimeError(f"production prior {name} is trainable")

    def set_covariates(self, T_ambient: float, RH: float) -> None:
        self.T_ambient.fill_(float(T_ambient))
        self.RH.fill_(float(RH))

    def set_covariate_tensors(self, T_ambient: Tensor, RH: Tensor) -> None:
        # Values are observed exogenous covariates, not inferred parameters.
        self.T_ambient = T_ambient.detach()
        self.RH = RH.detach()

    def volume_mL(self, L_mm: Tensor) -> Tensor:
        """Effective accessible volume V(L) [mL]."""
        added_length = (L_mm - self.L0_mm).clamp_min(0.0)
        return self.volume_at_L0_mL + self.pore_cross_section_mm2 * 1e-3 * added_length

    def dvolume_dt_mL_s(self, L_mm: Tensor, dL_mm_s: Tensor) -> Tensor:
        del L_mm
        return self.pore_cross_section_mm2 * 1e-3 * dL_mm_s

    def _features(self, t: Tensor, C_f: Tensor, C_b: Tensor) -> Tensor:
        ones = torch.ones_like(C_f)
        raw = torch.stack(
            [C_f, C_b, ones * t, ones * self.T_ambient, ones * self.RH], dim=-1
        )
        return (raw - self.res_mean) / self.res_scale

    def reaction_flux(self, t: Tensor, C_f: Tensor, C_b: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Return net surface capture flux J [nmol/m^2/s] and diagnostics."""
        cf = C_f.clamp_min(0.0)
        occupancy = (C_b / self.B_max).clamp(0.0, 1.0)
        cb = occupancy * self.B_max
        raw = self.residual(self._features(t, cf, cb))

        if self.cfg.dynamics == "residual_only":
            # Neural kinetics with only dimensional reference scales and boundary
            # gates; no frozen Langmuir coefficients enter this ablation.
            rate_scale = self.B_max / 900.0
            cf_gate = cf / (cf + self.res_scale[0].clamp_min(self.eps_t))
            j_on = rate_scale * torch.nn.functional.softplus(raw[..., 0]) * cf_gate * (1.0 - occupancy)
            j_off = rate_scale * torch.nn.functional.softplus(raw[..., 1]) * occupancy
        else:
            log_mult = torch.clamp(
                raw * float(self.cfg.residual_log_scale), min=-4.0, max=4.0
            )
            if self.cfg.dynamics == "physics_only":
                log_mult = torch.zeros_like(log_mult)
            j_on = self.k_on * cf * (self.B_max - cb) * torch.exp(log_mult[..., 0])
            j_off = self.k_off * cb * torch.exp(log_mult[..., 1])
        return j_on - j_off, {"j_on": j_on, "j_off": j_off, "raw": raw}

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        L, C_f, C_b = z.unbind(dim=-1)
        if bool(torch.any(L <= 0.0)):
            raise ValueError("L must stay > 0; t=0 is front arrival at the read window")
        dL = self.k_wash / L
        volume = self.volume_mL(L)
        dvolume = self.dvolume_dt_mL_s(L, dL)
        flux, _ = self.reaction_flux(t, C_f, C_b)

        # 1 nmol/m^2 of bound hCG over area A has mass A*MW_HCG ng.
        import priors as P

        surface_to_bulk = self.capture_area_m2 * P.MW_HCG_G_PER_MOL / volume
        dC_f = -(dvolume / volume) * C_f - surface_to_bulk * flux
        dC_b = flux
        return torch.stack([dL, dC_f, dC_b], dim=-1)

    def total_analyte_ng(self, z: Tensor) -> Tensor:
        """Closed-system mass invariant implied by the lumped equations."""
        import priors as P

        L, C_f, C_b = z.unbind(dim=-1)
        return self.volume_mL(L) * C_f + self.capture_area_m2 * P.MW_HCG_G_PER_MOL * C_b


class MeasurementModel(nn.Module):
    """Separate optics: ``I = alpha*C_b + beta``."""

    def __init__(self, alpha: float = 20.0, beta: float = 25.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(float(beta)))

    def forward(self, C_b: Tensor) -> Tensor:
        return self.alpha * C_b + self.beta


def integrate(
    model: MechanisticODE,
    measurement: MeasurementModel,
    z0: Tensor,
    t_eval: Tensor,
    *,
    method: str = "dopri5",
    use_adjoint: bool = True,
    atol: float = 1e-7,
    rtol: float = 1e-5,
    options: Optional[dict] = None,
) -> Tuple[Tensor, Tensor]:
    """Integrate the physical state and apply optics afterwards."""
    if bool(torch.any(z0[..., 0] <= 0.0)):
        raise ValueError("L0 must be strictly positive")
    solver = odeint_adjoint if use_adjoint else odeint
    kwargs = {"method": method, "atol": atol, "rtol": rtol}
    if options is not None:
        kwargs["options"] = options
    z_traj = solver(model, z0, t_eval, **kwargs)
    return z_traj, measurement(z_traj[..., 2])


if __name__ == "__main__":
    import priors as P

    torch.manual_seed(0)
    core = MechanisticODE.identifiable(dynamics="graybox")
    optics = MeasurementModel()
    z0 = torch.tensor([P.L0_MM, P.mIU_per_mL_to_ng_per_mL(125.0), 0.0])
    t = torch.linspace(0.0, 900.0, 91)
    z, intensity = integrate(core, optics, z0, t, use_adjoint=True)
    expected_l2 = z0[0] ** 2 + 2.0 * core.k_wash.detach() * t
    assert torch.allclose(z[:, 0] ** 2, expected_l2, rtol=1e-5, atol=1e-5)
    drift = float((core.total_analyte_ng(z) - core.total_analyte_ng(z[0])).abs().max())
    assert drift < 2e-4, drift
    assert float(z[:, 1].min()) >= -1e-5
    assert float(z[:, 2].min()) >= -1e-5
    assert float(z[:, 2].max()) <= float(core.B_max) + 1e-4
    loss = intensity[-1]
    loss.backward()
    grads = [p.grad for p in core.residual.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    core.validate_frozen_priors()
    print(f"I900={float(intensity[-1]):.4f} DN")
    print(f"mass drift={drift:.3e} ng")
    print("G1-G6 CORE RUNTIME: PASS")
