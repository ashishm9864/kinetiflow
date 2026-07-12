"""Literature priors and unit conversions for KinetiFlow-CP.

The production ODE uses a numerically well-scaled but dimensionally coherent
state convention:

``L``
    Washburn front position in mm.
``C_f``
    Free hCG concentration in ng/mL.
``C_b`` and ``B_max``
    Bound hCG / occupied capture-site amount density in nmol/m^2.

Using amount density for both ``C_b`` and ``B_max`` avoids the old error of
comparing IgG mass (150 kDa) with bound-complex mass.  Conversion between the
surface reaction flux and bulk depletion is handled explicitly through the
capture area and effective accessible volume in :mod:`mechanistic_ode`.

The geometry entries are low-confidence *virtual-assay* values.  They make the
synthetic simulator dimensionally closed; hardware measurements must replace
them before physical parameter claims are made.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Dict, Tuple


MW_HCG_G_PER_MOL: float = 37_000.0
MW_IGG_G_PER_MOL: float = 150_000.0
N_AVOGADRO: float = 6.022_140_76e23
MIU_PER_NG_HCG: float = 9.3


def ng_per_mL_to_M(c_ng_per_mL: float) -> float:
    """hCG concentration [ng/mL] -> mol/L [M]."""
    return c_ng_per_mL * 1e-6 / MW_HCG_G_PER_MOL


def M_to_ng_per_mL(c_M: float) -> float:
    """hCG concentration [M] -> ng/mL."""
    return c_M * MW_HCG_G_PER_MOL / 1e-6


def mIU_per_mL_to_ng_per_mL(c_mIU_per_mL: float) -> float:
    """hCG activity [mIU/mL] -> mass concentration [ng/mL]."""
    return c_mIU_per_mL / MIU_PER_NG_HCG


def ng_per_mL_to_mIU_per_mL(c_ng_per_mL: float) -> float:
    return c_ng_per_mL * MIU_PER_NG_HCG


def ug_per_cm2_to_mol_per_m2(sigma_ug_per_cm2: float, mw_g_per_mol: float) -> float:
    """Surface mass density [ug/cm^2] -> amount density [mol/m^2]."""
    return (sigma_ug_per_cm2 * 1e-2) / mw_g_per_mol


def mol_per_m2_to_ug_per_cm2(sigma_mol_per_m2: float, mw_g_per_mol: float) -> float:
    return sigma_mol_per_m2 * mw_g_per_mol / 1e-2


def mol_per_m2_to_nmol_per_m2(value: float) -> float:
    return value * 1e9


@dataclass(frozen=True)
class Prior:
    central: float
    low: float
    high: float
    units: str
    confidence: str
    source: str

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.low, self.central, self.high)


# Frozen biochemical priors.  These are never optimized by the production model.
K_ON = Prior(1e6, 1e5, 5e6, "M^-1 s^-1", "medium", "Kamat 2017; Klonisch 1996")
K_OFF = Prior(1e-4, 4e-6, 1e-3, "s^-1", "medium", "Murthy 1996; Ashish 2004")
K_D = Prior(1e-9, 3e-11, 1e-8, "M", "high", "Berger 1984; Klonisch 1996")
B_MAX = Prior(
    6.7e-8,
    1e-8,
    2e-7,
    "mol/m^2",
    "low",
    "derived from striping concentration; accessible fraction unmeasured",
)

D_HCG = Prior(5e-11, 4e-11, 6e-11, "m^2/s", "medium", "Stokes-Einstein")
U_FLOW = Prior(5e-5, 1.7e-5, 8.3e-5, "m/s", "medium", "Washburn wicking")
K_M = Prior(1.6e-6, 3e-7, 5e-6, "m/s", "low", "convective-diffusion scaling")

# k_wash = L * dL/dt.  At L0=5 mm the central value corresponds to the
# literature central flow speed 0.05 mm/s (3 mm/min), unlike the former 2.0
# mm^2/s value which implied 24 mm/min at the read window.
K_WASH = Prior(0.25, 0.085, 0.415, "mm^2/s", "low", "derived from U_FLOW at L0=5 mm")

HOOK_ONSET = Prior(
    200.0,
    40.0,
    1000.0,
    "IU/mL",
    "low",
    "hCG-LFA onset unmeasured; bracket only",
)

# Virtual-assay geometry used solely to close the synthetic mass balance.
# 5 mm strip width x 1 mm capture-line width = 5 mm^2.
CAPTURE_AREA_M2 = Prior(5e-6, 2.5e-6, 1e-5, "m^2", "low", "virtual geometry")
# Accessible sample/reservoir volume at L0.  This is deliberately explicit and
# must be measured on hardware rather than inferred from one optical channel.
VOLUME_AT_L0_ML = Prior(0.05, 0.02, 0.10, "mL", "low", "virtual geometry")
# Effective wetted pore cross-section; dV/dL = cross-section * 1e-3 mL/mm.
PORE_CROSS_SECTION_MM2 = Prior(0.35, 0.15, 0.60, "mm^2", "low", "5 mm x 0.1 mm x porosity")
L0_MM: float = 5.0


def damkohler(
    k_on_M: float = K_ON.central,
    B_max_mol_m2: float = B_MAX.central,
    k_m_m_s: float = K_M.central,
) -> float:
    """Da = k_on * B_max / k_m in consistent SI units."""
    return (k_on_M * 1e-3) * B_max_mol_m2 / k_m_m_s


def k_on_effective_M(
    k_on_M: float = K_ON.central,
    B_max_mol_m2: float = B_MAX.central,
    k_m_m_s: float = K_M.central,
) -> float:
    """Two-compartment effective association rate in M^-1 s^-1."""
    k_on_si = k_on_M * 1e-3
    effective_si = k_on_si * k_m_m_s / (k_on_si * B_max_mol_m2 + k_m_m_s)
    return effective_si * 1e3


def production_values() -> Dict[str, float]:
    """Exact frozen values expected in every production core/checkpoint."""
    return {
        "k_wash": K_WASH.central,
        "k_on": K_ON.central * (1e-6 / MW_HCG_G_PER_MOL),
        "k_off": K_OFF.central,
        "B_max": mol_per_m2_to_nmol_per_m2(B_MAX.central),
        "capture_area_m2": CAPTURE_AREA_M2.central,
        "volume_at_L0_mL": VOLUME_AT_L0_ML.central,
        "pore_cross_section_mm2": PORE_CROSS_SECTION_MM2.central,
        "L0": L0_MM,
    }


def prior_signature() -> str:
    """Stable hash used to reject checkpoints with altered frozen priors."""
    payload = json.dumps(production_values(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def to_ode_config(**overrides):
    """Build the production ODE config in the coherent state convention."""
    from mechanistic_ode import ODEConfig

    defaults = production_values()
    defaults.pop("L0")
    defaults.update(overrides)
    return ODEConfig(**defaults)


if __name__ == "__main__":
    values = production_values()
    assert abs(ng_per_mL_to_M(1.0) - 2.7027027027e-11) < 1e-20
    assert abs(values["B_max"] - 67.0) < 1e-12
    print("== production priors ==")
    for key, value in values.items():
        print(f"{key:24s} {value:.8g}")
    print(f"Damkohler             {damkohler():.3f}")
    print(f"prior signature       {prior_signature()}")
