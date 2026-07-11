"""
priors.py
=========
Literature-grounded priors, bounds, and unit conversions for KinetiFlow-CP v2.

This file operationalizes the Claude Science literature pass
(hCG_LFA_parameters_report.md / parameters.csv). Every value carries its source
tag and a confidence level. It is the single source of truth for parameter
priors, so mechanistic_ode.py and identifiability.py both draw from here rather
than hard-coding magic numbers.

VERIFICATION STATUS (checked independently, this session)
  - All unit conversions in the report reproduce exactly (concentration,
    k_on factor, k_off per-min->per-sec, mIU/ng, Damkohler corners).
  - WHO 5th IS hCG = NIBSC 07/364 confirmed against WHO/FDA/NIBSC sources;
    note it is URINARY-derived, so calibrating a RECOMBINANT standard against it
    carries a glycosylation-commutability caveat.
  - mIU/ng ~ 9.3 independently corroborated (kit insert: 1 ng/mL ~ 9.16 mIU).
  - CAUTION: the hCG-specific hook-onset numbers are attributed to Sathishkumar
    & Toley 2020, which is a C-REACTIVE-PROTEIN paper. Treat hook onset as an
    UNMEASURED quantity for hCG-LFA -> measure it on your own dilution series.

UNIT CONVENTION NOTE
  The report gives B_max in mol/m^2; the built model (mechanistic_ode.py) uses
  ug/cm^2 for both B_max and C_b. Because B_max is capture-*antibody* (150 kDa)
  and C_b is bound *complex*, the ug/cm^2 mass representation mixes molecular
  masses. The physically clean convention is MOLAR SURFACE DENSITY (mol/m^2 or
  mol/cm^2) for B_max and C_b. Conversions to both systems are provided; the
  molar system is recommended for the identifiability analysis and the eventual
  publication-grade model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# --------------------------------------------------------------------------- #
#  Physical constants                                                         #
# --------------------------------------------------------------------------- #
MW_HCG_G_PER_MOL: float = 37_000.0     # Lapthorn 1994 (36.7-37 kDa intact dimer)
MW_IGG_G_PER_MOL: float = 150_000.0    # capture antibody
N_AVOGADRO: float = 6.022_140_76e23
MIU_PER_NG_HCG: float = 9.3            # Partington 2018 anchor; range 8.4-16.8


# --------------------------------------------------------------------------- #
#  Unit conversions                                                           #
# --------------------------------------------------------------------------- #
def ng_per_mL_to_M(c_ng_per_mL: float) -> float:
    """hCG mass concentration [ng/mL] -> molar [M]. 1 ng/mL = 2.703e-11 M."""
    return c_ng_per_mL * 1e-6 / MW_HCG_G_PER_MOL


def M_to_ng_per_mL(c_M: float) -> float:
    return c_M * MW_HCG_G_PER_MOL / 1e-6


def mIU_per_mL_to_ng_per_mL(c_mIU_per_mL: float) -> float:
    """hCG activity [mIU/mL] -> mass [ng/mL] using the 9.3 mIU/ng anchor."""
    return c_mIU_per_mL / MIU_PER_NG_HCG


def ng_per_mL_to_mIU_per_mL(c_ng_per_mL: float) -> float:
    return c_ng_per_mL * MIU_PER_NG_HCG


def ug_per_cm2_to_mol_per_m2(sigma_ug_per_cm2: float, mw_g_per_mol: float) -> float:
    """Surface mass density [ug/cm^2] -> molar surface density [mol/m^2].
    1 ug/cm^2 = 1e-2 g/m^2; divide by MW. (IgG: 1 ug/cm^2 = 6.67e-8 mol/m^2.)"""
    return (sigma_ug_per_cm2 * 1e-2) / mw_g_per_mol


def mol_per_m2_to_ug_per_cm2(sigma_mol_per_m2: float, mw_g_per_mol: float) -> float:
    return sigma_mol_per_m2 * mw_g_per_mol / 1e-2


# --------------------------------------------------------------------------- #
#  Literature priors (central value + [low, high] bounds + provenance)        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Prior:
    central: float
    low: float
    high: float
    units: str
    confidence: str      # high | medium | low
    source: str

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.low, self.central, self.high)


# Binding kinetics — MOLAR system (as measured in the literature).
K_ON = Prior(1e6, 1e5, 5e6, "M^-1 s^-1", "medium", "Kamat 2017; Klonisch 1996 (SPR)")
K_OFF = Prior(1e-4, 4e-6, 1e-3, "s^-1", "medium", "Murthy 1996; Ashish 2004 (solid-phase/SPR)")
K_D = Prior(1e-9, 3e-11, 1e-8, "M", "high", "Berger 1984; Kamat 2017 (RIA/SPR)")

# Capture-site density — report gives mol/m^2; ~1 ug/cm^2 IgG = 6.67e-8 mol/m^2.
B_MAX = Prior(6.7e-8, 1e-8, 2e-7, "mol/m^2", "low",
              "derived from striping conc (Cate 2021; Walker 2023) — NO hCG-specific measurement")

# Transport.
D_HCG = Prior(5e-11, 4e-11, 6e-11, "m^2/s", "medium", "Stokes-Einstein, Rh~4-5 nm (glycosylated)")
U_FLOW = Prior(5e-5, 1.7e-5, 8.3e-5, "m/s", "medium", "Washburn wicking; Sathishkumar 2020")
K_M = Prior(1.6e-6, 3e-7, 5e-6, "m/s", "low", "k_m ~ sqrt(D*U/Lc), Lc~1 mm — INFER, do not fix")

# Front transport (lumped Washburn constant, model-specific placeholder).
K_WASH = Prior(2.0, 0.5, 5.0, "mm^2/s", "low", "lumped gamma*r*cos(theta)/(4*eta) — fit to front timing")

# Hook effect — WEAK for hCG-LFA (see verification note); bracket only.
HOOK_ONSET = Prior(200.0, 40.0, 1000.0, "IU/mL", "low",
                   "hCG-LFA onset NOT measured; bracket from CRP/serum analogues — MEASURE YOURS")


def damkohler(k_on_M: float = K_ON.central,
              B_max_mol_m2: float = B_MAX.central,
              k_m_m_s: float = K_M.central) -> float:
    """Da = k_on * B_max / k_m (all SI). Da > 1 => mass-transport limited.
    Central values give Da ~ 42 (verified)."""
    k_on_SI = k_on_M * 1e-3          # M^-1 s^-1 -> m^3 mol^-1 s^-1
    return k_on_SI * B_max_mol_m2 / k_m_m_s


def k_on_effective_M(k_on_M: float = K_ON.central,
                     B_max_mol_m2: float = B_MAX.central,
                     k_m_m_s: float = K_M.central) -> float:
    """Two-compartment effective on-rate k_on,eff = k_on*k_m/(k_on*B_max + k_m)
    [returned in M^-1 s^-1]. In the mass-transport-limited regime this collapses
    toward the transport ceiling and makes k_on / k_m confounded (see
    identifiability.py)."""
    k_on_SI = k_on_M * 1e-3
    k_on_eff_SI = k_on_SI * k_m_m_s / (k_on_SI * B_max_mol_m2 + k_m_m_s)
    return k_on_eff_SI * 1e3


def to_ode_config(**overrides):
    """Build a mechanistic_ode.ODEConfig seeded with literature priors converted
    to the model's (ng/mL, ug/cm^2) units. k_on is converted M^-1 s^-1 ->
    (ng/mL)^-1 s^-1 via x2.703e-11; B_max mol/m^2 -> ug/cm^2 via IgG MW."""
    from mechanistic_ode import ODEConfig

    conc_factor = 1e-6 / MW_HCG_G_PER_MOL                       # 2.703e-11
    k_on_model = K_ON.central * conc_factor                     # 1e6 -> 2.703e-5
    b_max_model = mol_per_m2_to_ug_per_cm2(B_MAX.central, MW_IGG_G_PER_MOL)

    defaults = dict(
        k_wash=K_WASH.central,
        k_on=k_on_model,
        k_off=K_OFF.central,
        B_max=b_max_model,
    )
    defaults.update(overrides)
    return ODEConfig(**defaults)


if __name__ == "__main__":
    print("== unit-conversion smoke test ==")
    print(f"1 ng/mL           = {ng_per_mL_to_M(1):.3e} M")
    print(f"25 mIU/mL         = {mIU_per_mL_to_ng_per_mL(25):.3f} ng/mL "
          f"= {ng_per_mL_to_M(mIU_per_mL_to_ng_per_mL(25)):.3e} M")
    print(f"1 ug/cm^2 IgG     = {ug_per_cm2_to_mol_per_m2(1, MW_IGG_G_PER_MOL):.3e} mol/m^2")
    print(f"B_max central     = {B_MAX.central:.2e} mol/m^2 "
          f"= {mol_per_m2_to_ug_per_cm2(B_MAX.central, MW_IGG_G_PER_MOL):.3f} ug/cm^2")
    print(f"Damkohler(central)= {damkohler():.1f}  (expect ~42 -> mass-transport limited)")
    print(f"k_on central      = {K_ON.central:.2e} M^-1 s^-1  "
          f"-> k_on,eff = {k_on_effective_M():.2e} M^-1 s^-1")
    cfg = to_ode_config()
    print(f"\nODEConfig from priors: k_on={cfg.k_on:.3e} (ng/mL)^-1 s^-1, "
          f"k_off={cfg.k_off:.1e} s^-1, B_max={cfg.B_max:.4f} ug/cm^2, k_wash={cfg.k_wash}")
