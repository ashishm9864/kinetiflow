"""Restricted, deployment-safe model interfaces.

Forecast models receive :class:`ForecastInputs`, never a dataset bundle carrying
concentration, day/lot identity, or targets.  This makes answer leakage difficult
by construction instead of relying on call-site discipline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class ForecastInputs:
    """The complete deployable information set for one batch.

    ``I_early`` contains only observations at ``t <= window_s``.  Temperature
    and humidity are measured ambient covariates available to both comparators.
    """

    t: Tensor                 # [W], seconds from front arrival
    I_early: Tensor           # [B, W], optical intensity [DN]
    T_ambient: Tensor         # [B], deg C
    RH: Tensor                # [B], percent

    def __post_init__(self) -> None:
        if self.t.ndim != 1 or self.I_early.ndim != 2:
            raise ValueError("ForecastInputs expects t[W] and I_early[B,W]")
        if self.I_early.shape[1] != self.t.numel():
            raise ValueError("time/intensity length mismatch")
        n = self.I_early.shape[0]
        if self.T_ambient.shape != (n,) or self.RH.shape != (n,):
            raise ValueError("T/RH batch size mismatch")
        for name, value in vars(self).items():
            if not torch.isfinite(value).all():
                raise ValueError(f"non-finite forecast input: {name}")

    def __len__(self) -> int:
        return int(self.I_early.shape[0])


class PointForecaster(Protocol):
    def predict(self, inputs: ForecastInputs, t_end: float = 900.0) -> np.ndarray:
        """Return endpoint predictions [B] without reading targets."""


def inputs_from_bundle(bundle, window_s: float = 60.0) -> ForecastInputs:
    """Extract only the allowed fields from a TraceBundle-like object."""
    mask = bundle.t <= float(window_s) + 1e-6
    if int(mask.sum()) < 2:
        raise ValueError(f"window {window_s}s contains fewer than two observations")
    return ForecastInputs(
        t=bundle.t[mask].float().clone(),
        I_early=bundle.I_obs[:, mask].float().clone(),
        T_ambient=bundle.T_ambient.float().clone(),
        RH=bundle.RH.float().clone(),
    )


def as_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)
