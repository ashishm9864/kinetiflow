"""Leakage-safe grouped data roles and persisted split manifests."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "traces.pt"
ROLE_FRACTIONS: Mapping[str, float] = {
    "train": 0.50,
    "validation": 0.10,
    "wr_reference": 0.10,
    "calibration": 0.15,
    "test": 0.15,
}


@dataclass
class TraceBundle:
    name: str
    idx: Tensor
    t: Tensor
    I_obs: Tensor
    C_f0: Tensor
    T_ambient: Tensor
    RH: Tensor
    day: Tensor
    lot: Tensor
    group_id: Tensor
    concentration_level: Tensor
    true_I_900: Tensor

    def __len__(self) -> int:
        return int(self.idx.numel())

    def days(self) -> set[int]:
        return set(map(int, self.day.tolist()))

    def lots(self) -> set[int]:
        return set(map(int, self.lot.tolist()))

    def groups(self) -> set[int]:
        return set(map(int, self.group_id.tolist()))


def load(path: str | Path = DEFAULT_PATH) -> Dict[str, Tensor]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run python src/synthetic.py")
    data = torch.load(path, weights_only=False)
    if "group_id" not in data:
        data["group_id"] = data["lot"].clone()
    return data


class _UnionFind:
    def __init__(self):
        self.parent: Dict[object, object] = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def atomic_groups(day: Tensor, lot: Tensor) -> List[List[int]]:
    """Connected day/lot components; the indivisible unit of assignment."""
    union = _UnionFind()
    for i in range(day.numel()):
        union.union(("day", int(day[i])), ("lot", int(lot[i])))
    groups: Dict[object, List[int]] = {}
    for i in range(day.numel()):
        groups.setdefault(union.find(("day", int(day[i]))), []).append(i)
    return sorted(groups.values(), key=lambda x: min(x))


def _role_counts(n_groups: int, fractions: Mapping[str, float]) -> Dict[str, int]:
    raw = {name: n_groups * frac for name, frac in fractions.items()}
    counts = {name: int(np.floor(value)) for name, value in raw.items()}
    remaining = n_groups - sum(counts.values())
    order = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    if any(value == 0 for value in counts.values()):
        raise ValueError(f"not enough atomic groups for all roles: {counts}")
    return counts


def _make_bundle(name: str, indices: Sequence[int], data: Dict[str, Tensor]) -> TraceBundle:
    idx = torch.tensor(sorted(map(int, indices)), dtype=torch.long)
    return TraceBundle(
        name=name,
        idx=idx,
        t=data["t"],
        I_obs=data["I_obs"][idx],
        C_f0=data["C_f0"][idx],
        T_ambient=data["T_ambient"][idx],
        RH=data["RH"][idx],
        day=data["day"][idx],
        lot=data["lot"][idx],
        group_id=data.get("group_id", data["lot"])[idx],
        concentration_level=data["concentration_level"][idx],
        true_I_900=data["true_I_900"][idx],
    )


def grouped_role_split(
    data: Dict[str, Tensor],
    seed: int = 0,
    fractions: Mapping[str, float] = ROLE_FRACTIONS,
) -> Dict[str, TraceBundle]:
    """Randomly assign whole atomic day/lot components to disjoint roles."""
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError("role fractions must sum to one")
    groups = atomic_groups(data["day"], data["lot"])
    counts = _role_counts(len(groups), fractions)
    order = np.random.default_rng(seed).permutation(len(groups)).tolist()
    bundles: Dict[str, TraceBundle] = {}
    cursor = 0
    for role in fractions:
        chosen = order[cursor:cursor + counts[role]]
        cursor += counts[role]
        indices = [i for group_index in chosen for i in groups[group_index]]
        bundles[role] = _make_bundle(role, indices, data)
    assert_no_leakage(*bundles.values())
    return bundles


def grouped_split(data: Dict[str, Tensor]) -> Tuple[TraceBundle, TraceBundle, TraceBundle]:
    """Compatibility wrapper; new code must use :func:`grouped_role_split`."""
    roles = grouped_role_split(data)
    train_indices = torch.cat([roles["train"].idx, roles["validation"].idx, roles["wr_reference"].idx])
    return (
        _make_bundle("train", train_indices.tolist(), data),
        roles["calibration"],
        roles["test"],
    )


def assert_no_leakage(*bundles: TraceBundle) -> None:
    for i, left in enumerate(bundles):
        for right in bundles[i + 1:]:
            day_overlap = left.days() & right.days()
            lot_overlap = left.lots() & right.lots()
            if day_overlap or lot_overlap:
                raise AssertionError(
                    f"group leakage {left.name}<->{right.name}: days={day_overlap}, lots={lot_overlap}"
                )


def manifest(bundles: Mapping[str, TraceBundle], seed: int) -> Dict:
    return {
        "seed": seed,
        "roles": {
            name: {
                "n": len(bundle),
                "indices": bundle.idx.tolist(),
                "days": sorted(bundle.days()),
                "lots": sorted(bundle.lots()),
                "groups": sorted(bundle.groups()),
            }
            for name, bundle in bundles.items()
        },
    }


def save_manifest(bundles: Mapping[str, TraceBundle], seed: int, path: str | Path) -> None:
    Path(path).write_text(json.dumps(manifest(bundles, seed), indent=2) + "\n")


if __name__ == "__main__":
    roles = grouped_role_split(load(), seed=0)
    for name, bundle in roles.items():
        print(f"{name:12s} n={len(bundle):3d} groups={sorted(bundle.groups())}")
    assert_no_leakage(*roles.values())
    print("G7 ZERO DAY/LOT OVERLAP: PASS")
