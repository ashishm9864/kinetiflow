"""
dataset.py
==========
Load the synthetic LFA traces and build LEAKAGE-SAFE grouped splits for
KinetiFlow-CP v2.

Splitting rule (non-negotiable, see CLAUDE.md): no experimental DAY and no strip
LOT may appear in more than one of {train, calibration, test}. Because a single
trace ties one day to one lot, days and lots are entangled: if day 3 and lot 7
ever co-occur, they must live in the same split. We therefore:

  1. Build a bipartite graph with a node per day and per lot; every trace adds an
     edge day <-> lot.
  2. Union-find the connected components. Each component is an ATOMIC group — the
     smallest set of days+lots that can move between splits without leaking.
  3. Greedily pack whole components into train / cal / test to approach the
     60 / 20 / 20 target by trace count.

This guarantees, by construction, zero day and zero lot overlap across splits;
the __main__ self-test asserts it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import Tensor


DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "traces.pt"


# --------------------------------------------------------------------------- #
#  Trace bundle                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class TraceBundle:
    """A split's worth of traces as a bundle of aligned tensors."""
    name: str
    idx: Tensor              # [n] original row indices into the full dataset
    t: Tensor                # [T] shared time grid
    I_obs: Tensor            # [n, T] noisy observed intensity
    C_f0: Tensor             # [n] initial free-analyte conc (z0 seed)
    T_ambient: Tensor        # [n] covariate
    RH: Tensor               # [n] covariate
    day: Tensor              # [n]
    lot: Tensor              # [n]
    concentration_level: Tensor  # [n]
    true_I_900: Tensor       # [n] noise-free 900 s equilibrium target

    def __len__(self) -> int:
        return int(self.idx.numel())

    def days(self) -> set:
        return set(int(x) for x in self.day.tolist())

    def lots(self) -> set:
        return set(int(x) for x in self.lot.tolist())


# --------------------------------------------------------------------------- #
#  Loading                                                                    #
# --------------------------------------------------------------------------- #
def load(path: str | Path = DEFAULT_PATH) -> Dict[str, Tensor]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python src/synthetic.py` first.")
    return torch.load(path, weights_only=False)


# --------------------------------------------------------------------------- #
#  Union-find for atomic (day, lot) groups                                    #
# --------------------------------------------------------------------------- #
class _UnionFind:
    def __init__(self):
        self.parent: Dict[object, object] = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:            # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _atomic_groups(day: Tensor, lot: Tensor) -> List[List[int]]:
    """Return lists of trace indices, one per connected (day/lot) component."""
    uf = _UnionFind()
    n = int(day.numel())
    for i in range(n):
        uf.union(("day", int(day[i])), ("lot", int(lot[i])))

    groups: Dict[object, List[int]] = {}
    for i in range(n):
        root = uf.find(("day", int(day[i])))
        groups.setdefault(root, []).append(i)
    # deterministic order: largest component first
    return sorted(groups.values(), key=len, reverse=True)


# --------------------------------------------------------------------------- #
#  Grouped split                                                              #
# --------------------------------------------------------------------------- #
def grouped_split(
    data: Dict[str, Tensor],
    fracs: Tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> Tuple[TraceBundle, TraceBundle, TraceBundle]:
    """Leakage-safe train/cal/test split grouped by (day AND lot).

    Whole (day, lot) connected components are packed greedily so each split
    approaches its target trace fraction; days and lots never cross splits.
    """
    day, lot = data["day"], data["lot"]
    N = int(day.numel())
    groups = _atomic_groups(day, lot)

    names = ["train", "cal", "test"]
    targets = [f * N for f in fracs]
    assigned: Dict[str, List[int]] = {n: [] for n in names}

    # Greedy: give each component to the split most under-filled vs its target.
    for members in groups:
        deficits = [targets[k] - len(assigned[names[k]]) for k in range(3)]
        pick = int(max(range(3), key=lambda k: deficits[k]))
        assigned[names[pick]].extend(members)

    bundles = tuple(_make_bundle(n, sorted(assigned[n]), data) for n in names)
    return bundles  # type: ignore[return-value]


def _make_bundle(name: str, indices: List[int], data: Dict[str, Tensor]) -> TraceBundle:
    idx = torch.tensor(indices, dtype=torch.long)
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
        concentration_level=data["concentration_level"][idx],
        true_I_900=data["true_I_900"][idx],
    )


def assert_no_leakage(train: TraceBundle, cal: TraceBundle, test: TraceBundle) -> None:
    """Raise AssertionError if any day or lot appears in more than one split."""
    for a, b in ((train, cal), (train, test), (cal, test)):
        day_overlap = a.days() & b.days()
        lot_overlap = a.lots() & b.lots()
        assert not day_overlap, f"DAY leakage {a.name}<->{b.name}: {day_overlap}"
        assert not lot_overlap, f"LOT leakage {a.name}<->{b.name}: {lot_overlap}"


# --------------------------------------------------------------------------- #
#  Self-test                                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    data = load()
    N = int(data["day"].numel())
    train, cal, test = grouped_split(data)

    print("== grouped split (leakage-safe by day AND lot) ==")
    for b in (train, cal, test):
        frac = len(b) / N
        print(f"   {b.name:5s}: {len(b):3d} traces ({frac*100:4.1f}%)  "
              f"days={sorted(b.days())}  lots={sorted(b.lots())}")

    assert_no_leakage(train, cal, test)
    assert len(train) + len(cal) + len(test) == N, "split does not cover all traces"

    # Explicit overlap report for the acceptance test.
    print("\n== leakage check ==")
    print(f"   train/cal  day overlap : {sorted(train.days() & cal.days())}")
    print(f"   train/test day overlap : {sorted(train.days() & test.days())}")
    print(f"   cal/test   day overlap : {sorted(cal.days() & test.days())}")
    print(f"   train/cal  lot overlap : {sorted(train.lots() & cal.lots())}")
    print(f"   train/test lot overlap : {sorted(train.lots() & test.lots())}")
    print(f"   cal/test   lot overlap : {sorted(cal.lots() & test.lots())}")
    print("\nNO DAY OR LOT LEAKAGE: PASS")
