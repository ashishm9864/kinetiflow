"""Prespecified synthetic-mismatch magnitude and mechanism ablation.

This study deliberately removes hierarchical day/lot effects so it isolates the
one thing synthetic data can answer: whether each learner behaves sensibly as a
known simulator mismatch is varied.  It cannot establish that the mismatch or a
gray-box advantage exists in a commercial strip.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

import dataset as D
from evaluate import RESULTS_DIR, mae, rmse, _frame_markdown
from interfaces import inputs_from_bundle
import synthetic
import train as GB


CASES = [
    ("combined", 0.0),
    ("combined", 0.25),
    ("combined", 0.5),
    ("combined", 1.0),
    ("combined", 2.0),
    ("supply", 1.0),
]


def run(epochs: int = 80, patience: int = 20) -> pd.DataFrame:
    rows = []
    for mechanism, scale in CASES:
        config = synthetic.SyntheticConfig(
            mismatch_mechanism=mechanism,
            mismatch_scale=scale,
            group_effect_scale=0.0,
            seed=70_000 + int(scale * 100) + (1_000 if mechanism == "supply" else 0),
        )
        started = time.time()
        data = synthetic.generate(config)
        roles = D.grouped_role_split(data, seed=0)
        for dynamics in ("physics_only", "residual_only", "graybox"):
            result = GB.train(
                roles["train"], roles["validation"],
                GB.TrainConfig(
                    dynamics=dynamics, epochs=epochs, patience=patience,
                    seed=0, tag=f"{dynamics}_{mechanism}_{scale}",
                ),
            )
            model = result["model"]
            inputs = inputs_from_bundle(roles["test"], 60.0)
            prediction = model.predict(inputs)
            inferred = model.infer_cf0(inputs)
            rows.append({
                "mechanism": mechanism,
                "mismatch_scale": scale,
                "model": dynamics,
                "rmse_dn": rmse(prediction, roles["test"].true_I_900),
                "mae_dn": mae(prediction, roles["test"].true_I_900),
                "cf0_rmse_ng_ml": rmse(inferred, roles["test"].C_f0),
                "cf0_mae_ng_ml": mae(inferred, roles["test"].C_f0),
                "best_validation_loss": result["best_validation"],
                "epochs_run": result["epochs_run"],
            })
        print(
            f"mismatch {mechanism:8s} scale={scale:4.2f}: "
            f"{time.time() - started:.1f}s",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS_DIR / "mismatch_ablation.csv", index=False)
    pivot = frame.pivot_table(
        index=["mechanism", "mismatch_scale"], columns="model", values="rmse_dn"
    ).reset_index()
    lines = [
        "# Synthetic mismatch ablation",
        "",
        "Day/lot effects are disabled here to isolate planted mismatch magnitude. Values are held-out I(900 s) endpoint RMSE in DN.",
        "",
        _frame_markdown(pivot),
        "",
        "This validates pipeline behavior only. A residual advantage on a perturbation chosen by the simulator author is not evidence that commercial hCG strips contain that perturbation.",
        "",
    ]
    (RESULTS_DIR / "mismatch_ablation.md").write_text("\n".join(lines))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()
    frame = run(args.epochs, args.patience)
    print(frame.to_string(index=False))
    print(f"wrote {RESULTS_DIR / 'mismatch_ablation.csv'}")


if __name__ == "__main__":
    main()
