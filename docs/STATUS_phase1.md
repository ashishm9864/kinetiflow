# KinetiFlow-CP v2 — Phase 1 Status Audit

**Date:** 2026-07-11 · **Mode:** read-only audit (nothing fixed) · **Env:** `.venv` Python
3.14.4, torch 2.13.0, torchdiffeq 0.2.5, numpy 2.5.1 (note: `src/CLAUDE.md` and the
outline say Python 3.11 — environment has drifted to 3.14).

> Honesty note: every PASS/FAIL below was produced by running the code, not by reading
> intent. Test artifacts that the runs overwrote (`checkpoints/graybox_best.pt`,
> `figures/training_loss.png`) were restored via `git checkout` so the repo is left
> pristine; `data/synthetic/*` and `figures/identifiability_gate.png` regenerated
> bit-identically. The **headline bad news is in section D (train.py) and F.**

---

## A. Inventory

Line counts from `wc -l`; binary sizes in bytes. "INCLUDED but not asked for" flagged.

### `src/`
| Lines | File | What it actually is |
|------:|------|---------------------|
| 353 | `mechanistic_ode.py` | Gray-box UDE (`MechanisticODE`), separate `MeasurementModel`, `integrate()`, gradient-flow self-test. Core module. |
| 340 | `synthetic.py` | Synthetic LFA generator (`TruthODE` = known physics + injected known mismatch), writes `data/synthetic/*`, early-window separation self-test. |
| 312 | `identifiability.py` | Fisher/sensitivity local-identifiability analysis + figure. |
| 433 | `train.py` | Training loop (gray-box + `physics_only` baseline), checkpoint I/O, loss-curve PNG. |
| 192 | `dataset.py` | Leakage-safe grouped splits via union-find over day↔lot edges. |
| 169 | `priors.py` | Literature priors, unit conversions, `to_ode_config()`. |
| 141 | `loss_functions.py` | `observation_loss` + `physics_bounds_loss` + combined. |
| 40 | `CLAUDE.md` | Project-context/constraints doc (this is the **only** CLAUDE.md; there is no root one). |
| 0 | `conformal.py` | **EMPTY (0 bytes)** — Phase-2 conformal layer not started. Unexpected only in that it is committed empty. |
| 0 | `data_pipeline.py` | **EMPTY (0 bytes)** — OpenCV video/ROI pipeline not started. |
| 28 | `_probe2.py` | **UNEXPECTED** — leftover debug scratch script (default-normalization training probe). |
| 32 | `_probe3.py` | **UNEXPECTED** — leftover debug scratch script (matched-normalization training probe). |

### `data/synthetic/`
| Size/Lines | File | What |
|------:|------|------|
| 157,605 B | `traces.pt` | dict of tensors: 200 traces × 91 timepoints + metadata. |
| 201 lines | `metadata.csv` | 200 traces + header (idx, level, day, lot, T, RH, C_f0, true_I_900). |

### `checkpoints/`
| Size | File | What |
|------:|------|------|
| 39,457 B | `graybox_best.pt` | Best gray-box model+optics+cal-pred reference. **Provenance caveat:** re-running `train.py` produces a *non-identical* checkpoint (it showed as modified after re-run), so the committed one is from an earlier/different run, not reproducible bit-for-bit from current code. |

### `figures/`
| Size | File | What |
|------:|------|------|
| 178,240 B | `identifiability_gate.png` | Identifiability 4-panel figure (regenerated identically). |
| 117,301 B | `training_loss.png` | Train/cal loss curves (regenerated; restored committed copy). |

### `tests/`
**MISSING — there is no `tests/` directory.** All testing is per-module `__main__` self-tests.

### Files outside the requested dirs (INCLUDED because you asked for the unexpected)
| Lines/Size | File | What |
|------:|------|------|
| 16 | `main.py` (repo root) | **PyCharm boilerplate** (`print_hi('PyCharm')`). Not project code; committed. |
| 0 B | `Login` (repo root) | **Empty stray file, committed.** No content, no extension. Almost certainly an accident. |
| 489 | `docs/CLAUDE final check.md` | The full project proposal/outline (note the space in the filename). |
| 162 | `docs/hCG_LFA_parameters_report.md` | Literature parameter report. |
| 30 | `docs/parameters.csv` | Literature parameter table w/ DOIs. |
| 177,488 B | `docs/identifiability_gate.png` | **Duplicate** of `figures/` version but a **different render** (`cmp` differs at byte 72). |
| 166,435 B | `docs/damkohler_regime_map.png` | **ORPHAN** — no script in the repo generates it. Provenance **UNSURE**. |

---

## B. Module status

**1. `mechanistic_ode.py` — DONE (with one self-test caveat).**
Identifiable training mode exists and is correct: `MechanisticODE.identifiable()`
(lines 174–190) builds a config with `trainable_physics=False`, freezing
`k_wash/k_on/k_off/B_max` and leaving `alpha,beta` (in `MeasurementModel`) + the neural
residual trainable. Gradient-flow self-test prints **`GRADIENT FLOW: PASS`**.
*Caveat:* the `__main__` acceptance test instantiates `ODEConfig()` directly (line 294)
with `trainable_physics=True` and the hardcoded placeholder priors — so the shipped
smoke test does **not** exercise the frozen/priors path that training actually uses.

**2. `synthetic.py` — DONE.** Generates 200 traces; injects a mass-conserving known
mismatch (cooperative + hook + temp terms, lines 125–133) into C_f/C_b only.
Separation self-test **PASS (5.16×, target > 3×)**.

**3. `dataset.py` — DONE.** Union-find groups day↔lot into atomic components, greedy-packs
to 60/20/20. Self-test **PASS — zero day and zero lot leakage**.

**4. `train.py` — PARTIAL (runs end-to-end but FAILS its own acceptance gate).**
Works: builds identifiable core, trains gray-box + `physics_only` baseline, early stops,
saves checkpoint, exact reload. **Broken/incomplete:**
- Calibration-drop acceptance **FAILS: −21.7% vs > 50% target** (section D).
- The fix for this is present but **not wired in**: `_build_core()`, `RES_MEAN`,
  `RES_SCALE` (lines 48–57) are **dead code** — `train()` calls
  `MechanisticODE.identifiable()` at line 220 *without* the operating-range-matched
  normalization, using the ODEConfig defaults its own comment says "blinds the residual."
- **Dead config:** `sched_factor/sched_patience/min_lr` (lines 70–72) imply an
  LR scheduler that `train()` never builds — LR is constant.
- Runtime warning at line 263: `float(total)` on a grad-requiring tensor (missing
  `.detach()`); harmless but sloppy.

**5. Everything else:**
`priors.py` — DONE (unit smoke test passes, Damköhler ≈ 41.9).
`loss_functions.py` — DONE (self-test PASS).
`identifiability.py` — DONE (runs, writes figure, verdict coherent).
`conformal.py`, `data_pipeline.py` — **MISSING (empty stubs).**
`_probe2.py`, `_probe3.py` — debug leftovers (not part of the pipeline).
`main.py` — boilerplate. `Login` — stray empty file.

---

## C. Scientific constraint audit

### C1 — Integration uses `odeint_adjoint` (continuous adjoint)? **PASS**
The *training* path uses the continuous adjoint:
- `train.py:171` `solver = odeint_adjoint if use_adjoint else odeint`
- `train.py:250` (training loop) `... use_adjoint=True`
- `mechanistic_ode.py:279` same dispatch; `integrate()` default `use_adjoint=True`
  (`:257`), self-test calls it (`:307`).
Non-adjoint `odeint` is used only where it should be: `eval_loss`/`init_optics`
(`train.py:187,208`, both `@torch.no_grad`/no-grad), and data-gen/sensitivity
(`synthetic.py`, `identifiability.py`). Correct.

### C2 — Washburn front is 1/L, NOT sqrt? **PASS**
- `mechanistic_ode.py:197` `dL = self.k_wash / L_safe`
- `identifiability.py:100` `dL = k_wash / L_safe`
The sqrt form appears **only** as the explicitly-rejected form in NOTE 1
(`mechanistic_ode.py:32–36`). ⚠️ The outline still carries the wrong form:
`docs/CLAUDE final check.md:193` `dL/dt = sqrt( γ_r / 8ηL(t) )` — the code correctly
overrides the outline.

### C3 — Initial front L0 > 0 (~5 mm) everywhere? **PASS**
`mechanistic_ode.py:303` `z0 = [5.0, ...]`; `synthetic.py:84` `L0: float = 5.0`, used
at `:208` and `:253`; `identifiability.py:74` `Z0 = [5.0, ...]`; `train.py:80` `L0=5.0`
→ `make_batch` `torch.full((B,), L0)` (`:152`). 1/L guarded by `clamp_min(eps)`
(`mechanistic_ode.py:195`, `eps=1e-6`).

### C4 — Neural residual corrects ONLY C_f, C_b (not L, not I_obs)? **PASS (verified by reading)**
`mechanistic_ode.py:209–213`: residual out-dim = 2; `delta[...,0]→dC_f`,
`delta[...,1]→dC_b`; `dL` is the raw physics term, untouched. Residual input is
`(C_f, C_b, t, T, RH)` — L is not even an input. In `GrayBoxUDE.forward`
(`train.py:139–142`) the residual-scaling likewise only adjusts channels 1,2; `dL =
full[...,0]` passes through. I_obs is computed by a separate model (C5), never by the
residual.

### C5 — Optical model separate from the ODE? **PASS**
`MeasurementModel` is its own `nn.Module` (`mechanistic_ode.py:219–244`), applied to the
solved `C_b` **after** integration (`:284` `I_pred = measurement(z_traj[...,2])`;
`train.py:173`). It is not part of the vector field.

### C6 — In training mode, are k_on/k_off/B_max frozen and only alpha/beta/residual trainable? **PASS (runtime proof below)**
Printed at runtime from `MechanisticODE.identifiable()` and the exact `train()` setup:
```
=== MechanisticODE.identifiable() — physics params ===
  log_k_wash   requires_grad=False  [frozen]
  log_k_on     requires_grad=False  [frozen]
  log_k_off    requires_grad=False  [frozen]
  log_B_max    requires_grad=False  [frozen]
=== Actual train() gray-box trainable set ===
  TRAINABLE: core.residual.0/2/4 weight+bias, meas.alpha, meas.beta   (residual=4674 params)
  FROZEN:    core.log_k_wash, core.log_k_on, core.log_k_off, core.log_B_max
=== physics_only baseline ===
  TRAINABLE: ['meas.alpha', 'meas.beta']   (residual also frozen)
```

### C7 — Import from priors.py, or hardcoded? **MIXED**
The **core kinetics do import from priors** via `to_ode_config()`
(`priors.py:136–153` → runtime: `k_on=2.703e-5, k_off=1e-4, B_max=1.005 µg/cm², k_wash=2.0`).
But hardcoded physical constants remain:
- `mechanistic_ode.py:70–73` — `ODEConfig` placeholder defaults `k_wash=2.0, k_on=1e-4,
  k_off=1e-3, B_max=1.0` (NOTE 2 admits these are non-physical placeholders). These are
  what the `__main__` self-test actually runs on.
- `mechanistic_ode.py:229–230` — `MeasurementModel` defaults `alpha=100.0, beta=20.0`.
- `identifiability.py:66–67,74` — `alpha=100.0, beta=20.0`, `Z0` seed `125 mIU/mL`.
- `synthetic.py:55–56,69–71,84` — `alpha_true=2000.0, beta_true=25.0`,
  `coop_gain=4.0, hook_gain=0.06, temp_coeff=0.10`, `L0=5.0`.
- `train.py:48–49,75,80,223` — `RES_MEAN/RES_SCALE`, `residual_scale=8e-4`, `L0=5.0`,
  `MeasurementModel(alpha=100.0, beta=20.0)`.
Verdict: no magic numbers in the *frozen kinetics*, but optics init, L0, and the
synthetic perturbation gains are hardcoded across modules.

### C8 — Splits group by day AND lot with ZERO overlap? **PASS (proof from `python src/dataset.py`)**
```
train: 120 traces (60.0%)  days=[0,1,2,3]  lots=[0,1,2]
cal  :  40 traces (20.0%)  days=[4,5]      lots=[3]
test :  40 traces (20.0%)  days=[6,7]      lots=[4]
train/cal, train/test, cal/test  day overlap : [] [] []
train/cal, train/test, cal/test  lot overlap : [] [] []
NO DAY OR LOT LEAKAGE: PASS
```

### C9 — Hardcoded absolute paths? **PASS (none)**
`grep -rnE "/home/|/Users/|C:\\|/root/|/tmp/" src/ main.py` → no matches. All paths are
`Path(__file__).resolve().parents[1] / ...` (portable).

---

## D. Test results (real output, trimmed)

**`python src/mechanistic_ode.py`**
```
monotone L   : True
C_b <= B_max : True
all finite   : True
--- gradient-flow check (continuous adjoint) ---  [all 12 params [ ok ]]
GRADIENT FLOW: PASS
```
→ **mechanistic_ode: "GRADIENT FLOW: PASS" — YES.** ✅

**`python src/synthetic.py`** (0–60 s early-window separation)
```
between-level spread : 2.253 DN
within-level noise   : 0.437 DN
ratio (want > 3x)    : 5.16x
SEPARATION: PASS
```
→ **between=2.253, within=0.437, ratio=5.16× (> 3×) — PASS.** ✅

**`python src/dataset.py`** → `NO DAY OR LOT LEAKAGE: PASS`. ✅

**`python src/train.py`** (4:03 wall; baseline then gray-box)
```
UserWarning: train.py:263 float(total) on requires_grad tensor  (missing .detach())
baseline best cal loss = 5.0270 (epoch 58)
gray-box best cal loss = 3.9348 (epoch 74)
================ ACCEPTANCE (D) ================
gray-box cal loss: epoch0=5.0272 -> best=3.9348  (-21.7%)   FAIL (>50%)
all gradients finite: True  PASS
gray-box (3.9348) beats physics_only (5.0270): True (residual improves cal by 21.7%)  PASS
checkpoint reload max|Δpred| = 0.00e+00  PASS (<1e-5)
```
→ **Calibration drop = −21.7% → FAILS the > 50% gate.**
→ **Gray-box (3.9348) DOES beat physics_only baseline (5.0270) — PASS**, but by exactly
the same 21.7% (at epoch 0 the residual is zero, so gray-box epoch0 ≈ baseline optimum).
→ Reload reproduction exact (0.0). Gradients finite.

**Root-cause A/B (ran the two leftover probes):**
```
_probe2.py  DEFAULT normalization  (== what train.py runs):  cal0=5.027 -> 3.812  decrease=24.2%  (FAIL)
_probe3.py  MATCHED normalization  (the un-wired _build_core): cal0=5.027 -> 2.498  decrease=50.3%  (PASS)
```
The only material difference is the residual input normalization. `train()` uses the
default (C_b centered at 0.5, scale 0.5) which compresses the real C_b (~0–0.06) to a
near-constant feature; the matched normalization (`RES_MEAN/RES_SCALE`, C_b scale 0.025)
recovers the cooperative-term signal and clears 50%. **This is the difference between
FAIL and PASS, and the fix is already written but not connected (section F).**

**`python src/priors.py`** — PASS: `Damköhler(central)=41.9`, `1 ng/mL=2.703e-11 M`,
`B_max=1.005 µg/cm²`, `ODEConfig k_on=2.703e-5, k_off=1e-4, B_max=1.005, k_wash=2.0`.

**`python src/loss_functions.py`** — `penalty on violating state (expect 7): 7.0` →
`self-test: PASS`.

**`python src/identifiability.py`** — runs clean; key result: single-compartment Fisher
`kappa=1.1e19` (60 s) / `2.0e13` (900 s), **B_max↔alpha corr = +1.0000** (perfectly
confounded); two-compartment rank 6/7 (one exactly-flat direction), **k_on↔k_m corr =
+1.0000**. This is the quantitative justification for freezing the confounded members —
consistent with the identifiable() training mode.

No test errored; no tracebacks to paste.

---

## E. Shortcuts, stubs, and TODOs (blunt)

1. **`src/conformal.py` and `src/data_pipeline.py` are 0-byte empty files** — the entire
   conformal-abstention layer (the project's headline novelty) and the OpenCV video
   pipeline are **not started**, only committed as empty placeholders.
2. **`train.py` un-wired fix / dead code:** `_build_core()`, `RES_MEAN`, `RES_SCALE`
   (`train.py:48–57`) are defined and never called. This is the operating-range-matched
   normalization that turns the FAIL (24%) into a PASS (50%). See F.
3. **`train.py` dead config:** `sched_factor`, `sched_patience`, `min_lr` (`:70–72`) —
   `train()` never constructs a scheduler, so the residual LR never decays. (The probes
   build the scheduler manually; `train()` does not.)
4. **`train.py:263`** `history["train"].append(float(total))` on a grad-requiring tensor
   → `UserWarning`. Needs `.detach()`.
5. **`mechanistic_ode.py` self-test doesn't test the real mode:** `__main__` uses
   `ODEConfig()` (`trainable_physics=True`, placeholder priors), so "GRADIENT FLOW: PASS"
   is proven for the *unfrozen* model, not the frozen/priors training config.
6. **Placeholder physics priors:** `ODEConfig` defaults (`:70–73`) and NOTE 2 admit the
   rate constants are unit-mismatched stand-ins ("PLACEHOLDER priors"). The synthetic
   perturbation gains (`coop/hook/temp`) are invented, not literature-derived (which is
   fine for synthetic data, but is a fabricated signal — the model is learning a mismatch
   *we planted*, not a real one).
7. **`_probe2.py`, `_probe3.py`** — debug scratch scripts committed into `src/`.
8. **`Login`** (empty) and **`main.py`** (PyCharm boilerplate) committed at repo root.
9. **`docs/damkohler_regime_map.png`** — orphan figure with no generator in the repo
   (provenance UNSURE); **`docs/identifiability_gate.png`** duplicates `figures/` with a
   different render.
10. **`requirements.txt`** lists `opencv-python`, `scikit-learn`, `pandas`, `scipy` — none
    are imported by any current module (CSV is written by hand in `synthetic.py`); the
    conformal `pot` dep is commented out. Env is Python 3.14, not the stated 3.11.
11. **Minor:** `synthetic.py` prints "1%" noise while `noise_frac=0.013` (1.3%, rounded).
12. **Prose/impl mismatch:** `identifiability.py:311` verdict says "treat the optical scale
    alpha and B_max as the tunable pair," but the model **freezes** B_max and trains only
    alpha (correct, since corr=1.0 — you can train only one; the prose is loose).

---

## F. Where it stopped

**The session died mid-tuning of the gray-box residual, one wire away from passing the
training gate.** The trail: the author found that `train()`'s default residual
normalization blinds the model to the C_b (cooperative) signal, wrote the corrected
normalization as `RES_MEAN`/`RES_SCALE` + `_build_core()` in `train.py`, verified the fix
in two scratch probes (`_probe2` default = 24% FAIL vs `_probe3` matched = 50% PASS) —
**but never replaced `MechanisticODE.identifiable()` at `train.py:220` with `_build_core()`.**
So the committed `train.py` still runs the failing configuration.

**Single next unfinished thing:** wire the operating-range-matched normalization into
`train()` (use `_build_core()` instead of the bare `identifiable()` call at `train.py:220`),
re-run `python src/train.py`, and confirm the calibration drop clears > 50%. Then delete
the `_probe*.py` scratch files. *(Reported only — not done, per scope. Note the probe
result is 50.3%, i.e. barely over the line, so it should be confirmed in the full
pipeline, not assumed.)*

**After that, still unstarted for later phases:** `conformal.py` (SCP/WR-CP abstention),
`data_pipeline.py` (OpenCV ROI + Savitzky-Golay), and Baseline B (the 1D-TCN) — none exist.

---

## G. Git state

`git status` → **clean working tree** (after I restored the two test-overwritten
artifacts; nothing uncommitted otherwise).

```
git log --oneline -10
e0e8ff9 Partial: recovered work from Claude Code session 1
e53f50c Add CLAUDE.md project context   (added src/CLAUDE.md, 41 lines — there is no root CLAUDE.md)
841186f Phase 1 start: verified core modules + research docs
```

**Committed vs uncommitted:** everything is committed — all 26 tracked files, including
the empty stubs (`conformal.py`, `data_pipeline.py`), the debug probes (`_probe2/3.py`),
the stray `Login`, PyCharm `main.py`, and all binary artifacts (`traces.pt`,
`graybox_best.pt`, PNGs). There are no uncommitted changes and no untracked project files.
Repo-hygiene smell: generated artifacts and scratch files are versioned alongside source.

---

## AUDIT COMPLETE

- **train.py fails its own acceptance gate: calibration drop −21.7% vs the > 50% target.**
  The fix is already written but left as dead code (`_build_core`/`RES_MEAN`/`RES_SCALE`)
  and never wired into `train()`; the leftover probes prove matched normalization reaches
  50.3%. This is exactly where the previous session stopped.
- **All non-negotiable scientific constraints (C1–C6, C8, C9) PASS**, verified by running
  the code: continuous adjoint, 1/L Washburn, L0=5 mm, residual touches only C_f/C_b,
  separate optics, frozen kinetics (runtime-proven), zero day/lot leakage, no absolute
  paths. C7 is mixed: kinetics import from priors, but optics/L0/synthetic gains are
  hardcoded.
- **Core self-tests PASS:** `GRADIENT FLOW: PASS`, separation 5.16×, zero leakage,
  identifiability confirms perfect B_max↔alpha and k_on↔k_m confounding — the science
  scaffolding is sound.
- **Two whole subsystems are empty stubs:** `conformal.py` and `data_pipeline.py` (0 bytes)
  — the conformal abstention layer (the headline novelty) and the video pipeline don't
  exist yet; Baseline B (TCN) is also absent.
- **Repo hygiene is poor:** committed debug probes, an empty `Login` file, PyCharm
  boilerplate `main.py`, an orphan `damkohler_regime_map.png` with no generator, a
  duplicated figure, unused dependencies, and the committed checkpoint isn't reproducible
  from current code. None of these break the science, but they obscure it.
