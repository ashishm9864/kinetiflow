# KinetiFlow-CP v2

## What this is
A scientific-ML project forecasting the 15-minute equilibrium optical intensity
of a commercial hCG lateral-flow assay (pregnancy strip) from the first ~60 s of
kinetic video, using a latent-state gray-box Universal Differential Equation
(UDE) wrapped in a conformal-prediction abstention layer. Target venue: IRIS
national science fair. Author is a high-school student (intermediate Python).

## Build & run
- Python 3.11, PyTorch 2.x + torchdiffeq (continuous adjoint solver).
- Install: `pip install -r requirements.txt`
- Run a module's self-test: `python src/<module>.py`

## Architecture (state z = [L, C_f, C_b])
- L = fluid-front position [mm]; C_f = free analyte [ng/mL]; C_b = bound complex.
- Known physics: Lucas-Washburn front (dL/dt = k_wash/L, the 1/L form — NOT sqrt)
  + Qian-Bau Langmuir binding. A neural residual corrects ONLY the chemistry
  channels (C_f, C_b), never L, never the optics.
- Optics are a SEPARATE model: I_obs = alpha*C_b + beta (kept out of the ODE).

## Non-negotiable scientific constraints (a wrong one loses the project)
- Integrate with torchdiffeq odeint_adjoint (continuous adjoint).
- Initial front position L0 must be > 0 (~5 mm). L0 -> 0 makes 1/L blow up and
  the adjoint underflows. t = 0 means "front arrives at the read window".
- IDENTIFIABILITY (proven in identifiability.py): from the single output I_obs,
  alpha and B_max are perfectly confounded, and k_on/k_off/k_m are not separately
  recoverable. Therefore the trained model FREEZES k_on, k_off, B_max at the
  values in priors.py and learns only {alpha, beta, the neural residual}. Do not
  make the frozen kinetics learnable.
- Priors live in priors.py — never hard-code magic numbers; import from there.
- Splits must be grouped by experimental DAY and strip LOT to prevent leakage.

## Files already built and verified (do not rewrite unless asked)
src/mechanistic_ode.py, src/loss_functions.py, src/priors.py,
src/identifiability.py

## Style
- Type hints + short docstrings on public functions.
- Prefer small, testable functions. Every new module gets a `__main__` self-test.
- Do not use browser localStorage or any hidden state.