

================================================================================
PROJECT OUTLINE 
================================================================================


1. RESEARCH QUESTION
────────────────────
Can a latent-state gray-box Universal Differential Equation — whose internal
state variables explicitly represent the physical quantities governing lateral
flow assay development (fluid front position L(t), free labeled analyte
concentration C_f(t), and test-line bound complex surface density C_b(t)) —
accurately forecast the 15-minute thermodynamic equilibrium optical intensity
of a commercial colorimetric LFA from only the initial 60 seconds of transient
kinetic video data?

Furthermore, can a split conformal prediction layer, augmented with Wasserstein
regularization for robustness under bounded environmental covariate shifts
(temperature ±8°C, relative humidity ±20%), autonomously implement a clinical
abstention protocol that safely rejects high-uncertainty predictions —
demonstrating superior reliability to a black-box discrete-time 1D Temporal
Convolutional Network baseline?


2. PROBLEM STATEMENT
─────────────────────
Point-of-care lateral flow assays are the primary diagnostic tool of
decentralized medicine globally. Despite their ubiquity, three persistent
limitations remain unsolved:

Limitation 1 — Mandatory Incubation Delay
  LFAs require 15 to 20 minutes before a reliable equilibrium readout. Reading
  early causes false negatives. Reading too late causes false positives via the
  hook effect, where excess analyte saturates all binding sites and prevents
  sandwich complex formation.

Limitation 2 — Black-Box Model Brittleness
  Data-driven architectures (CNN-LSTMs, TCNs) can forecast equilibrium from
  early kinetic data but have no internal representation of the continuous fluid
  dynamics and binding kinetics governing assay development. Under environmental
  perturbations — temperature shifts that alter reaction rates, humidity changes
  that modify capillary viscosity — they generate confidently incorrect
  predictions with no mechanism to flag uncertainty.

Limitation 3 — Intensity-Space Physics Conflation (new in v2)
  Existing physics-informed approaches write differential equations directly on
  observed optical intensity signals. This conflates the camera's measurement
  model with the underlying biochemical kinetics, preventing true physical
  parameter identifiability and creating models that cannot separate chemical
  dynamics from illumination artifacts.

KinetiFlow-CP v2 targets all three limitations by formulating the ODE system
on physical latent state variables — not image intensity — and wrapping the
resulting forecast in a formally guaranteed conformal abstention layer.


3. LITERATURE REVIEW
─────────────────────

3.1 Foundational Scientific Machine Learning
  Chen et al. (2018) introduced Neural Ordinary Differential Equations,
  establishing continuous-depth solver-based models for hidden dynamic states.
  Rackauckas et al. (2020) formalized Universal Differential Equations,
  combining known mechanistic ODEs with learnable neural residuals for
  physically interpretable, data-efficient modeling.

3.2 Physics-Informed Reaction-Diffusion Modeling
  Saita and Hori (2025, arXiv:2512.19416) demonstrated Neural ODEs for
  temperature-dependent chemical kinetics in droplet microreactors, but
  confined their work to well-mixed liquid-phase systems neglecting spatial
  transport.

  Cai, Zhou, and Ren (2026, arXiv:2603.28478) proved that neglecting diffusion
  coupling in reaction-diffusion systems completely distorts learned kinetic
  parameters. For lateral flow assays, where analyte must physically migrate
  through porous nitrocellulose via capillary advection before binding at the
  test line, this constitutes a mathematical mandate for explicit transport
  coupling.

  SPIN-ODE and Stiff-PINN established that stiff chemical kinetics require
  implicit integration strategies to prevent solver failure and gradient
  explosion during backpropagation.

3.3 State-Space Modeling of Lateral Flow Assays — The Direct Gap
  Qian and Bau (2003, Analytical Biochemistry, 322:89–98) established the
  canonical reaction-diffusion framework for sandwich LFA kinetics, coupling
  Darcy-flow capillary advection with reversible Langmuir binding at the test
  line. This remains the standard mechanistic basis for LFA computational
  modeling.

  Zeng, Wang, and Li et al. (2011, IEEE Trans Biomed Eng, 58:1959–1966; 2012,
  IEEE Trans Nanotechnol, 11:321–327) proved that LFA kinetics can be modeled
  as a state-space system and identified from transient optical time-series via
  Extended Kalman Filtering and particle filter approaches.

  These are the most important prior art papers for this project. They proved
  the feasibility of LFA dynamic parameter estimation from limited data.
  However, the Zeng et al. approaches operate in discrete time, use empirical
  (non-physics-constrained) state transition functions, and provide no
  uncertainty quantification for distributional shifts. KinetiFlow-CP v2
  advances beyond this prior art on all three dimensions simultaneously.

3.4 Deep Learning for Temporal Point-of-Care Diagnostics
  Lee et al. (2024, Nature Communications, Vol. 15, Article 1695,
  DOI: 10.1038/s41467-024-46069-2) published the TIMESAVER framework,
  achieving 1–2 minute LFA readouts via YOLO + CNN-LSTM, outperforming human
  experts at the full 15-minute readout. This is the state-of-the-art
  black-box benchmark.

  Han et al. (2024) demonstrated deep learning for paper-based vertical flow
  assays with high-sensitivity cardiac troponin detection. Both works share the
  limitation of operating as physics-blind black boxes vulnerable to
  environmental shift.

3.5 Uncertainty Quantification for Point-of-Care Systems
  Goncharov et al. (2025, ACS Nano, arXiv:2512.21335) demonstrated Monte Carlo
  Dropout uncertainty rejection for a computational POC sensor, improving
  diagnostic sensitivity from 88.2% to 95.7%. However, MCDO is a Bayesian
  heuristic — it provides no formal statistical coverage guarantees.

  Xu et al. (2025, ICLR, arXiv:2501.13430) introduced Wasserstein-Regularized
  Conformal Prediction (WR-CP), addressing general distribution shifts by
  minimizing the Wasserstein distance between calibration and test
  non-conformity score distributions via importance weighting and regularized
  representation learning.

  Manokhin and Grønhaug (January 2026, arXiv:2601.19944) benchmarked post-hoc
  calibration methods across 21 classifiers and 30 tasks, demonstrating Venn-
  Abers predictors outperform Platt scaling for probabilistic calibration.


4. NOVELTY CLAIM
──────────────────
Three distinct original contributions are claimed. All three were confirmed
unprecedented by deep research search across literature through June 2026.

Contribution 1 — First continuous-time latent-state model of LFA kinetics
  Prior LFA state-space models (Zeng et al. 2011–2012) use discrete-time
  Kalman filter structures with empirical transition functions. KinetiFlow-CP v2
  is the first to embed the continuous-time Qian-Bau reaction-diffusion
  framework into a differentiable Neural ODE, solved with adjoint-method
  backpropagation, operating on physically interpretable latent states and
  enabling true kinetic parameter identifiability.

Contribution 2 — First distribution-free abstention system for POC diagnostics
  Goncharov et al. demonstrated MCDO uncertainty rejection without formal
  coverage guarantees. KinetiFlow-CP v2 is the first to apply frequentist
  conformal prediction with mathematically rigorous finite-sample coverage
  bounds to a POC diagnostic system, triggering a clinically actionable
  "ABSTAIN/RETEST" protocol under formally characterized distributional shift.

Contribution 3 — First integrated physics-constrained + conformal-safe LFA
reader
  No prior paper combines: transport-coupled continuous-time Neural ODE on
  latent states + distribution-free conformal abstention + paper-based capillary
  microfluidic LFAs, with the dual objective of early readout acceleration and
  environmental robustness.


5. METHODOLOGY
───────────────

5.1 Hardware (Locked)
  Camera: Raspberry Pi HQ Camera Module (IMX477 sensor), manual exposure lock
  (ISO 100, shutter 1/500s, AWB permanently disabled, raw Bayer output).
  Smartphone cameras are excluded — their ISPs apply proprietary auto-exposure
  and noise-reduction that introduce nonlinear optical artifacts the ODE solver
  misreads as chemical kinetic events.

  Imaging rig: Fully enclosed lightproof enclosure with matte black interior,
  constant-current LED panel (6500K, PWM-free, current-regulated), and a
  Munsell N5 matte grey reference calibration patch visible in all frames.
  Recordings with >3 DN illumination drift are automatically rejected.

  Signal processing: OpenCV pipeline extracts test-line ROI. Savitzky-Golay
  filter (window = 15, poly-order = 3) removes high-frequency noise while
  preserving kinetic inflection points. Frame timestamps logged at millisecond
  resolution.

5.2 Latent-State Gray-Box ODE System (The Core Architecture Change)
  The ODE operates on physical latent states, not on image intensity.
  This is the central architectural change from v1.

  State vector:
    z(t) = [ L(t),  C_f(t),  C_b(t) ]

    L(t)   = fluid front position along the nitrocellulose membrane [mm]
    C_f(t) = free labeled analyte concentration at the test-line zone [ng/mL]
    C_b(t) = bound antibody-antigen complex surface density at test line [μg/cm²]

  Known physics component (from Lucas-Washburn 1921 and Qian & Bau 2003):
    dL/dt   = sqrt( γ_r / 8ηL(t) )                              [Washburn transport]
    dC_f/dt = -(dL/dt / L)·C_f - k_on·C_f·(B_max - C_b) + k_off·C_b  [Langmuir]
    dC_b/dt =  k_on·C_f·(B_max - C_b) - k_off·C_b              [capture kinetics]

    γ_r    = Washburn permeability constant (surface tension γ, pore radius r,
              viscosity η)
    k_on, k_off = association and dissociation rate constants, initialized from
                  Qian & Bau (2003) literature priors
    B_max  = total capture antibody site surface density at test line

  Neural residual (3-layer MLP, 64 hidden units, input dim = 5):
    f_θ(C_f, C_b, t, T_ambient, RH) → [δC_f, δC_b]

    Learns: membrane tortuosity deviations, non-ideal steric blocking,
    temperature-dependent viscosity corrections beyond Washburn approximation,
    humidity-driven surface tension shifts.

  Full Universal Differential Equation:
    dz/dt = KnownPhysics(z, t) + [0, f_θ1(z, t, T, RH), f_θ2(z, t, T, RH)]

  Measurement equation (separates physics from camera observation — new in v2):
    I_obs(t) = α · C_b(t) + β + ε(t)

    α, β are learned optical calibration parameters. ε ~ N(0, σ²) is
    observation noise. The neural residual NEVER touches I_obs. It corrects
    the kinetic equations only. This is what makes the physics real, not
    cosmetic. Beer-Lambert linearity is empirically verified via a calibration
    curve in Phase 1. If nonlinearity is detected at high concentrations, the
    measurement model is upgraded to a sigmoidal form: I = α·tanh(β·C_b + γ).

  Loss function (incorporating Gemini Rec. 2 — physical bounds enforcement):
    L_total = L_obs + λ · L_physics

    L_obs    = (1/T) Σ_t [I_obs(t) - (α·C_b(t) + β)]²         [observation fit]
    L_physics = ReLU(-C_f) + ReLU(-C_b) + ReLU(C_b - B_max)   [physical bounds]

    The physics penalty prevents the model from predicting negative
    concentrations or bound complex exceeding total available sites.

5.3 Software Stack (Locked)
  Python 3.11 + PyTorch 2.x + torchdiffeq.
  Julia/SciML is excluded — the learning curve is incompatible with the
  September 2026 deadline for a student with Python proficiency.

  ODE solver hierarchy:
    Primary:  torchdiffeq implicit_adams (adaptive step-size, handles moderate
              stiffness)
    Fallback: scipy.integrate.solve_ivp(method='Radau') for strips with
              unusually stiff kinetics identified in Phase 1 pilot
    Backprop: continuous adjoint sensitivity method (memory-efficient for long
              integration windows)
    Tolerances: atol=1e-4, rtol=1e-3 (tightened to atol=1e-6 during pilot)


6. CONFORMAL PREDICTION IMPLEMENTATION
────────────────────────────────────────

6.1 Stage 1 — Split Conformal Prediction (Primary Method)
  Using the calibration set (20% of data, grouped by experimental day):
    Non-conformity score: α_i = |I_true,i − Î_pred,i|
    Prediction interval at 1-δ confidence:
      C_hat(x_test) = [Î_pred ± q_hat(δ)]
      q_hat(δ) = Quantile_{(1-δ)(1 + 1/n_cal)}(α_1, ..., α_n_cal)

  SCP provides the primary finite-sample coverage guarantee under the
  exchangeability assumption. It is the main reported method.

6.2 Stage 2 — WR-CP Extension (OOD Robustness Benchmark)
  On the perturbed OOD test set, WR-CP minimizes Wasserstein distance between
  calibration and test non-conformity score distributions via importance
  weighting. Benchmarked against SCP to quantify coverage improvement under
  physical shift.

  Note: If n_cal ≈ 80 samples produce unstable importance weight estimates,
  SCP is retained as the primary method and WR-CP is reported as exploratory.
  This does not undermine the core novelty claim.

6.3 Abstention Protocol
  System outputs "ABSTAIN / RETEST STRIP" if either condition holds:
    (a) Conformal interval straddles the clinical decision threshold θ_clinical
        (locked on calibration data before any test evaluation)
    (b) Interval width |C_hat| exceeds the clinical tolerance threshold w_max

  All thresholds are locked on calibration data only. No test set leakage
  is possible by construction.

  Important: The coverage guarantee is maintained under bounded Wasserstein
  shift — not under arbitrary OOD perturbation. This must be stated precisely
  when presenting to judges (correction from v1 overclaim).


7. DATA COLLECTION PLAN
────────────────────────

Analyte (Locked): Commercial OTC hCG/LH strips (Pregaplan or equivalent,
  available at Indian pharmacies) + synthetic recombinant hCG standard from
  a chemical supplier.
  No human biological fluids. No specialized clinical reagents.
  cTnI is excluded — procurement complexity and cost are not feasible for a
  school-level project within the timeline.

Dataset target: 400–500 total recordings

  Concentration levels: 5 levels (0×, 0.5×, 1×, 2×, 5× detection threshold)
  Clean set: ~40–50 recordings per concentration level (200–250 total)
             at 25°C ±1°C, 40–50% RH
  OOD perturbed set: ~80 recordings
             (~27 at 38°C, ~27 at 28°C, ~26 at ≥75% RH)
             Never used during training or calibration.
  Minimum 5 separate experimental days, multiple strip lot numbers.

  Data splits: 60% Train / 20% Calibration / 20% Test (clean) +
               separate OOD perturbed test set
  Splits grouped strictly by experimental day AND strip lot number to
  prevent temporal and batch leakage.

SRC Compliance:
  Form 1 (Student Checklist) — to be completed before any experimental work
  Form 2 (Qualified Scientist) — mentor signature required
  Form 6 (Hazardous Substances) — for synthetic hCG standard and buffer
  No BSL-2 handling required.


8. THREE-WAY BASELINE DESIGN
──────────────────────────────
All three models trained on identical data splits with matched parameter budgets.
This is what makes the novelty claim credible — ablations causally attribute
improvements to specific components.

Baseline A — Pure Physics ODE
  The Qian-Bau latent-state ODE system with known physics only and no neural
  residual (f_θ = 0). Tests the performance ceiling of hand-coded mechanistic
  physics alone.

Baseline B — 1D Temporal Convolutional Network (TCN)
  1D-TCN (3 residual blocks, dilated causal convolutions, parameter count
  matched to the MLP residual) trained end-to-end on raw I_obs(t) sequences.
  Explicitly labeled as "1D TCN" — not a reproduction of TIMESAVER, which used
  a different analyte, hardware, and architecture. Establishes the black-box
  discrete-time performance ceiling on the same dataset.

KinetiFlow-CP — Full Gray-Box Neural ODE + Conformal Abstention
  Complete framework as described in Sections 5 and 6.


9. EVALUATION METRICS
───────────────────────
Pre-specified before any model training begins. All reported on held-out test
sets after model selection on the validation split only.

  RMSE / MAE              — Forecast accuracy on 15-minute equilibrium intensity
                            (all three models)
  Coverage                — Fraction of test samples where true value falls in
                            conformal interval (target ≥ 90%) [KinetiFlow-CP]
  Interval Width          — Mean conformal interval width; narrower is better
                            at equal coverage [KinetiFlow-CP]
  OOD Coverage Drop       — Coverage difference between clean and perturbed
                            test sets; quantifies distributional robustness
                            [SCP vs WR-CP comparison]
  Abstention Rate         — Fraction of strips triggering ABSTAIN protocol
  Selective Accuracy      — Diagnostic accuracy on non-abstained strips only;
                            expected to far exceed full-set accuracy
  Decision Time t*        — Earliest time at which conformal interval collapses
                            below w_max; proxy for early-readout capability


10. EXECUTION TIMELINE
───────────────────────

Phase 1 — Setup and Identifiability Pilot (Weeks 1–3)
  Week 1:   Complete SRC Forms 1 and 2. Secure mentor (highest priority — see
            Section 12). Procure hardware (RPi HQ Camera, LED panel, enclosure
            materials) and reagents (600+ strips minimum, recombinant hCG
            standard, buffer solutions).
  Week 2:   Build imaging rig. Implement OpenCV ROI extraction pipeline and
            Savitzky-Golay filter. Implement latent-state Neural ODE in
            torchdiffeq and verify solver stability on synthetic data before
            touching real strips.
  Week 3:   20-strip identifiability pilot — record 5 concentration levels × 4
            strips. Verify that early traces (0–60s) diverge uniquely by
            concentration. This is a necessary precondition for the forecasting
            task; do not proceed to full data collection if traces overlap.
            Measure empirical calibration curve for Beer-Lambert verification.

Phase 2 — Full Data Collection (Weeks 4–7)
  Execute 400–500 strip recordings under clean and perturbed conditions.
  Log all metadata: strip lot number, date, temperature, humidity, camera
  calibration values.

Phase 3 — Training, Ablation, Validation (Weeks 8–12)
  Train all three models on grouped splits. Calibrate SCP layer. Benchmark
  WR-CP vs SCP on OOD test set. Run full ablation (physics-only vs
  residual-only vs gray-box). Compute all Section 9 metrics on locked test sets.

Phase 4 — Write-Up (Weeks 12–14)
  Research paper, poster, judge-facing presentation.
  Publish code and dataset publicly — this alone signals maturity to judges
  and is something recent ISEF winners in adjacent categories have done.


11. RISK REGISTER
──────────────────

Risk 1 — ODE Identifiability Failure
  The 3-state system [L, C_f, C_b] may not be uniquely inferrable from a
  single-output observation I_obs(t) = αC_b + β (multiple parameter
  combinations may produce identical output trajectories).
  Mitigation: Initialize k_on, k_off from Qian-Bau priors. Run local
  identifiability analysis (sensitivity matrix rank check) on pilot data.
  If identifiability fails, fix k_on and k_off as constants and learn only
  the neural residual.

Risk 2 — Beer-Lambert Nonlinearity
  Linearity I_obs = αC_b + β may not hold at 5× concentration levels where
  gold nanoparticle stacking causes nonlinear absorbance.
  Mitigation: Calibration curve in Phase 1 pilot. Upgrade to sigmoidal model
  if nonlinearity detected. This does not require re-architecting the system.

Risk 3 — Mathematical Stiffness
  High k_on·C_f·B_max terms at elevated concentrations can create stiff ODE
  dynamics causing implicit_adams to take prohibitively small step sizes.
  Mitigation: Radau fallback (scipy). Stiffness pre-screened in identifiability
  pilot at all five concentration levels.

Risk 4 — WR-CP Instability at n_cal ≈ 80
  Insufficient calibration samples for stable Wasserstein importance weight
  estimation. This is the most likely technical failure point.
  Mitigation: If WR-CP is unstable, report SCP as the primary method and WR-CP
  as an exploratory extension. The core novelty (latent-state ODE + conformal
  abstention) is intact with SCP alone.

Risk 5 — No Mentor Secured
  ISEF SRC Form 2 requires a Qualified Scientist signature before any
  experimental work begins. Failing to secure a mentor by end of Week 1
  legally prevents the project from starting.
  Mitigation: Email IIT faculty in the first week with this outline document.


12. MENTOR PROFILE AND CONTACT STRATEGY
─────────────────────────────────────────
Securing a mentor in Week 1 is the single highest-priority task.

Required expertise (any one of these qualifies):
  - Neural ODEs / Scientific Machine Learning
  - Conformal prediction or statistical uncertainty quantification
  - Point-of-care diagnostics or microfluidics

Primary targets:
  IIT Bombay, Delhi, Jodhpur, or Madras — departments with active SciML,
  dynamical systems, or uncertainty quantification research groups.
  (CS, Applied Mathematics, or Biomedical Engineering departments.)

Contact strategy:
  Email the faculty member directly. Attach this outline document (2-3 pages
  of the core sections). Request a 20-minute video call. Explicitly mention
  the ISEF Form 2 requirement in the email — many IIT faculty have signed
  Form 2 for school students before and understand the process.

Fallback:
  PhD researcher or postdoctoral fellow at TIFR, IISc, IIIT, or any NIT
  who meets ISEF qualified scientist requirements.

Questions to discuss with mentor on first call:
  (a) Is the 3-state latent system identifiable from single-output I_obs?
      Should k_on, k_off be fixed from Qian-Bau priors or jointly learned?
  (b) For OTC hCG strips at the planned concentration levels, how stiff are
      the binding kinetics? Is implicit_adams sufficient?
  (c) Is Beer-Lambert linearity valid for gold-nanoparticle hCG strips across
      all five concentrations? At what concentration does it break down?
  (d) With ~80 calibration samples, is WR-CP importance weight estimation
      likely to be stable? What minimum n_cal would you recommend?
  (e) Does synthetic recombinant hCG standard require Form 6 in addition to
      Forms 1 and 2?


13. WHAT THIS PROJECT STILL IS NOT
────────────────────────────────────
This document is a proposal. As of the date of writing, no experimental data
exists. IRIS judges evaluate completed original research. The outline, however
strong technically, only becomes competitive once:

  (a) The identifiability pilot in Phase 1 succeeds
  (b) Real experimental data is collected across all five concentration levels
  (c) All three models are trained and the ablation is complete
  (d) Metrics in Section 9 are computed and reported honestly
  (e) A mentor has reviewed and signed off on the methodology

Current realistic win probability at IRIS 1st Prize (my assessment):
  As proposal: ~3%
  After full execution with all changes above: ~40–45%

The gap between those two numbers is filled entirely by execution — running the
experiment, getting results, and being able to defend every technical decision
in front of a PhD-level judge. The concept is solid. The execution is what
determines whether this wins or not.


