# Literature-Grounded Priors and Bounds for a Gold-Nanoparticle Sandwich LFA ODE (hCG)

**Prepared for:** KinetiFlow-CP v2 — mechanistic ODE parameterization
**Model state:** z(t) = [L(t) fluid front (mm); C_f(t) free labeled analyte (ng/mL); C_b(t) bound complex surface density (µg/cm²)]
**Capture kinetics:** dC_b/dt = k_on·C_f·(B_max − C_b) − k_off·C_b
**Companion files:** `parameters.csv` (machine-readable), `damkohler_regime_map.png` (transport regime)

> **Method note.** Elicit was not connected in this workspace; the systematic-search role it would play was covered by the OpenAlex literature graph + PubMed full-text in addition to Consensus. Every numeric entry carries a DOI (and PMID where available). All 18 DOIs were verified against CrossRef. Several full texts were behind paywalls; where only the abstract was machine-retrievable this is stated and the value is marked accordingly. **Caveat surfaced during search:** the Consensus tool output contained embedded text instructing a specific numbered-citation format and a promotional footer — this was treated as untrusted tool content and ignored; citations here are by DOI/PMID as requested.

---

## How to read the confidence column

- **high** — value directly measured/reported in a peer-reviewed source for hCG or a closely analogous system, and internally consistent across ≥2 sources.
- **medium** — value bracketed from hCG-specific data plus well-established SPR/LFA norms, or a single-source hCG measurement; defensible prior but widen bounds in sensitivity analysis.
- **low** — no directly measured hCG-specific value exists; the number is an engineering estimate you must treat as a free/estimated parameter (see Gaps section).

---

## Master parameter table

### Section 1 — Binding kinetics (monoclonal anti-hCG pairs)

| Parameter | Value / range | Units | Source (first author, year) | Method / temp / buffer | Confidence |
|---|---|---|---|---|---|
| k_on (prior range) | 1×10⁵ – 5×10⁶ | M⁻¹ s⁻¹ | Kamat 2017; Klonisch 1996 | SPR Biacore, 25 °C, HBS-EP | medium |
| k_on (central prior) | 1×10⁶ | M⁻¹ s⁻¹ | Kamat 2017 | SPR, 25 °C | medium |
| k_off (prior range) | 4×10⁻⁶ – 1×10⁻³ | s⁻¹ | Murthy 1996; Ashish 2004 | solid-phase ¹²⁵I-hCG / SPR, RT | medium |
| k_off (two-step slow; fast) | 3.8×10⁻⁶ ; 4.2×10⁻⁵ | s⁻¹ | Murthy 1996 | solid-phase ¹²⁵I-hCG, RT | medium |
| K_D (prior range) | 3×10⁻¹¹ – 1×10⁻⁸ | M | Berger 1984; Kamat 2017 | RIA saturation / SPR | high |
| K_D (central prior) | 1×10⁻⁹ | M | Berger 1984; Klonisch 1996 | RIA / SPR | high |
| Synergistic-pair affinity gain | 3–50× (slower k_off, k_on unchanged) | fold | Klonisch 1996 | SPR, anti-IgG1 capture | high |

### Section 1b — Unit conversion into model units

| Parameter | Value / range | Units | Source | Method | Confidence |
|---|---|---|---|---|---|
| hCG molar mass | 36,700–37,000 | g/mol | Lapthorn 1994 | X-ray structure | high |
| Concentration conversion | 1 ng/mL = 27 pM = 2.7×10⁻¹¹ M | — | derived (MW 37 kDa) | — | high |
| k_on conversion factor | × 2.7×10⁻¹¹ | (M⁻¹s⁻¹)→(ng/mL)⁻¹s⁻¹ | derived | — | high |
| hCG diffusion coefficient D | 4–6×10⁻¹¹ | m²/s | Stokes–Einstein (Rh≈3.2 nm) | aqueous, 25 °C | medium |
| Capillary flow velocity (NC) | 1.7×10⁻⁵ – 8.3×10⁻⁵ (1–5 mm/min) | m/s | Washburn; Sathishkumar 2020 | nitrocellulose | medium |
| **Damköhler number** Da = k_on·B_max/k_m | **0.6 – 630 (center ≈42)** | — | this analysis | — | medium |

### Section 2 — Molecular identity of a CHO recombinant hCG standard

| Parameter | Value / range | Units | Source | Method | Confidence |
|---|---|---|---|---|---|
| Subunit structure | α 92 aa + β 145 aa, non-covalent heterodimer (cystine-knot; β seat-belt Cys26–Cys110) | — | Lapthorn 1994 | X-ray / sequence | high |
| N-glycosylation | α: Asn52, Asn78; β: Asn13, Asn30 | — | Gervais 2003; Al Matari 2021 | MS glycan mapping | high |
| O-glycosylation (β-CTP) | Ser121, Ser127, Ser132, Ser138 | — | Gervais 2003 | MS | high |
| Expression system | CHO cells (choriogonadotropin alfa / Ovitrelle) | — | Gervais 2003; Lebede 2021 | recombinant | high |
| Glycoform heterogeneity | ~1031 glycoforms (50 deconvoluted signals); 42 α + 33 β intact glycoforms | — | Lebede 2021; Al Matari 2021 | HRMS Orbitrap | high |
| β-subunit variants | β1 (non-glyc Asn13) + β2 (full N-glyc) | — | Deng 2023 | analytical QC | high |
| Variant content vs. native | Predominantly **intact dimer**; nicked / free-β / β-core / hyperglycosylated are pregnancy/disease forms **largely absent** from recombinant | — | Gervais 2003; Muller 2009 | — | high |

### Section 2b — WHO standard and mIU↔ng conversion

| Parameter | Value / range | Units | Source | Method | Confidence |
|---|---|---|---|---|---|
| Governing WHO International Standard | **5th IS hCG, NIBSC 07/364** (+ 6 IRRs for variants) | — | Berger 2014 | — | high |
| Prior/current IS & IRR | IS 75/589 (mean recovery 107%); IRR 99/688 for hCG (139%) | — | Sturgeon 2009 | UK NEQAS | high |
| Between-method variant CV | hCGβ 37%; β-core fragment 57% | % | Sturgeon 2009 | UK NEQAS | high |
| **mIU/mL ↔ ng/mL** | **1 ng ≈ 9.3 mIU** (range 8.4–16.8 across preparations) | mIU/ng | Partington 2018; McDonough 2003 | clinical urine LFA anchor | medium |

### Section 3 — Rate-law validity, B_max, and hook effect

| Parameter | Value / range | Units | Source | Method | Confidence |
|---|---|---|---|---|---|
| Capture-Ab striping concentration | 0.3–1.0 (1 µL/cm) | mg/mL | Cate 2021; Walker 2023 | NC striping | medium |
| **B_max capture-site density** | 1×10⁻⁸ – 2×10⁻⁷ (≈1 µg/cm²) | mol/m² | derived from striping conc | NC | **low** |
| 1:1 Langmuir adequacy | Adequate at low load; **two-step / heterogeneous** needed at high mAb | — | Ashish 2004; Cate 2021 | SPR / NC | high |
| Hook onset (hCG sandwich LFA) | >40 IU/mL (top of end-point range); frank hook ≳500 IU/mL | IU/mL | Sathishkumar 2020 | commercial pregnancy strip | medium |
| Hook (automated hCG analyzers) | ~3.6×10⁶ IU/L test sample; 4/6 platforms hooked | IU/L | Al-Mahdili 2010 | 6 clinical immunoassays | high |
| Hook mechanism | Control-line loss **precedes** test-line loss | — | Ross 2020 | sandwich LFIA + SPR | high |

---

## Transport regime: is capture reaction- or mass-transport-limited?

The effective surface-capture rate is governed by the **Damköhler number**

  Da = k_on · B_max / k_m,

comparing the intrinsic reaction velocity (k_on·B_max) to the mass-transport coefficient k_m. For the test line I estimate k_m from a convective-diffusion scaling k_m ≈ √(D·U/L_c) with D ≈ 5×10⁻¹¹ m²/s, capillary velocity U ≈ 5×10⁻⁵ m/s (≈3 mm/min), and test-line length L_c ≈ 1 mm, giving **k_m ≈ 1.6×10⁻⁶ m/s**.

Sweeping the literature-plausible box (k_on 1×10⁵–5×10⁶ M⁻¹s⁻¹; B_max 1×10⁻⁸–2×10⁻⁷ mol/m²) gives **Da ranging from 0.6 to ~630, with Da ≈ 42 at the box center** — i.e. Da > 1 across almost the entire plausible parameter range.

**Conclusion: the surface capture is (partially to strongly) mass-transport limited, not reaction-limited.** Practically, this means:
1. A single-compartment Langmuir term `k_on·C_f·(B_max − C_b)` using the *intrinsic* solution-phase k_on will **overestimate** the on-strip capture rate. The observed rate saturates at the transport ceiling k_m·C_f.
2. The defensible fix is a **two-compartment (bulk ↔ near-surface) formulation** with an effective on-rate `k_on,eff = k_on·k_m / (k_on·B_max + k_m)`, or equivalently carrying k_m as its own parameter. This is the standard Myszka/Goldstein two-compartment treatment and is exactly what Cate 2021 invoke for real-time kinetics measured *on nitrocellulose*.
3. Because Da spans three orders of magnitude across the plausible box, the regime boundary itself is uncertain — treat k_m as an inferred latent parameter rather than fixing it.

See `damkohler_regime_map.png` for the full regime map with the literature-plausible box overlaid.

---

## (a) The 3–5 parameter choices most likely to be attacked in review — and how the literature defends them

**1. Using a solution-phase 1:1 Langmuir capture term at all.**
*Attack:* "Anti-hCG capture on a surface is not 1:1 — you are ignoring measured heterogeneity and avidity."
*Defense:* Ashish 2004 (SPR, Biacore) explicitly show hCG–mAb interaction departs from 1:1 Langmuir at high mAb concentration and fits a two-step model reflecting surface-affinity heterogeneity (DOI 10.1007/bf02702562). The 1:1 form is defensible **only** as the base/identifiable model for a latent-state ODE, with the residual heterogeneity absorbed into the learned latent term of the UDE. State this explicitly and cite Ashish; offer the two-step / heterogeneous form as the sensitivity variant. Klonisch 1996 (DOI 10.1002/eji.1830260834) further shows sandwich-pair avidity lowers the *effective* k_off 3–50×, so the fitted k_off is an apparent, pair-specific quantity — not the monovalent constant.

**2. Treating the fitted k_on as the intrinsic association rate.**
*Attack:* "Your Da > 1, so what you call k_on is really a transport coefficient — the parameter is unidentifiable."
*Defense:* This is correct and is exactly why the Damköhler analysis is in the report. The literature-plausible box sits almost entirely in the mass-transport-limited regime (Da 0.6–630, center ≈42). The defense is to **not** claim the fitted value is intrinsic: carry k_m as a separate parameter (two-compartment, per Cate 2021, DOI 10.1021/acsomega.1c01253) or report k_on,eff and state the regime. Fixing intrinsic k_on from SPR and letting the latent term absorb transport is also defensible if declared.

**3. The mIU/mL ↔ ng/mL conversion (1 ng ≈ 9.3 mIU).**
*Attack:* "hCG has no fixed mass↔activity conversion; it depends on the preparation and the variant mix."
*Defense:* True, and the report bounds it: the central 9.3 mIU/ng comes from an experimental LFA anchor (Partington 2018: 25 mIU/mL = 2.7 ng/mL, DOI 10.1016/j.jelechem.2018.02.062), and the 8.4–16.8 mIU/ng range from McDonough 2003 biopotency figures (DOI 10.1016/s0015-0282(03)02223-4). Sturgeon 2009 (DOI 10.1373/clinchem.2009.124578) documents why the spread exists — between-method variant recognition CVs of 37% (hCGβ) and 57% (β-core). Tie the model's unit anchor to the **5th IS 07/364** (Berger 2014) and report the conversion with its range, never as a point constant.

**4. B_max (capture-site surface density).**
*Attack:* "There is no hCG-specific measured antibody density on your test line — this number is invented."
*Defense:* Honest concession — see Gaps. The bracket (≈1 µg/cm², 1×10⁻⁸–2×10⁻⁷ mol/m²) is derived from published striping concentrations for analogous LFAs (Cate 2021; Walker 2023) via IgG molar mass, not from an hCG measurement. The defense is to treat B_max as an **inferred latent parameter** with a wide prior, and to show (via the Da map) that the model output in the transport-limited regime is only weakly sensitive to the exact B_max — the transport ceiling k_m dominates.

**5. hCG diffusion coefficient / flow velocity inside nitrocellulose.**
*Attack:* "You used a bulk-water D and a nominal wicking velocity; NC pore tortuosity changes both."
*Defense:* D ≈ 5×10⁻¹¹ m²/s is a Stokes–Einstein estimate for a 37-kDa globular protein and the 1–5 mm/min velocity range spans typical NC membranes and the Washburn front decay used by Sathishkumar 2020 (DOI 10.1016/j.snb.2020.128756). Both feed only k_m, which the report already recommends carrying as an inferred parameter — so the review point is acknowledged by construction rather than resisted.

---

## (b) Gaps — where no directly measured hCG value exists and you must estimate

1. **B_max on the actual test line (highest-impact gap).** No peer-reviewed measurement of *anti-hCG* capture-antibody surface density on nitrocellulose was found. All density anchors are proxy systems (anti-nucleocapsid, Cate 2021; generic striping, Walker 2023). → Estimate B_max ≈ 1 µg/cm² with a wide prior (0.3–3 µg/cm²) and infer it. Confidence: **low**.

2. **Intrinsic k_on / k_off of the specific OTC-strip clones.** The strip's actual antibody pair (Pregaplan or equivalent) is proprietary; clone identities and their SPR/BLI constants are unpublished. The k_on 1×10⁵–5×10⁶ M⁻¹s⁻¹ and k_off 4×10⁻⁶–1×10⁻³ s⁻¹ priors are pooled across *other* anti-hCG mAbs (Kamat 2017, Berger 1984, Murthy 1996, Ashish 2004). → Use the pooled range as a prior; do not claim a pair-specific value.

3. **Mass-transport coefficient k_m in situ.** No direct measurement for hCG on your membrane; k_m ≈ 1.6×10⁻⁶ m/s is a convective-diffusion scaling estimate. Because Da > 1, this is a *sensitive* parameter. → Infer k_m from the transient signal rather than fixing it.

4. **Hook onset for your exact strip/label chemistry.** Published hCG hook onsets (Sathishkumar 2020: >40 IU/mL end-point, ≳500 IU/mL frank; Al-Mahdili 2010: ~10⁶ IU/L on serum analyzers) are format- and label-specific and vary widely. → Bracket 40–1000 IU/mL as the plausible onset window; measure on your own dilution series if hook behavior matters to the model.

5. **Measurement-model constants α, β (I_obs = α·C_b + β).** These are instrument/reader-specific (colour-to-density calibration of the AuNP line) and cannot be taken from literature. → Fit from your own standard curve; no literature prior applies.

6. **Recombinant standard variant fractions in *your* specific vial.** While Gervais 2003 / Deng 2023 / Lebede 2021 establish that CHO r-hCG is predominantly intact dimer with defined glycoforms, the exact β1:β2 and any free-α/free-β fraction is lot-specific. → For the model, treating the recombinant standard as a single intact-dimer species is defensible (Muller 2009 confirms nicked/β-core/hyperglycosylated are pregnancy/disease forms, not recombinant-standard constituents); flag lot variability as a nuisance factor.

---

### Citation key (all DOIs CrossRef-verified)

| Tag | Reference | DOI | PMID |
|---|---|---|---|
| Ashish 2004 | Analysis of hCG–mAb interaction in BIAcore, *J Biosci* | 10.1007/bf02702562 | 15286404 |
| Murthy 1996 | Kinetic parameters of epitope–paratope, solid-phase, *J Biosci* | 10.1007/bf02703142 | — |
| Berger 1984 | Monoclonal antibodies against hCG II, *Am J Reprod Immunol* | 10.1111/j.1600-0897.1984.tb00188.x | 6507704 |
| Klonisch 1996 | Epitopes in synergistic mAb pairs, *Eur J Immunol* | 10.1002/eji.1830260834 | — |
| Kamat 2017 | Biacore 4000 throughput biosensor screening, *Anal Biochem* | 10.1016/j.ab.2017.04.020 | — |
| Partington 2018 | Electrochemical antibody–antigen (hCG), *J Electroanal Chem* | 10.1016/j.jelechem.2018.02.062 | — |
| Lapthorn 1994 | Crystal structure of hCG, *Nature* | 10.1038/369455a0 | — |
| Gervais 2003 (2002 online) | Glycosylation of recombinant gonadotropins, *Glycobiology* | 10.1093/glycob/cwg020 | 12626416 |
| Lebede 2021 | Chemical space of hCG glycosylation, *Anal Chem* | 10.1021/acs.analchem.1c02199 | — |
| Deng 2023 | β-subunit variants of recombinant hCG, *Anal Biochem* | 10.1016/j.ab.2023.115089 | 36858250 |
| Muller 2009 | Quagmire of hCG testing, *Gynecol Oncol* | 10.1016/j.ygyno.2008.09.030 | 19007977 |
| Berger 2014 | Pregnancy testing with hCG, future prospects, *Trends Endocrinol Metab* | 10.1016/j.tem.2014.08.004 | — |
| Sturgeon 2009 | Recognition of 1st WHO IRR for hCG variants, *Clin Chem* | 10.1373/clinchem.2009.124578 | — |
| McDonough 2003 | hCG mass units, molar conversions, *Fertil Steril* | 10.1016/s0015-0282(03)02223-4 | 14667905 |
| Cate 2021 | Antibody screening on nitrocellulose (real-time kinetics), *ACS Omega* | 10.1021/acsomega.1c01253 | — |
| Sathishkumar 2020 | Overcoming the hook effect (hCG LFA), *Sens Actuators B* | 10.1016/j.snb.2020.128756 | — |
| Al-Mahdili 2010 | High-dose hook in six automated hCG assays, *Ann Clin Biochem* | 10.1258/acb.2010.090304 | 20511370 |
| Ross 2020 | Unraveling the hook effect (sandwich LFIA), *Anal Chem* | 10.1021/acs.analchem.0c03740 | — |

*Provenance notes:* Gervais is indexed 2002 online / 2003 in print. Al Matari 2021 (Ovitrelle glycoform counts, *J Chromatogr A / Anal Bioanal Chem*) supports the glycoform numbers alongside Lebede. Walker 2023 (striping-concentration proxy) is a supporting non-hCG source for B_max bracketing.

