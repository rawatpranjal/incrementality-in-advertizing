# Holdout (RCT) Analysis

## Overview

Randomized controlled trial analysis using a holdout design where a small percentage of users are withheld from ad treatment. This provides an experimental benchmark for causal incrementality measurement, complementing the observational econometric approaches used elsewhere.

## Data Requirements

- **Unit of analysis:** User-level
- **Design:** 97% treatment (see ads) / 3% control (holdout, no ads)
- **Input tables:** AUCTIONS_USERS (for assignment), PURCHASES (for outcomes), CLICKS/IMPRESSIONS (for treatment verification)
- **Sample/filters:** Users with valid assignment status; analysis period aligned with holdout experiment

## Pipeline

1. `01_simple_holdout_analysis.ipynb` — Basic ITT estimation and ARPU comparison
2. `01_simple_holdout_analysis_fixed.ipynb` — Corrected version with fixes
3. `02_focused_itt_analysis.ipynb` — Focused intent-to-treat analysis with robustness checks

## Model Specification

**Intent-to-Treat (ITT):**
```
ITT = E[Y | Z=1] - E[Y | Z=0]
```
where Z is the random assignment indicator (1=treatment, 0=control)

**ARPU Comparison:**
```
ARPU_treatment = Σ Revenue_i / N_treatment
ARPU_control = Σ Revenue_i / N_control
Lift = (ARPU_treatment - ARPU_control) / ARPU_control
```

**Statistical Test:**
```
H_0: μ_treatment = μ_control
H_1: μ_treatment ≠ μ_control
```
Two-sample t-test or proportion test depending on outcome metric

**Variables:**
- Y: Purchase indicator, purchase count, or revenue
- Z: Random assignment to treatment (sees ads) or control (holdout)

**Interpretation:**
- ITT captures the causal effect of being assigned to see ads (inclusive of non-compliance)
- With perfect compliance, ITT = ATE (average treatment effect)
- Small control group (3%) limits precision but provides unbiased estimate

## Key Files

| File | Purpose |
|------|---------|
| `01_simple_holdout_analysis.ipynb` | Initial ITT and ARPU analysis |
| `01_simple_holdout_analysis_fixed.ipynb` | Corrected analysis |
| `02_focused_itt_analysis.ipynb` | Detailed ITT with diagnostics |

## Outputs

- ITT point estimates with confidence intervals
- ARPU by treatment arm
- Percentage lift calculations
- Statistical significance tests
- Balance checks on pre-treatment covariates

## Connections

- Provides experimental benchmark for `panel/` observational estimates
- Validates assumptions underlying `regression-discontinuity/` and other quasi-experimental methods
- Ground truth for `deep-learning/` causal inference comparisons
