# Appendices: Causality Reference Materials

## Overview

Supporting theoretical documentation and reference materials on causal inference methods. Contains derivations, sensitivity analysis frameworks, and decomposition techniques that underpin the empirical analyses in other folders.

## Contents

### Sensitivity Analysis
- `sensitivity_analysis.tex` — LaTeX document on sensitivity analysis for unmeasured confounding
- `sensitivity_analysis.pdf` — Compiled PDF
- `compute_sensitivity_examples.py` — Numerical examples for sensitivity bounds
- `verify_examples.py` — Verification of sensitivity calculations
- `sensitivity_examples.txt` — Example outputs

### Effect Decomposition
- `decomposition.tex` — LaTeX document on causal effect decomposition
- `decomposition.pdf` — Compiled PDF

## Key Topics Covered

1. **Sensitivity Analysis** — Rosenbaum bounds and related methods for assessing robustness to unmeasured confounding
2. **Effect Decomposition** — Breaking down total effects into direct and indirect components
3. **Identification Assumptions** — Formal conditions required for causal identification

## Audience

Reference material for understanding the theoretical foundations of:
- Selection on observables assumptions in `panel/`
- Regression discontinuity validity in `regression-discontinuity/`
- Instrumental variables logic in `shopping-episode/`
- Experimental design in `holdouts/`

## Connections

- Provides theoretical background for all empirical analyses
- Sensitivity bounds applicable to `panel/` and `shopping-episode/` estimates
- Decomposition relevant to `causal-attribution/` direct vs indirect effects
