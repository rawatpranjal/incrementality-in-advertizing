# Deep Learning Analysis

## Overview

Deep neural network approaches for estimating treatment effects, including comparisons with traditional econometric methods. Explores nonlinear treatment effect functions, heterogeneous effects via Double ML, and G-computation methods. Serves as a benchmark for whether neural networks improve upon linear fixed effects models.

## Data Requirements

- **Unit of analysis:** Vendor×Week panels, User-level
- **Input tables:** AUCTIONS_RESULTS, CLICKS, IMPRESSIONS, PURCHASES
- **Sample/filters:** Varies by script; some use simulated data for validation
- **Features:** Ad metrics, vendor characteristics, time indicators

## Pipeline

1. Simulation scripts — Validate methods on known DGPs
2. Panel estimation scripts — Apply to real vendor-week data
3. Comparison scripts — Benchmark neural nets vs econometrics

## Model Specifications

**Deep Neural Network Fixed Effects:**
```
Y_vw = f_θ(X_vw) + α_v + λ_w + ε_vw
```
where f_θ is a neural network with parameters θ

**Double/Debiased ML:**
```
Stage 1: E[D|X] = g(X)  (propensity)
Stage 2: E[Y|X] = m(X)  (outcome)
Stage 3: θ = E[(Y - m(X))/(D - g(X))]
```
Orthogonalized estimation of treatment effect

**G-Computation:**
```
ATE = E[Y(1)] - E[Y(0)]
    = E[E[Y|D=1,X]] - E[E[Y|D=0,X]]
```
Outcome model-based effect estimation

**Heterogeneous Treatment Effects:**
```
τ(x) = E[Y(1) - Y(0) | X=x]
```
CATE estimation via neural network

**Interpretation:**
- Neural networks can capture nonlinear dose-response relationships
- Double ML provides valid inference even with flexible ML first stages
- Comparison reveals whether nonlinearity matters empirically

## Key Files

| File | Purpose |
|------|---------|
| `vendor_week_panel_deep_learning_fixed_effects_neural_network_analysis.py` | DNN with FE on real data |
| `vendor_week_panel_r_feols_fixed_effects_econometric_analysis.py` | R fixest benchmark |
| `panel_fixed_effects_deep_learning_vs_traditional_econometrics_comparison.py` | Head-to-head comparison |
| `panel_nonlinear_beta_function_deep_neural_network_high_dimensional.py` | Nonlinear treatment functions |
| `panel_two_way_fixed_effects_r_fixest_integration_simulation.py` | Simulation validation |
| `heterogeneous_treatment_effects_simulation.py` | HTE simulation |
| `heterogeneous_treatment_effects_simulation_scaled.py` | Scaled HTE simulation |
| `heterogeneous_effects_with_g_demo.py` | G-computation demo |
| `test_scaled_simulation_demo.py` | Testing utilities |
| `PERCENTAGE_METRICS_SUMMARY.md` | Results summary |

## Outputs

- Neural network vs econometric comparison tables
- Nonlinear dose-response estimates
- CATE distributions
- Simulation coverage and bias diagnostics
- Results in stdout captured to .txt files

## Connections

- Direct comparison with `panel/` fixed effects models
- Builds on `time-series/` dynamic specifications
- Informs whether complex methods improve on simpler approaches
