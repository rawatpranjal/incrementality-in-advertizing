# Causal Attribution (Adstock Models)

## Overview

Attribution modeling using adstock transformations and distributed lag models to capture the delayed and decaying effects of ad exposures on purchases. Adstock models account for carryover effects where past exposures continue to influence current outcomes with diminishing impact over time.

## Data Requirements

- **Unit of analysis:** User×Week with lagged exposure history
- **Input tables:** IMPRESSIONS, CLICKS, PURCHASES
- **Sample/filters:** Users with multi-week observation windows
- **Temporal structure:** Weekly aggregation with multiple lags

## Pipeline

1. `01_adstock_data_pull.ipynb` — Extract user-level ad exposure and purchase data
2. `02_adstock_panel_construction.ipynb` — Construct panel with adstock transformations

## Model Specification

**Adstock Transformation:**
```
Adstock_t = X_t + λ·Adstock_{t-1}
```
where λ ∈ [0,1] is the decay rate

**Equivalent Geometric Distributed Lag:**
```
Adstock_t = Σ_{k=0}^∞ λ^k · X_{t-k}
```

**Regression with Adstock:**
```
Y_t = β_0 + β_1·Adstock(Clicks)_t + β_2·Adstock(Impressions)_t + γ·Z_t + ε_t
```

**Alternative: Polynomial Distributed Lag (PDL):**
```
Y_t = α + Σ_{k=0}^K β_k·X_{t-k} + ε_t
```
with β_k parameterized by polynomial in k

**Variables:**
- Y_t: Purchases in period t
- X_t: Ad exposures (clicks, impressions) in period t
- λ: Adstock decay parameter (estimated or calibrated)
- K: Maximum lag length

**Interpretation:**
- β_1 captures total effect of clicks inclusive of carryover
- λ indicates persistence: high λ means effects last longer
- Half-life = log(0.5)/log(λ) periods

## Key Files

| File | Purpose |
|------|---------|
| `01_adstock_data_pull.ipynb` | Data extraction for attribution |
| `02_adstock_panel_construction.ipynb` | Adstock panel construction and estimation |

## Outputs

- Adstock decay parameter estimates
- Total and marginal effect estimates
- Lag distribution weights
- Attribution percentages by touchpoint

## Connections

- Relates to `time-series/ardl/` for ARDL bounds testing approach
- Complements `time-series/var/` IRF analysis for dynamic effects
- Informs `shopping-sessions/` Netflix-style attribution methodology
