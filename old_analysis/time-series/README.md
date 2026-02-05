# Time Series Analysis

## Overview

Time series econometric methods for analyzing dynamic relationships between ad exposures (impressions, clicks) and conversions (purchases). Employs vector autoregression (VAR), structural VAR (SVAR), vector error correction models (VECM), ARDL bounds testing, and factor-augmented approaches to capture lagged effects and feedback dynamics.

## Data Requirements

- **Unit of analysis:** Aggregate time series at half-hourly/hourly/daily frequency
- **Input tables:** CLICKS, IMPRESSIONS, PURCHASES aggregated over time
- **Sample/filters:** Time period 2025-03-01 to 2025-09-30
- **Data files:** `data/half_hourly_clicks_*.parquet`, `data/half_hourly_purchases_*.parquet`

## Pipeline

1. `time-series-data-collection.ipynb` — Data extraction and aggregation
2. `unit-roots-and-seasonal/` — Stationarity testing and seasonal adjustment
3. `var/` — VAR and SVAR estimation
4. `vecm/` — Cointegration and VECM analysis
5. `ardl/` — ARDL bounds testing
6. `dfm/` — Dynamic factor models
7. `favar/` — Factor-augmented VAR
8. `its/` — Interrupted time series analysis

## Model Specifications

**VAR(p):**
```
Y_t = c + A_1·Y_{t-1} + A_2·Y_{t-2} + ... + A_p·Y_{t-p} + ε_t
```
where Y_t = [Clicks_t, Purchases_t]'

**SVAR (Recursive identification):**
```
B_0·Y_t = c + B_1·Y_{t-1} + ... + B_p·Y_{t-p} + u_t
```
with B_0 lower triangular (Cholesky decomposition)

**VECM (if cointegrated):**
```
ΔY_t = Π·Y_{t-1} + Γ_1·ΔY_{t-1} + ... + Γ_{p-1}·ΔY_{t-p+1} + ε_t
```
where Π = αβ' captures long-run equilibrium adjustment

**ARDL(p,q):**
```
Y_t = α + Σ_{i=1}^p φ_i·Y_{t-i} + Σ_{j=0}^q β_j·X_{t-j} + ε_t
```

**Interpretation:**
- Impulse response functions (IRF) show dynamic effect of click shock on purchases over time
- Forecast error variance decomposition (FEVD) quantifies contribution of clicks to purchase variation
- Granger causality tests assess predictive relationships

## Key Files

| File | Purpose |
|------|---------|
| `var/svar_analysis.R` | Structural VAR estimation in R |
| `var/svar_analysis.py` | SVAR estimation in Python |
| `var/bsvar_*.R` | Bayesian SVAR with various priors |
| `vecm/vecm_master.py` | VECM cointegration analysis |
| `ardl/ardl_master.py` | ARDL bounds testing |
| `dfm/dfm_unified_analysis.py` | Dynamic factor model |
| `favar/favar_analysis.py` | Factor-augmented VAR |
| `its/policy_impact_consolidated.py` | Interrupted time series |
| `unit-roots-and-seasonal/integrated_comprehensive_analysis.py` | Stationarity diagnostics |

## Subfolders

| Folder | Contents |
|--------|----------|
| `var/` | VAR, SVAR, Bayesian SVAR scripts |
| `vecm/` | Vector error correction models |
| `ardl/` | Autoregressive distributed lag models |
| `dfm/` | Dynamic factor models |
| `favar/` | Factor-augmented VAR |
| `its/` | Interrupted time series |
| `unit-roots-and-seasonal/` | Unit root tests, seasonal adjustment |
| `data/` | Parquet data files |
| `papers/` | Reference literature |

## Outputs

- IRF plots showing dynamic click→purchase responses
- FEVD tables decomposing variance sources
- Granger causality test statistics
- Cointegration rank tests
- Results stored in `*/results/*.txt`

## Connections

- Relates to `panel/` for cross-sectional vs time-series comparison
- Relates to `causal-attribution/` for distributed lag and adstock modeling
- Complements `shopping-sessions/` IRF analysis
