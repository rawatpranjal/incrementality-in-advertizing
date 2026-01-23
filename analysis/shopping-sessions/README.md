# Shopping Sessions Analysis

## Overview

Session-based analysis that groups user interactions into shopping sessions using gap-based definitions. Compares multiple attribution approaches including a Netflix-inspired methodology. This provides an alternative to the shopping-episode approach with different session boundary definitions.

## Data Requirements

- **Unit of analysis:** Shopping sessions (defined by inactivity gaps)
- **Input tables:** IMPRESSIONS, CLICKS, PURCHASES, AUCTIONS_USERS
- **Sample/filters:** Users with multi-session histories
- **Session definition:** Interactions within time threshold grouped together

## Pipeline

1. `01_pull_shopping_sessions.ipynb` — Extract and sessionize interaction data
2. `02_eda.ipynb` — Exploratory analysis of session characteristics
3. `03_shopping_session_panels.ipynb` — Construct session-level panels
4. `04_shopping_session_estimates.ipynb` — Estimate session-level effects
5. `05_netflix_attribution_approach_panel.ipynb` — Netflix-style attribution panel
6. `06_netflix_attribution_approach_estimates.ipynb` — Netflix attribution estimates
7. `07_netflix_panel.ipynb` — Extended Netflix panel construction
8. `08_netflix_estimates.ipynb` — Final Netflix methodology estimates

## Model Specification

**Session-Level Regression:**
```
Purchase_s = α + β·AdExposure_s + γ·SessionFeatures_s + ε_s
```
where s indexes sessions

**Netflix-Style Attribution:**
Attribution approach that accounts for:
- Multiple touchpoints within session
- Time-decay weighting
- Cross-session effects

**IRF Analysis:**
```
Y_{t+h} = f(Shock_t) for h = 0, 1, 2, ...
```
Impulse response of purchases to ad exposure shocks

**Variables:**
- Y: Session-level purchase indicator or amount
- X: Ad exposures within session (impressions, clicks)
- Session features: Duration, number of interactions, product diversity

**Interpretation:**
- β captures within-session effect of ad exposure on purchase
- Netflix approach provides multi-touch attribution weights
- IRF shows persistence of effects across sessions

## Key Files

| File | Purpose |
|------|---------|
| `01_pull_shopping_sessions.ipynb` | Data extraction and sessionization |
| `02_eda.ipynb` | Session EDA |
| `03_shopping_session_panels.ipynb` | Panel construction |
| `04_shopping_session_estimates.ipynb` | Main estimation |
| `05_netflix_attribution_approach_panel.ipynb` | Netflix panel |
| `06_netflix_attribution_approach_estimates.ipynb` | Netflix estimates |
| `07_netflix_panel.ipynb` | Extended Netflix panel |
| `08_netflix_estimates.ipynb` | Final Netflix estimates |
| `generate_irf_plot.py` | IRF visualization |
| `test_vcov.py` | Variance-covariance testing |

## Outputs

- Session-level effect estimates
- Attribution weights by touchpoint
- IRF plots
- Comparison across attribution methodologies

## Connections

- Alternative to `shopping-episode/` with different session definitions
- Relates to `causal-attribution/` adstock approach
- IRF analysis complements `time-series/` dynamic modeling
