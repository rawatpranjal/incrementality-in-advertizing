# Panel Fixed Effects Analysis

## Overview

Panel data analysis using two-way fixed effects models to estimate the causal effect of ad clicks on purchases. Multiple units of analysis are examined (Vendor×Week, User×Week, User×Vendor×Week) to assess robustness and understand heterogeneity in treatment effects.

## Data Requirements

- **Unit of analysis:** Three panel configurations:
  - Vendor×Week: Aggregate vendor-level weekly outcomes
  - User×Week: User-level weekly outcomes
  - User×Vendor×Week: User-vendor pair weekly outcomes
- **Input tables:** AUCTIONS_RESULTS, CLICKS, PURCHASES, IMPRESSIONS
- **Sample/filters:** Joined on AUCTION_ID×PRODUCT_ID composite key; purchases linked to ad funnel events

## Pipeline

1. `vendor_week.ipynb` — Vendor-level weekly panel construction and estimation
2. `user_week.ipynb` — User-level weekly panel construction and estimation
3. `user_vendor_week.ipynb` — User×Vendor weekly panel construction and estimation
4. `user_vendor_feols.ipynb` — R fixest integration for high-dimensional fixed effects

## Model Specification

**Equation (Vendor×Week):**
```
Purchases_vw = α_v + λ_w + β·Clicks_vw + ε_vw
```

**Equation (User×Week):**
```
Purchases_uw = α_u + λ_w + β·Clicks_uw + ε_uw
```

**Equation (User×Vendor×Week):**
```
Purchases_uvw = α_uv + λ_w + β·Clicks_uvw + ε_uvw
```

**Variables:**
- Y (Purchases): Count or indicator of purchase outcomes
- X (Clicks): Count of ad clicks (treatment intensity)
- α: Unit fixed effects (absorbs time-invariant confounders)
- λ: Time fixed effects (absorbs aggregate shocks)

**Interpretation:**
- β captures the within-unit effect of an additional click on purchases, controlling for unit and time heterogeneity
- Identification assumes parallel trends conditional on fixed effects

## Key Files

| File | Purpose |
|------|---------|
| `vendor_week.ipynb` | Vendor-level panel analysis |
| `user_week.ipynb` | User-level panel analysis |
| `user_vendor_week.ipynb` | User×Vendor panel analysis |
| `user_vendor_feols.ipynb` | R fixest high-dimensional FE estimation |

## Outputs

- Coefficient estimates for click effects with clustered standard errors
- Comparison across panel specifications
- Diagnostic statistics (R², within-R², F-statistics)

## Connections

- Relates to `deep-learning/` for nonlinear extensions and neural network comparisons
- Relates to `time-series/` for aggregate dynamics beyond cross-sectional variation
- Feeds into overall incrementality estimates synthesis
