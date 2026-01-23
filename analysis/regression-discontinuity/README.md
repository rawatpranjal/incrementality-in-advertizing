# Regression Discontinuity Analysis

## Overview

Regression discontinuity (RD) design exploiting the auction mechanism where ads are shown based on bid rankings. The sharp discontinuity at the winner threshold (rank = 1 wins impression) provides quasi-experimental variation for estimating the causal effect of winning an ad impression on downstream outcomes.

## Data Requirements

- **Unit of analysis:** Auction×Product (bid-level)
- **Running variable:** Auction ranking position (RANKING from AUCTIONS_RESULTS)
- **Cutoff:** Rank = 1 (winner threshold)
- **Input tables:** AUCTIONS_RESULTS (rankings, bids), IMPRESSIONS, CLICKS, PURCHASES
- **Sample/filters:** Competitive auctions with multiple bidders; focus on marginal winners/losers

## Pipeline

1. `rd_eda.ipynb` — Exploratory data analysis of ranking distributions and density
2. `funnel.ipynb` — Funnel analysis across auction stages
3. `rd.ipynb` — Main RD estimation

## Model Specification

**Sharp RD:**
```
Y_i = α + τ·D_i + f(X_i - c) + D_i·g(X_i - c) + ε_i
```
where:
- X_i = ranking (running variable)
- c = 1 (cutoff)
- D_i = 1{X_i ≤ c} (treatment: won impression)
- f(·), g(·) = polynomial or local linear functions

**Local Linear Regression (preferred):**
```
Y_i = α + τ·D_i + β_1·(X_i - c) + β_2·D_i·(X_i - c) + ε_i
```
estimated within bandwidth h of cutoff

**Variables:**
- Y: Click indicator, purchase indicator, or revenue
- D: Treatment (won the auction, rank ≤ threshold)
- X: Ranking position (lower = better)

**Interpretation:**
- τ captures the local average treatment effect (LATE) at the cutoff
- Identifies effect for marginal winners—those just barely winning vs just barely losing
- Requires continuity of potential outcomes at cutoff (no manipulation)

## Key Files

| File | Purpose |
|------|---------|
| `rd_eda.ipynb` | EDA: ranking distributions, density tests |
| `funnel.ipynb` | Ad funnel stage analysis |
| `rd.ipynb` | Main RD estimation and inference |

## Outputs

- RD treatment effect estimates with robust standard errors
- Bandwidth sensitivity analysis
- McCrary density test for manipulation
- Covariate balance at cutoff
- RD plots showing discontinuity

## Connections

- Complements `panel/` by using auction-level rather than aggregate variation
- Relates to `holdouts/` as another source of causal identification
- Auction mechanism connects to `shopping-episode/` bid-level analysis
