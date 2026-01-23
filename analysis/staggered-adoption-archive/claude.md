# Staggered Adoption DiD: Methodology Reference

This document provides a high-level methodology guide for the incrementality analysis pipeline. Read alongside `results/FULL_RESULTS.txt` to understand the complete analysis.

---

## 1. Research Question

**Primary Question:** Does advertising on the platform cause incremental sales for vendors?

**Setting:** Marketplace platform where vendors (sellers) pay to promote product listings. Vendors adopt advertising at different times (staggered treatment).

---

## 2. Data Structure

### Panel Definition
```
Unit of Analysis: VENDOR_ID × WEEK (i, t)
Observations: 846,430 vendor-weeks
Vendors: 142,920 unique vendors
Time Period: 26 weeks (2025-03-24 to 2025-09-15)
```

### Outcome Variables
```
Y_it ∈ {
    promoted_gmv:     GMV from click-attributed purchases (7-day window)
    log_promoted_gmv: Log transformation of promoted GMV
    clicks:           Click count on promoted listings
    impressions:      Impression count on promoted listings
    organic_gmv:      GMV from non-attributed purchases
    total_gmv:        All GMV regardless of attribution
}
```

### Treatment Variables
```
total_spend:    Sum of winning bids (ad expenditure)
has_spend:      Binary indicator (total_spend > 0)
```

### Attribution Logic
```
A purchase is "promoted" if:
1. USER_ID matches a prior click
2. PRODUCT_ID matches the clicked product
3. purchase_time > click_time
4. purchase_time ≤ click_time + 7 days
```

---

## 3. Treatment Definition

### Cohort Assignment
```
G_i = min{t : Spend_it > 0}    First week vendor i has positive ad spend
G_i = ∞                         Never-treated vendors (control group)
```

### Treatment Indicator
```
D_it = 1{t ≥ G_i}              = 1 if vendor is treated at time t, 0 otherwise
```

### Relative Time (Event Time)
```
e = t - G_i                    Weeks relative to treatment adoption
e < 0:                         Pre-treatment periods
e = 0:                         Treatment period
e > 0:                         Post-treatment periods
```

### Sample Composition
```
Treated vendors:      139,356 (97.5%)
Never-treated:          3,564 (2.5%)
Treatment cohorts:          26 (one per week)
```

---

## 4. Core Estimators

### 4.1 Callaway-Sant'Anna Group-Time ATT (Script 04)

**Estimand:**
```
ATT(g, t) = E[Y_it(1) - Y_it(0) | G_i = g]
```

**Estimation (Never-Treated Comparison):**
```
ATT(g, t) = {E[Y_it | G = g] - E[Y_{i,g-1} | G = g]}
          - {E[Y_it | G = ∞] - E[Y_{i,g-1} | G = ∞]}

Where:
- g = treatment cohort (week of first ad spend)
- t = calendar time
- G = ∞ for never-treated units
- Y_{i,g-1} = outcome in period before treatment (baseline)
```

### 4.2 Event Study Aggregation (Script 05)

**Aggregation Function:**
```
θ(e) = Σ_g w_g × ATT(g, g+e)

w_g = n_g / Σ_g n_g    (cohort-size weights)
```

**Pre-Trend Test:**
```
H_0: θ(e) = 0  for all e < 0
```

**Average Post-Treatment Effect:**
```
θ̄_post = (1/|E_post|) Σ_{e≥0} θ(e)
```

### 4.3 Two-Way Fixed Effects (Script 06)

**Static TWFE:**
```
Y_it = α_i + λ_t + β D_it + ε_it

Where:
- α_i = vendor fixed effect (time-invariant heterogeneity)
- λ_t = week fixed effect (common time shocks)
- β   = average treatment effect (ATT under parallel trends)
- ε_it = idiosyncratic error
```

**Dynamic TWFE:**
```
Y_it = α_i + λ_t + Σ_e β_e × 1{t - G_i = e} + ε_it

β_e = effect at relative time e
```

**Note:** TWFE places negative weights on some ATT(g,t) when treatment effects are heterogeneous across cohorts or time. Already-treated units can serve as implicit controls for newly-treated units.

---

## 5. Identification Assumptions

### 5.1 Parallel Trends (Conditional)
```
E[Y_it(0) - Y_{i,t-1}(0) | G = g] = E[Y_it(0) - Y_{i,t-1}(0) | G = ∞]
```

**Testable Implication:** θ(e) ≈ 0 for e < 0

### 5.2 No Anticipation
```
Y_it(g) = Y_it(0)  for all t < g
```

### 5.3 Irreversibility
```
D_it = 1 for all t ≥ G_i
```

### 5.4 SUTVA
```
Y_it(D) = Y_it(D_it)
```

---

## 6. Extended Estimators

### 6.1 IV-2SLS for Elasticity (Script 07)

**First Stage:**
```
ln(Spend_it) = π Z_it + μ_i + λ_t + ν_it
```

**Second Stage:**
```
ln(GMV_it) = β ln(Ŝpend_it) + μ_i + λ_t + ε_it
```

**Instrument:** Z_it = Auction competition intensity (average competing bidders in vendor's auctions, excluding vendor's own bids)

### 6.2 Cannibalization Test (Script 08)

```
Organic_GMV_it = α + δ × Promoted_GMV_it + μ_i + λ_t + ε_it

δ = 0
δ = -1
-1 < δ < 0
δ > 0
```

**Incremental ROAS Framework:**
```
Incremental_Sales = Promoted_Sales × (1 + δ)
iROAS = (1 + δ) × observed_ROAS
```

### 6.3 Doubly Robust DiD (Script 16)

**Propensity Score:**
```
P(T = 1 | X) = logit(X'γ)

X = {pre_auction_count, pre_avg_price_point, pre_weeks_active, ...}
```

**IPW Weights (for controls):**
```
w_i = P̂(T=1|X_i) / (1 - P̂(T=1|X_i))
```

**IPW Estimator:**
```
ATT_IPW = Ȳ_treated - (Σ_j w_j Y_j) / (Σ_j w_j)
```

### 6.4 Extensive/Intensive Margin (Script 17)

**Extensive Margin (any sale):**
```
any_sale_it = 1{promoted_gmv_it > 0}

P(any_sale = 1) = Φ(α_i + λ_t + β_ext × D_it)
```

**Intensive Margin (conditional GMV):**
```
E[log(GMV) | GMV > 0] = α_i + λ_t + β_int × D_it + ε_it
```

**Total Effect Decomposition:**
```
Total = Extensive + Intensive
```

---

## 7. Heterogeneous Treatment Effects

### 7.1 Vendor Segmentation (Script 11)

**Activity Quartiles:**
```
Q1_Inactive:  pre_auction_count = 0
Q2_Low:       1-25th percentile
Q3_Medium:    25-50th percentile
Q4_High:      50-75th percentile
Q5_VeryHigh:  75-100th percentile
```

**Price Point Quartiles:**
```
P0_Unknown:   No pre-treatment price data
P1_Budget:    1-25th percentile of avg price
P2_MidLow:    25-50th percentile
P3_MidHigh:   50-75th percentile
P4_Premium:   75-100th percentile
```

**Vendor Personas (Decision Tree):**
```
1. New_Adopter:      pre_auction_count = 0 (92% of vendors)
2. Power_Seller:     High activity (Q4-Q5) AND good ranking (R1-R2)
3. Premium_Boutique: High price (P4)
4. Active_Generalist: Medium-high activity (Q3-Q4)
5. Casual_Seller:    Remainder
```

### 7.2 HTE Estimation (Script 12)

Estimate TWFE separately by segment:
```
Y_it = β_s × D_it + α_i + λ_t + ε_it    for segment s
```

### 7.3 Conditional Parallel Trends (Script 13)

**Standardized Difference:**
```
d = (μ_treated - μ_control) / sqrt((σ²_treated + σ²_control) / 2)

Thresholds: |d| < 0.10, |d| < 0.25, |d| > 0.25
```

---

## 8. Economic Metrics

### 8.1 Return on Ad Spend (Script 14)

**Observed ROAS:**
```
ROAS = Σ promoted_gmv / Σ total_spend
```

**Incremental ROAS:**
```
iROAS = (exp(ATT) - 1) × (avg_baseline_gmv / avg_spend)
```

**Profitability Threshold:**
```
ROAS > 1
ROAS < 1
```

---

## 9. Results Summary

### Primary Findings

| Outcome | ATT | SE | t-stat | p-value |
|---------|-----|-----|--------|---------|
| log(promoted_gmv) | 0.000584 | 0.000273 | 2.14 | 0.032 |
| Clicks | 0.036 | 0.005 | 7.2 | <0.001 |
| Impressions | 1.079 | 0.15 | 7.2 | <0.001 |

### Economic Metrics

| Metric | Value |
|--------|-------|
| Overall ROAS | 0.09 |
| Vendors with ROAS > 1 | 198 / 139,356 (0.1%) |
| Pre-trends p-value | 1.0 |

### Robustness Checks

| Test | Result |
|------|--------|
| Pre-trends | θ(e) ≈ 0 for e < 0 |
| TWFE estimate | 0.00058 |
| CS estimate | 0.00008 |
| IPW estimate | 0.00132 |
| DR estimate | 0.00132 |

### Extensive Margin

| Margin | ATT | p-value |
|--------|-----|---------|
| Extensive (P > 0) | 0.000077 | 0.024 |
| Intensive (E\|>0) | — | — |

---

## 10. Script Reference

| Script | Purpose | Key Output |
|--------|---------|------------|
| 00_explore_cohorts | Data exploration | Cohort sizes, funnel summary |
| 01_data_audit | Join validation | Orphan rates, attribution feasibility |
| 02_panel_construction | Build panel | Vendor×week dataset with outcomes |
| 03_cohort_assignment | Treatment timing | G_i, D_it, relative time |
| 04_callaway_santanna | CS estimator | ATT(g,t) for all cohort-time pairs |
| 05_event_study_aggregation | Aggregate to event-time | θ(e) coefficients, PT tests |
| 06_twfe_comparison | Compare estimators | TWFE vs CS, bias demonstration |
| 07_iv_2sls | Instrumental variables | GMV elasticity w.r.t. spend |
| 08_cannibalization | Substitution test | δ coefficient, ROAS by quartile |
| 09_robustness_diagnostics | Multiple checks | Placebo, balance, attrition |
| 10_pretreatment_covariates | Build covariates | X_i for HTE and balancing |
| 11_vendor_segmentation | Create segments | Personas, quartiles |
| 12_hte_by_segment | Segment-level ATT | ATT by persona/quartile |
| 13_conditional_parallel_trends | Within-segment PT | Balance tables, covariate adjustment |
| 14_segment_roas | Profitability | ROAS and iROAS by segment |
| 15_placement_category_quality | Context analysis | Performance by placement/category |
| 16_doubly_robust_did | Estimator comparison | TWFE vs IPW vs OR vs DR |
| 17_extensive_margin | Margin decomposition | Extensive/intensive effects |
| 99_combine_results | Aggregate outputs | FULL_RESULTS.txt |

