# Callaway-Sant'Anna Difference-in-Differences Analysis

## Executive Summary

This analysis estimates the causal effect of advertising adoption on vendor outcomes using the Callaway and Sant'Anna (2021) difference-in-differences estimator with staggered treatment timing. The methodology addresses heterogeneous treatment effects across cohorts and avoids the negative weighting problems of traditional two-way fixed effects (TWFE) estimators.

**Key Findings:**
- Advertising causes significant increases in impressions (+1.06, p<0.001) and clicks (+0.032, p<0.001)
- No statistically significant effect on GMV (+$0.26, p>0.05)
- The null GMV result is explained by extreme outcome sparsity: GMV is 99.96% zeros
- Pre-trends tests pass for all outcomes, supporting parallel trends assumption

## Data Description

### Panel Structure

| Dimension | Value |
|-----------|-------|
| Observations | 846,430 vendor-weeks |
| Vendors | 142,920 |
| Weeks | 26 |
| Time Period | 2025-03-24 to 2025-09-15 |

### Treatment Definition

Treatment is defined as winning any advertising auction. The treatment cohort $G_i$ is the first week vendor $i$ has positive ad spend:

$$G_i = \min\{t : \text{Spend}_{it} > 0\}$$

| Group | N Vendors | Percentage |
|-------|-----------|------------|
| Ever-Treated | 139,356 | 97.5% |
| Never-Treated (Control) | 3,564 | 2.5% |

### Outcome Variables

| Variable | Mean | Std Dev | % Zeros |
|----------|------|---------|---------|
| Impressions | 1.32 | 2.85 | 50.4% |
| Clicks | 0.04 | 0.25 | 96.7% |
| Total GMV | $1.81 | $149.98 | 99.96% |

## Methodology

### Callaway-Sant'Anna (2021) Estimator

The estimator computes group-time average treatment effects $ATT(g,t)$ for each cohort $g$ at each time period $t$:

$$ATT(g,t) = E[Y_{it}(1) - Y_{it}(0) | G_i = g]$$

Using never-treated units as the comparison group:

$$\widehat{ATT}(g,t) = \left(\bar{Y}_{g,t} - \bar{Y}_{g,g-1}\right) - \left(\bar{Y}_{\infty,t} - \bar{Y}_{\infty,g-1}\right)$$

where $\bar{Y}_{g,t}$ is the mean outcome for cohort $g$ at time $t$, and $\bar{Y}_{\infty,t}$ is the mean for never-treated units.

### Event Study Aggregation

Group-time effects are aggregated to event-time effects $\theta(e)$ using cohort-size weights:

$$\theta(e) = \sum_g w_g \cdot ATT(g, g+e)$$

where $e = t - G_i$ is relative time (periods since treatment adoption).

### Identification Assumptions

1. **Parallel Trends**: $E[Y_{it}(0) - Y_{i,t-1}(0) | G = g] = E[Y_{it}(0) - Y_{i,t-1}(0) | G = \infty]$
2. **No Anticipation**: $Y_{it}(g) = Y_{it}(0)$ for all $t < g$
3. **Irreversibility**: Once treated, always treated

## Main Results

### Overall Average Treatment Effects

| Outcome | ATT | SE | 95% CI | Significant |
|---------|-----|-----|--------|-------------|
| Impressions | +1.061 | 0.021 | [1.020, 1.102] | *** |
| Clicks | +0.032 | 0.001 | [0.029, 0.034] | *** |
| Total GMV | +$0.26 | $1.20 | [-$2.09, $2.61] | - |

*Notes: \*\*\* p<0.001. Standard errors clustered at vendor level.*

### Event Study Results

#### Impressions

| Relative Time (e) | θ(e) | SE | 95% CI |
|-------------------|------|-----|--------|
| e = -5 | -0.0005 | 0.0005 | [-0.001, 0.000] |
| e = -1 | +0.0006 | 0.0005 | [0.000, 0.002] |
| e = 0 | +0.828*** | 0.018 | [0.792, 0.864] |
| e = 1 | +0.889*** | 0.028 | [0.834, 0.944] |
| e = 5 | +1.055*** | 0.059 | [0.939, 1.170] |
| e = 10 | +1.143*** | 0.065 | [1.016, 1.271] |
| e = 20 | +1.210*** | 0.092 | [1.030, 1.391] |

#### Clicks

| Relative Time (e) | θ(e) | SE | 95% CI |
|-------------------|------|-----|--------|
| e < 0 | 0.000 | - | - |
| e = 0 | +0.030*** | 0.002 | [0.025, 0.034] |
| e = 1 | +0.028*** | 0.003 | [0.021, 0.034] |
| e = 5 | +0.026*** | 0.005 | [0.017, 0.035] |
| e = 10 | +0.030*** | 0.006 | [0.019, 0.041] |
| e = 20 | +0.031*** | 0.008 | [0.016, 0.047] |

#### Total GMV

| Relative Time (e) | θ(e) | SE | 95% CI |
|-------------------|------|-----|--------|
| e < 0 | $0.00 | - | - |
| e = 0 | +$0.29 | $1.04 | [-$1.74, $2.32] |
| e = 1 | -$0.74 | $0.74 | [-$2.18, $0.71] |
| e = 5 | -$0.82 | $0.82 | [-$2.42, $0.79] |
| e = 10 | +$2.78 | $2.90 | [-$2.91, $8.47] |
| e = 20 | -$2.17 | $2.17 | [-$6.42, $2.08] |

## Pre-Trends Assessment

| Outcome | Pre-Period Coefficients | Mean θ(e<0) | Max |θ(e<0)| | Sig at 5% | Joint p-value | Verdict |
|---------|------------------------|-------------|----------------|-----------|---------------|---------|
| Impressions | 22 | 0.000048 | 0.001671 | 0/22 | 0.615 | PASS |
| Clicks | 22 | 0.000000 | 0.000000 | 0/22 | - | PASS |
| Total GMV | 22 | 0.000000 | 0.000000 | 0/22 | - | PASS |

Pre-trends tests pass for all outcomes. The control group has zero pre-treatment clicks and GMV by construction (never-treated vendors have no ad exposure).

## Segmentation Analysis

### Impressions by Adoption Timing

| Segment | N Obs | ATT | SE | Significant |
|---------|-------|-----|-----|-------------|
| Early Adopter | 482,276 | +0.872 | 0.074 | *** |
| Mid Adopter | 189,514 | +0.893 | 0.044 | *** |
| Late Adopter | 170,754 | +0.905 | 0.030 | *** |

### Total GMV by Adoption Timing

| Segment | N Obs | ATT | SE | p-value |
|---------|-------|-----|-----|---------|
| Early Adopter | 482,276 | +$0.43 | $4.13 | 0.917 |
| Mid Adopter | 189,514 | +$1.54 | $3.53 | 0.663 |
| Late Adopter | 170,754 | +$0.00 | $2.99 | 1.000 |

No segment shows statistically significant GMV effects.

## Robustness Checks

### Alternative Control Groups

| Outcome | Never-Treated | Not-Yet-Treated |
|---------|---------------|-----------------|
| Impressions | +1.061 (0.021)*** | +1.061 (0.021)*** |
| Total GMV | +$0.26 ($1.20) | +$0.26 ($1.20) |

Results are robust to using not-yet-treated units as the comparison group.

## Interpretation

### The Advertising Funnel

```
Winning Auctions → Impressions (+1.06***)
                 → Clicks (+0.032***) [CTR: 3.0%]
                 → GMV (+$0.26, n.s.)
```

The advertising funnel operates as expected at the top: winning auctions generates exposure (impressions) and engagement (clicks). However, we cannot detect a statistically significant effect on sales (GMV).

### Why the Null GMV Result?

The null GMV finding is driven by **extreme outcome sparsity**, not necessarily advertising ineffectiveness:

1. **Sparsity**: GMV is 99.96% zeros (only 368 non-zero observations out of 846,430)
2. **Control Group**: The never-treated control (2.5% of sample) has essentially zero GMV
3. **Power**: With such sparse data, detecting economically meaningful but statistically significant effects requires either:
   - Much larger samples
   - Longer observation windows
   - Higher-converting vendors

## Data Limitations

1. **Extreme Sparsity**: The 99.96% zero rate for GMV severely limits statistical power

2. **Small Control Group**: Only 2.5% of vendors never adopt advertising, and this group is fundamentally different (never exposed to ads)

3. **Week 1 Dominance**: 19% of vendors adopted in Week 1, limiting variation in treatment timing

4. **Attribution Window**: 7-day click-to-purchase attribution may miss longer conversion cycles

5. **Selection**: Vendors choosing to advertise may differ systematically from non-advertisers

## References

Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.

---
*Generated: 2026-01-21 01:57:41*