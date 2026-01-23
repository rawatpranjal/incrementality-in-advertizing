# Multi-Day Shopping Session Analysis

## 1. Data

### Source
- Dataset: `shopping-sessions/data/raw_sample_*.parquet`
- Date range: 2025-03-25 to 2025-09-20
- Duration: 179 days (26 weeks)

### Raw Counts

| Table | Rows |
|-------|------|
| Auctions | 286,422 |
| Bids | 11,254,106 |
| Impressions | 1,118,310 |
| Clicks | 34,260 |
| Purchases | 4,904 |

### Unique Entities

| Entity | Count |
|--------|-------|
| Users in clicks | 1,544 |
| Users in purchases | 931 |
| Users in both | 701 |
| Vendors | 18,603 |
| Products in clicks | 28,597 |

### Spend

| Metric | Value |
|--------|-------|
| Total spend | $193,034 |
| Post-click spend | $8,334 |
| Post-click spend share | 4.3% |
| Post-click purchases | 237 |

### Panel

| Panel | Rows | Units | Weeks | Vendors |
|-------|------|-------|-------|---------|
| (User, Week, Vendor) | 28,619 | 1,544 users | 26 | 18,603 |
| (Session, Week, Vendor) 1d | 28,902 | 7,669 sessions | 26 | 18,603 |
| (Session, Week, Vendor) 3d | 28,663 | 5,402 sessions | 26 | 18,603 |
| (Session, Week, Vendor) 7d | 28,619 | 3,883 sessions | 26 | 18,603 |

---

## 2. Model Equations

### Model 2: User-Week-Vendor Fixed Effects

$$Y_{utv} = \alpha_u + \lambda_t + \phi_v + \beta \cdot C_{utv} + \varepsilon_{utv}$$

### Model 2 Variants

$$Y_{utv} = \alpha_u + \beta \cdot C_{utv} + \varepsilon_{utv} \quad \text{(User FE only)}$$

$$Y_{utv} = \alpha_u + \lambda_t + \beta \cdot C_{utv} + \varepsilon_{utv} \quad \text{(User + Week FE)}$$

$$Y_{utv} = \alpha_{ut} + \phi_v + \beta \cdot C_{utv} + \varepsilon_{utv} \quad \text{(User×Week + Vendor FE)}$$

### Model 3: Session-Week-Vendor Fixed Effects

$$Y_{stv} = \alpha_s + \lambda_t + \phi_v + \beta \cdot C_{stv} + \varepsilon_{stv}$$

### Placebo Model

$$Y_{u,t-1,v} = \alpha_u + \lambda_t + \phi_v + \beta^{pl} \cdot C_{utv} + \varepsilon_{utv}$$

### Conversion Model

$$D_{utv} = \alpha_u + \lambda_t + \phi_v + \beta^D \cdot C_{utv} + \varepsilon_{utv}$$

where $D_{utv} = \mathbf{1}\{Y_{utv} > 0\}$

### Window Sweep Model

$$Y^{(L)}_{utv} = \alpha_u + \lambda_t + \phi_v + \beta^{(L)} \cdot C_{utv} + \varepsilon_{utv}$$

where $Y^{(L)}_{utv} = \sum_{k=0}^{L} Y_{u,t+k,v}$

### Heterogeneity Model

$$Y_{utv} = \alpha_u + \lambda_t + \phi_v + \beta_0 \cdot C_{utv} + \beta_1 \cdot (C_{utv} \times \text{TopRank}_{utv}) + \varepsilon_{utv}$$

where $\text{TopRank}_{utv} = \mathbf{1}\{\text{avg\_rank}_{utv} \leq 3\}$

### Cross-Vendor Model

$$Y_{utv} = \alpha_u + \lambda_t + \phi_v + \beta \cdot C_{utv} + \gamma \cdot C^{other}_{ut} + \varepsilon_{utv}$$

where $C^{other}_{ut} = \sum_{v' \neq v} C_{utv'}$

### Lagged Model

$$Y_{utv} = \alpha_u + \lambda_t + \phi_v + \beta \cdot C_{utv} + \delta \cdot C^{total}_{u,t-1} + \varepsilon_{utv}$$

### User-Week Aggregation Model

$$Y^{total}_{ut} = \alpha_u + \lambda_t + \beta \cdot C^{total}_{ut} + \varepsilon_{ut}$$

where $Y^{total}_{ut} = \sum_v Y_{utv}$ and $C^{total}_{ut} = \sum_v C_{utv}$

### Variable Definitions

| Variable | Definition |
|----------|------------|
| $Y_{utv}$ | Spend by user $u$ on vendor $v$ in week $t$ |
| $C_{utv}$ | Clicks by user $u$ on vendor $v$ in week $t$ |
| $\alpha_u$ | User fixed effect |
| $\lambda_t$ | Week fixed effect |
| $\phi_v$ | Vendor fixed effect |
| $\alpha_s$ | Session fixed effect |
| $\alpha_{ut}$ | User×Week fixed effect |
| SE | Clustered by user |

---

## 3. Main Results

### Model 2: User-Week-Vendor

| Specification | β | SE | p | N | R² Within |
|---------------|---|----|----|---|-----------|
| User + Week + Vendor | -0.231 | 0.183 | 0.207 | 14,637 | 0.001 |
| User only | 0.248 | 0.130 | 0.056 | 28,173 | 0.001 |
| User + Week | 0.248 | 0.130 | 0.057 | 28,173 | 0.001 |
| User×Week + Vendor | 0.042 | 0.094 | 0.650 | 12,264 | 0.000 |

### Model 3: Session-Week-Vendor

| Gap | Sessions | β | SE | p | N | R² Within |
|-----|----------|---|----|---|---|-----------|
| 1 day | 7,669 | -0.118 | 0.099 | 0.233 | 11,089 | 0.001 |
| 3 days | 5,402 | -0.172 | 0.175 | 0.326 | 12,351 | 0.001 |
| 7 days | 3,883 | -0.238 | 0.199 | 0.232 | 13,254 | 0.001 |

---

## 4. Robustness

### 4.1 Placebo (Past Spend ~ Future Clicks)

| Model | β | SE | p |
|-------|---|----|---|
| Y_t ~ C_{t+1} \| user + week + vendor | 0.648 | 0.512 | 0.210 |

### 4.2 Conversion (D = 1{Y > 0})

| Model | β | SE | p | Baseline D |
|-------|---|----|---|------------|
| D ~ C \| user + week + vendor | -0.002 | 0.003 | 0.601 | 0.82% |

### 4.3 Window Sweep

| L (weeks) | β | SE |
|-----------|---|----|
| 0 | -0.231 | 0.183 |
| 1 | -0.161 | 0.177 |
| 2 | -0.204 | 0.185 |
| 4 | -0.171 | 0.178 |

### 4.4 Heterogeneity by Rank

Model: Y ~ C + C×TopRank | user + week + vendor

(TopRank = 1 if avg_rank ≤ 3)

| Coefficient | β | SE | p |
|-------------|---|----|---|
| C | 0.238 | 0.113 | 0.035 |
| C × TopRank | 0.233 | 0.113 | 0.039 |

---

## 5. Cross-Vendor

### 5.1 Own vs Other Clicks

Model: Y ~ C + C_other | user + week + vendor

| Coefficient | β | SE | p |
|-------------|---|----|---|
| C (own vendor) | -0.226 | 0.181 | 0.213 |
| C_other | -0.001 | 0.001 | 0.167 |

### 5.2 Lagged Total Clicks

Model: Y ~ C + C_total_lag | user + week + vendor

| Coefficient | β | SE | p |
|-------------|---|----|---|
| C | -0.307 | 0.251 | 0.227 |
| C_total_lag | -0.003 | 0.003 | 0.349 |

### 5.3 User-Week Aggregation

Model: Y_total ~ C_total | user + week

| Coefficient | β | SE | p |
|-------------|---|----|---|
| C_total | 0.093 | 0.054 | 0.088 |

---

## 6. Data Characteristics

| Metric | Value |
|--------|-------|
| Vendors with 1 observation | 13,371 (72%) |
| Vendors with >1 observation | 5,232 (28%) |
| User-vendor pairs with C variation | 575 |
| User-vendor pairs with >1 obs | 1,455 |
| Cells with Y > 0 | 235 |
| Cells with C > 0 | 28,577 |
| Cells with C > 1 | 4,316 |

---

## 7. Summary Table

| Model | β | 95% CI | p |
|-------|---|--------|---|
| Full FE (U+W+V) | -0.231 | [-0.590, 0.128] | 0.207 |
| User×Week + Vendor | 0.042 | [-0.142, 0.226] | 0.650 |
| Session FE (1d) | -0.118 | [-0.312, 0.076] | 0.233 |
| Heterogeneity: C | 0.238 | [0.017, 0.459] | 0.035 |
| Heterogeneity: C×TopRank | 0.233 | [0.012, 0.454] | 0.039 |
| Placebo | 0.648 | [-0.356, 1.652] | 0.210 |
| User-Week Agg | 0.093 | [-0.013, 0.199] | 0.088 |

---

## 8. Diagnostics

### 8.1 Sample Construction

| Metric | Value |
|--------|-------|
| Panel rows | 28,619 |
| Cells with C > 0 | 28,577 |
| Cells with C = 0 | 42 |
| Cells with Y > 0 | 235 |
| Singleton users | 446 |
| Singleton vendors | 13,371 |
| Non-singleton vendors | 5,232 |
| Rows from non-singleton vendors | 15,248 |

### 8.2 Within-FE Variation

| Metric | Value |
|--------|-------|
| Total user-vendor pairs | 26,699 |
| Pairs with >1 observation | 1,455 |
| Pairs with C variation (std > 0) | 575 |
| Clicks from pairs with C variation | 2,620 |
| Spend from pairs with C variation | $2,354 |
| % of total clicks from varied pairs | 7.6% |
| % of total spend from varied pairs | 28.2% |

### 8.3 Outcome Scaling

| Metric | Value |
|--------|-------|
| Y range | $0 to $499 |
| Y mean (Y > 0) | $35.46 |
| Y median (Y > 0) | $22.00 |
| Y 99th percentile | $0 |
| C range | 0 to 18 |
| C mean (C > 0) | 1.20 |

### 8.4 Sparsity

| Metric | Value |
|--------|-------|
| Cells with Y > 0 | 235 |
| Cells with C > 0 | 28,577 |
| Cells with both Y > 0 and C > 0 | 193 |
| Cells with Y > 0 but C = 0 | 42 |
| Cells with C > 0 but Y = 0 | 28,384 |

### 8.5 Y Definition

| Metric | Value |
|--------|-------|
| Y definition | Post-click mappable spend only |
| Panel total Y | $8,334 |
| Total platform spend | $193,034 |
| Y / Total spend | 4.3% |

### 8.6 Week Definition

| Metric | Value |
|--------|-------|
| Week definition | ISO calendar week (Monday start) |
| Weeks in data | 26 |
| First week start | 2025-03-25 |
| Last week start | 2025-09-15 |

### 8.7 Session Construction

| Gap | Rows | Sessions |
|-----|------|----------|
| 1 day | 28,902 | 7,669 |
| 3 days | 28,663 | 5,402 |
| 7 days | 28,619 | 3,883 |

### 8.8 Click-Spend Correlation

| Level | Correlation(C, Y) |
|-------|-------------------|
| User-level | 0.248 |
| Vendor-level | 0.075 |

### 8.9 Power Inputs

| Metric | Value |
|--------|-------|
| Baseline conversion (D > 0) | 0.82% |
| N in main FE model | 14,637 |
| SE of β in conversion model | 0.003 |
| Detectable effect at 80% power | 0.0084 |

### 8.10 Heterogeneity Sample

| Metric | Value |
|--------|-------|
| Rows with rank data | 28,568 |
| Rows without rank data | 51 |
| % with rank | 99.8% |

### 8.11 TopRank Distribution

| Threshold | Count | % |
|-----------|-------|---|
| avg_rank ≤ 1 | 3,053 | 10.7% |
| avg_rank ≤ 2 | 5,822 | 20.4% |
| avg_rank ≤ 3 | 7,636 | 26.7% |
| avg_rank ≤ 5 | 10,925 | 38.2% |
| avg_rank ≤ 10 | 17,348 | 60.7% |

### 8.12 Back-of-Envelope

| Metric | Value |
|--------|-------|
| Post-click spend / total clicks | $0.243 |
| Main FE β | -$0.231 |
| User-week agg β | $0.093 |
| TopRank β (non-top) | $0.238 |
| TopRank β (top) | $0.471 |

### 8.13 Non-Singleton Vendors

| Metric | Value |
|--------|-------|
| Rows after dropping singletons | 15,248 |
| Users in non-singleton | 1,212 |
| Vendors remaining | 5,232 |

### 8.14 Overlap Users

| Metric | Value |
|--------|-------|
| Users in both clicks and purchases | 701 |
| Panel rows from overlap users | 24,809 |
| Clicks from overlap users | 29,926 |
| Spend from overlap users | $8,334 |

### 8.15 Week Outliers

**Top 3 weeks by clicks:**

| Week | Clicks |
|------|--------|
| 202534 | 2,045 |
| 202536 | 1,580 |
| 202529 | 1,567 |

**Top 3 weeks by spend:**

| Week | Spend |
|------|-------|
| 202529 | $646 |
| 202533 | $590 |
| 202520 | $566 |

### 8.16 Large Purchases

| Metric | Value |
|--------|-------|
| Max purchase in panel | $499 |
| Purchases > $100 | 9 |
| Purchases > $200 | 3 |
| Purchases > $500 | 0 |
| Share of spend from top 10 cells | 24.1% |

### 8.17 Standard Errors

| Metric | Value |
|--------|-------|
| Clustering | User-level |

---

## 9. Advanced Diagnostics

### 9.0 FE Progression

| Specification | β(C) | SE | p-value | N |
|---------------|------|-----|---------|---|
| None (Pooled OLS) | 0.2213 | 0.1259 | 0.0790 | 28,619 |
| Week only | 0.2220 | 0.1250 | 0.0760 | 28,619 |
| User only | 0.2481 | 0.1299 | 0.0564 | 28,173 |
| Vendor only | -0.1996 | 0.1676 | 0.2338 | 15,248 |
| User + Week | 0.2482 | 0.1303 | 0.0572 | 28,173 |
| User + Vendor | -0.2299 | 0.1834 | 0.2105 | 14,637 |
| Week + Vendor | -0.1982 | 0.1677 | 0.2377 | 15,248 |
| User + Week + Vendor | -0.2313 | 0.1830 | 0.2068 | 14,637 |

### 9.1 The 575 Problem

| Metric | With Variation | Without Variation |
|--------|----------------|-------------------|
| Pairs | 575 | 26,124 |
| Mean obs per pair | 2.58 | 1.04 |
| Mean total clicks | 4.56 | 1.21 |
| Mean total spend | $4.09 | $0.23 |
| Pairs with Y > 0 | 55 | 179 |

Regression on 575-pair subset: β = -0.6595, SE = 0.3612, p = 0.070

### 9.2 Category FE (Instead of Vendor FE)

| Specification | β(C) | SE | p-value |
|---------------|------|-----|---------|
| User + Week + Category (77) | 0.2461 | 0.1315 | 0.0616 |
| User + Week + Department (6) | 0.2480 | 0.1303 | 0.0574 |

### 9.3 Hurdle Model

| Part | Model | β(C) | SE | p-value |
|------|-------|------|-----|---------|
| 1 | D ~ C \| user + week + vendor | -0.0016 | 0.0030 | 0.6009 |
| 2 | log(1+Y) ~ C \| user + vendor (Y > 0) | -0.4925 | inf | nan |

### 9.4 Outlier Sensitivity

**Top 10 purchases:**

| Y | C | Week |
|---|---|------|
| $499 | 3 | 202524 |
| $300 | 1 | 202520 |
| $215 | 0 | 202534 |
| $200 | 1 | 202531 |
| $150 | 1 | 202521 |
| $150 | 0 | 202529 |
| $150 | 0 | 202537 |
| $140 | 4 | 202520 |
| $106 | 0 | 202532 |
| $100 | 2 | 202536 |

**Y distribution (Y > 0):**

| Percentile | Value |
|------------|-------|
| 25th | $15 |
| 50th | $22 |
| 75th | $40 |
| 90th | $65 |
| 99th | $210 |

**Transform regressions:**

| Transform | β(C) | SE | p-value |
|-----------|------|-----|---------|
| Log(1+Y) | -0.0080 | 0.0106 | 0.4478 |

### 9.5 Comparison Shopping

| Group | Mean Clicks | Mean Vendors | Click-to-Purchase Ratio |
|-------|-------------|--------------|-------------------------|
| Converters (151 users) | 113.35 | 83.84 | 72.8 |
| Non-converters (1,393 users) | 12.31 | 10.08 | - |

### 9.6 Conversion by Rank

| Rank | N | Conversions | Conv Rate | Total Spend |
|------|---|-------------|-----------|-------------|
| 1 | 3,053 | 33 | 1.08% | $1,244 |
| 2 | 2,769 | 21 | 0.76% | $752 |
| 3 | 1,814 | 21 | 1.16% | $523 |
| 4-5 | 3,289 | 23 | 0.70% | $743 |
| 6-10 | 6,423 | 42 | 0.65% | $1,648 |
| 11+ | 11,220 | 53 | 0.47% | $1,586 |

### 9.7 Attribution Lag

| Metric | Value |
|--------|-------|
| Min lag (hours) | 0.0 |
| Median lag (hours) | 2.7 |
| Mean lag (hours) | 103.2 |
| Max lag (hours) | 3,578.1 |
| Negative lags (ETL issue) | 0 |
| Cross-week purchases | 49 (20.7%) |

### 9.8 Cross-Week Journeys

| Metric | Value |
|--------|-------|
| Purchases in same week as click | 188 (79.3%) |
| Purchases in different week | 49 (20.7%) |

### 9.9 User-Vendor Pair Repeat Behavior

| Weeks per Pair | Count | % |
|----------------|-------|---|
| 1 week | 25,244 | 94.6% |
| 2 weeks | 1,172 | 4.4% |
| 3+ weeks | 283 | 1.1% |
| Max weeks | 11 | - |

### 9.10 Session Characteristics

| Gap | Sessions | Avg Clicks | Conv Rate |
|-----|----------|------------|-----------|
| 1d | 7,669 | 4.47 | 2.80% |
| 2d | 2,756 | 6.06 | 3.01% |
| 3d | 5,402 | 6.34 | 3.76% |
| 5d | 2,174 | 7.68 | 3.63% |
| 7d | 3,883 | 8.82 | 4.76% |

### 9.11 Lagged Click Effect

| Metric | Value |
|--------|-------|
| Corr(C_lag, Y) | 0.0182 |
| Avg Y after high-click week | $2.12 |
| Avg Y after low-click week | $1.39 |

### 9.12 Click Distribution

| C | Count | % |
|---|-------|---|
| 0 | 42 | 0.1% |
| 1 | 24,261 | 84.8% |
| 2 | 3,468 | 12.1% |
| 3 | 592 | 2.1% |
| 4 | 141 | 0.5% |
| 5+ | 115 | 0.4% |

### 9.13 Active Weeks

| Metric | Value |
|--------|-------|
| Total user-weeks | 5,734 |
| Cells with C = 0 | 42 (0.15%) |
| Mean clicks in C=0 user-weeks (other vendors) | 13.22 |
