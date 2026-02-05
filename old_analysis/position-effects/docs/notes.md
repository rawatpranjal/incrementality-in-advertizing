# Econometric Methods for Position Effect Estimation

## 1. Position-Based Model (PBM) and Cascade Models

### Examination Probability Framework
The core identification problem: observed clicks conflate item relevance with position-driven examination probability.

**PBM Decomposition:**
```
P(click | item i, position k) = P(examine | position k) × P(click | examine, item i)
                               = γ_k × θ_i
```

Where:
- γ_k = examination probability at position k (decreasing in k)
- θ_i = item relevance/attractiveness

**EM Algorithm for PBM:**
1. E-step: Compute expected examination given observed click
   - If clicked: examined with probability 1
   - If not clicked: P(examined | no click) = γ_k(1 - θ_i) / (1 - γ_k × θ_i)
2. M-step: Update γ_k and θ_i using expected examination counts

**Cascade Model Extension (Craswell et al., 2008 WSDM):**
User examines positions sequentially; stops after clicking or abandoning.
```
P(examine position k) = ∏_{j<k} (1 - click_j) × (1 - abandon_j)
```

**Dynamic Bayesian Network (Chapelle & Zhang, 2009 WWW):**
Adds satisfaction state—user may click but continue if unsatisfied.
```
P(click_k | examine_k) = α_i
P(satisfied_k | click_k) = σ_i
P(examine_{k+1} | satisfied_k=0) = γ (continuation probability)
```

### Implementation Considerations
- Need sufficient position variation per item for identification
- Random position experiments ideal; otherwise leverage auction randomness
- Aggregate across users for stable γ_k estimates

---

## 2. Latent User Type Mixture Models

### Organic vs Sponsored Preference Heterogeneity
Users differ in propensity to engage with promoted content. Ignoring this heterogeneity biases position effect estimates.

**Two-Type Mixture Model:**
```
P(click | position k, sponsored=1) = π × P(click | type=organic-avoider, k)
                                    + (1-π) × P(click | type=neutral, k)
```

Where π is mixing proportion (latent).

**Extended Type Structure:**
- Type 1 (Ad-avoiders): Lower baseline CTR on promoted, steeper position decay
- Type 2 (Ad-neutral): Similar CTR organic/promoted
- Type 3 (Deal-seekers): Higher CTR on promoted at top positions

**Estimation Approach:**
1. EM algorithm over user types
2. Use organic click behavior (if observable) as type indicators
3. Allow type probabilities to vary with user covariates

**Identification Challenge:**
We only observe promoted impressions/clicks. Identification relies on:
- Within-user variation in position assignment
- Functional form assumptions on type-specific position effects
- Auxiliary data (session length, organic purchases) to inform type membership

---

## 3. IPW and Doubly Robust Estimators

### Propensity-Based Correction (Joachims et al., 2017)
When position assignment is non-random, weight observations by inverse propensity of position.

**IPW Estimator for Position k Effect:**
```
Ê[Y | do(position=k)] = (1/n) × Σ_i [Y_i × I(position_i = k) / P(position_i = k | X_i)]
```

**Propensity from Auction Mechanism:**
The auction provides natural propensities—ranking depends on bid × quality score.
```
P(position = k | item i, auction a) = f(bid_i, quality_i, competitor bids)
```

Estimate propensities using:
1. Multinomial logit on position given auction features
2. Machine learning on auction-level data (bid, quality, competing bids)

**Dual Learning for Propensities (Ai et al., 2018):**
Jointly learn:
- Relevance model: P(click | item, position)
- Propensity model: P(position | item, auction)

Avoids need for randomized position data.

**Doubly Robust Estimator:**
Combines outcome regression with IPW for robustness:
```
DR = (1/n) Σ_i [μ̂(k, X_i) + (Y_i - μ̂(position_i, X_i)) × I(position_i=k) / ê(k|X_i)]
```

Where μ̂ is outcome model, ê is propensity model. Consistent if either model correct.

---

## 4. Regression Discontinuity at Auction Score Cutoffs

### Narayanan & Kalyanam (2015, Marketing Science) Approach
Auction winners/losers determined by rank of bid × quality score. Sharp cutoff at winning threshold creates RDD opportunity.

**Setup:**
- Running variable: Auction score (bid × quality) relative to cutoff
- Treatment: Winning impression slot (vs. not shown)
- Outcome: Click, conversion, or purchase

**Identification:**
Items just above/below winning threshold are similar on observables by construction. Treatment effect identified at the cutoff.

**Position-Specific RDD:**
For position k effect, compare:
- Items ranked exactly k (just made cutoff for position k)
- Items ranked k+1 (just missed position k)

**Implementation:**
```
Y_i = α + τ × I(rank_i ≤ k) + f(score_i - cutoff_k) + ε_i
```

Local linear regression or polynomial in running variable.

**Practical Considerations:**
- Need sufficient density near cutoffs
- Manipulation tests: Is there bunching at cutoffs?
- Bandwidth selection: Imbens-Kalyanaraman optimal bandwidth
- Check covariate smoothness across cutoff

---

## 5. Discrete-Time Hazard Models with Competing Risks

### Session-Level Click/Exit Modeling
Model user session as sequence of position exposures with click or exit at each step.

**Hazard Framework:**
```
h_k = P(event at position k | survived to position k)
```

**Competing Risks:**
At each position, user can:
1. Click promoted item (λ_click,k)
2. Click organic item (λ_organic,k) — unobserved in our data
3. Exit session (λ_exit,k)

**Discrete-Time Logit:**
```
logit(h_k) = α + β × position_k + γ × item_features + θ × user_features
```

**Frailty Model for User Heterogeneity:**
```
h_{ik} = h_0(k) × exp(X_i'β + u_i)
```

Where u_i is user-specific random effect (e.g., Gamma distributed).

**Identification Challenge:**
We observe promoted clicks only. Organic clicks and exits are competing risks that censor our outcome.

**Partial Solution:**
Use session-level purchase data to distinguish:
- Sessions ending in promoted purchase (clicked promoted)
- Sessions ending in organic purchase (likely clicked organic)
- Sessions with no purchase (exit or organic click without conversion)

---

## 6. Partial Identification and Bounds

### Manski-Style Bounds
When key variables (organic behavior) are unobserved, provide bounds on treatment effects.

**Setup:**
- Want: E[Y(1) - Y(0)] (effect of position 1 vs not shown)
- Observe: Only promoted impressions/clicks
- Missing: Organic engagement, counterfactual if not shown

**Worst-Case Bounds:**
```
Lower bound: E[Y | position=1] - 1  (assume all non-shown would click)
Upper bound: E[Y | position=1] - 0  (assume none non-shown would click)
```

**Monotonicity Refinement:**
Assume showing an ad can only increase click probability:
```
P(click | shown) ≥ P(click | not shown)
```

Tightens upper bound.

**Instrumental Variable Bounds:**
Use auction score as instrument. Bounds on LATE (local average treatment effect) for compliers.

---

## Key References

1. **Craswell, N., Zoeter, O., Taylor, M., & Ramsey, B. (2008).** An experimental comparison of click position-bias models. *WSDM*. — Cascade model for position effects.

2. **Chapelle, O., & Zhang, Y. (2009).** A dynamic Bayesian network click model for web search ranking. *WWW*. — DBN with satisfaction modeling.

3. **Joachims, T., Swaminathan, A., & Schnabel, T. (2017).** Unbiased learning-to-rank with biased feedback. *WSDM*. — IPW estimators for position bias.

4. **Ai, Q., Bi, K., Luo, C., Guo, J., & Croft, W. B. (2018).** Unbiased learning to rank with unbiased propensity estimation. *SIGIR*. — Dual learning approach.

5. **Narayanan, S., & Kalyanam, K. (2015).** Position effects in search advertising and their moderators: A regression discontinuity approach. *Marketing Science*. — RDD at auction cutoffs.

6. **Yang, S., & Ghose, A. (2010).** Analyzing the relationship between organic and sponsored search advertising: Positive, negative, or zero interdependence? *Marketing Science*. — Organic-sponsored interaction effects.

7. **Manski, C. F. (2003).** *Partial Identification of Probability Distributions*. Springer. — Bounds approach.

8. **Agarwal, A., Hosanagar, K., & Smith, M. D. (2011).** Location, location, location: An analysis of profitability of position in online advertising markets. *Journal of Marketing Research*. — Position value in auctions.

---

## Data Requirements Checklist

For implementing these methods, verify:

- [x] Q1: Are impression timestamps unique within auction, or batched? → **Batched** (60% zero-gaps)
- [x] Q2: What is the distribution of positions per auction? → **Median 7 impressions/auction**
- [x] Q3: What is the maximum rank that ever receives an impression? → **Rank 38+, gradual decline**
- [x] Q4: Do products appear at multiple positions across auctions? → **Yes, 87% have rank variation**
- [x] Q5: Do users with more auctions show more rank variation for same products? → **Mean range 8.6 ranks**
- [x] Q6: What is typical time between auctions for same user (session definition)? → **Median 18s between clicks**

---

## SUMMARY OF FACTS FOR OBSERVATIONAL STUDY

---

### THE SETTING

**Platform:** Poshmark or similar fashion resale marketplace
**Auction type:** Sponsored product ads, second-price-ish with quality weighting
**Placements:** 4 types (search results, carousels, PDP, homepage)

---

### DATA AVAILABLE (15-min pull, 1% users)

| Table | Count | Key Fields |
|-------|-------|------------|
| Auctions | 78,318 | placement, user_id, timestamp |
| Bids | 3,748,381 | ranking, is_winner, quality, final_bid, pacing, price |
| Impressions | 192,307 | auction_id, product_id, timestamp |
| Clicks | 6,090 | auction_id, product_id, timestamp |
| Catalog | 1,039,783 | brand, color, size, price, description |

---

### CONFIRMED MECHANICS

| Fact | Evidence | Confidence |
|------|----------|------------|
| **RANKING = order by (QUALITY × FINAL_BID) desc** | 85.75% exact match within auction, Rank-1 has max score 99% | ✅ HIGH |
| **QUALITY ≈ predicted CTR** | AUC 0.68, calibration correlation 0.81 | ✅ HIGH |
| **PACING does not affect ranking** | Adding PACING hurts R² (0.57 → 0.50) | ✅ HIGH |
| **FINAL_BID ≠ CVR × Price** | R² = -1.06, bids are manually set | ✅ HIGH |

---

### THE FUNNEL

```
48 bids/auction → 38 winners (79%) → 7 impressions (18% of winners) → 0.03 clicks (3% CTR)
```

| Stage | Mechanism | What We Know |
|-------|-----------|--------------|
| Bid → Winner | Top ~38 by score | Deterministic, IS_WINNER flag |
| Winner → Impression | **UNKNOWN** | Gradual decline by rank (not top-N, not random) |
| Impression → Click | User decision | CTR ~3%, non-monotonic in rank |

**Critical unknown:** How are 7 of 38 winners selected for impression?

---

### DISPLAY POSITION VS BID RANK

| Metric | Value | Implication |
|--------|-------|-------------|
| Correlation | 0.87 | Related but not identical |
| Exact match | 26% | Usually different |
| MAE | 3.7 ranks | Substantial reshuffling |
| Timestamp reliability | 60% zero-gaps | Batch logging, can't infer display order |

**Conclusion:** RANKING (bid rank) ≠ display position. We cannot recover true display position from timestamps.

---

### SELECTION INTO IMPRESSION

| Rank | Winners | Impressed | Rate |
|------|---------|-----------|------|
| 1 | 78K | 20K | 26% |
| 5 | 76K | 8K | 11% |
| 10 | 74K | 6K | 8% |
| 20 | 72K | 3K | 4% |
| 38 | ~70K | ~1K | ~1.5% |

**Pattern:** Gradual decline, not sharp cutoff. Monotonicity violated at ranks 9-10, 11-12.

---

### CTR BY RANK (IMPRESSED ONLY)

| Rank | Impressions | Clicks | CTR |
|------|-------------|--------|-----|
| 1 | 20,718 | 605 | 2.92% |
| 2 | 20,494 | 550 | 2.68% |
| 3 | 11,380 | 310 | 2.72% |
| 4 | 11,023 | 347 | **3.15%** |
| 5 | 8,256 | 248 | 3.00% |
| 10 | 5,773 | 174 | 3.01% |

**Anomaly:** CTR is NOT monotonically decreasing. Rank 4 > Rank 3 > Rank 2.

---

### RANK 4 ANOMALY

| Placement | Rank 3 CTR | Rank 4 CTR | Anomaly? |
|-----------|------------|------------|----------|
| 1 (search?) | 2.47% | 3.07% | ✅ YES |
| 2 | 3.53% | 4.14% | ✅ YES |
| 3 (feed?) | 2.66% | 2.57% | ❌ No |
| 5 | 1.14% | 1.54% | ❌ No |

**Hypothesis:** UI layout differs by placement. Rank 4 may be a visually prominent slot in some placements.

---

### MULTI-CLICK BEHAVIOR

| Metric | Value |
|--------|-------|
| Auctions with 2+ clicks | 32.7% |
| Different products clicked | 84.2% |
| Median time between clicks | 18 seconds |
| Potential duplicates (<1s) | 1.1% |

**Implication:** Users comparison shop. Cascade model (click = terminate) is violated. Cannot use standard survival/hazard.

---

### PLACEMENT HETEROGENEITY

| Placement | Volume | Rank 1 Imp Rate | Median Imps/Auction | Character |
|-----------|--------|-----------------|---------------------|-----------|
| 3 | 62% | 11% | Low | Low-prominence feed |
| 1 | 14% | 74% | High | High-prominence search |
| 5 | 14% | 4% | Very low | Minimal exposure |
| 2 | 11% | ~50% | Medium | Moderate prominence |

**Implication:** Position effects likely differ dramatically by placement. Should stratify or analyze separately.

---

### WHAT CAUSAL METHODS ARE FEASIBLE

| Method | Requirement | Status |
|--------|-------------|--------|
| **RDD** | Sharp score discontinuity | ✅ 80% gaps < 0.01, median gap 0.0004 |
| **PBM** | Same ad at different positions | ✅ 87% of user-product pairs have rank variation |
| **IV (N competitors)** | First stage strength | ✅ Testable, plausible |
| **Within-auction FE** | Rank variation within auction | ✅ By construction |
| **Survival/Cascade** | Sequential examination, single termination | ❌ 32% multi-click violates |
| **IPW** | Propensity overlap | ⚠️ 73% overlap but ESS only 14% |

---

### WHAT WENT WRONG IN FIRST ANALYSIS

| Model | Error |
|-------|-------|
| **FE** | Demeaned at product level, not auction level |
| **Control Function** | Stage 1 R²=3.7% when true R²=85%; predicted absolute rank, not within-auction |
| **IPW** | Propensity ignored competition; P(rank\|bid,quality) is wrong |
| **Survival** | Aggregated to auction level, lost ad-position variation |

**Result:** All models showed CTR *increasing* with rank (backwards), confirming severe selection bias.

---

### CORE IDENTIFICATION CHALLENGE

```
Observed CTR(k) = θ(k) × α(k) × S(k)
                    ↑       ↑       ↑
              position  selection  survival
               effect   into imp   to rank k
```

We observe the product. We want θ(k).

**Selection confounds:**
1. Products at high ranks have higher QUALITY (by construction)
2. Products at high ranks in low-competition auctions get more impressions
3. Users who scroll deep may have higher intent

---

### WHAT WE CAN CREDIBLY ESTIMATE

| Estimand | Identification | Assumption |
|----------|----------------|------------|
| **LATE at rank margins** | RDD | Continuity at score cutoff |
| **Average position bias θ_k** | PBM | Separability of position and relevance |
| **Bounds on position effect** | Partial ID | Monotonicity (optional) |
| **Position effect conditional on auction** | Within-auction FE | No time-varying confounds within auction |

---

### KEY NUMBERS TO REMEMBER

| Metric | Value |
|--------|-------|
| Overall CTR (impressed) | 2.8% |
| Rank 1 CTR | 2.92% |
| Rank 10 CTR | 3.01% |
| Winner → Impression rate | 6.2% |
| Rank 1 → Impression rate | 26% |
| Score gap at boundary (median) | 0.0004 |
| User-product pairs with rank variation | 847K (87%) |
| Mean rank range per user-product | 8.6 |

---

### RECOMMENDED PATH FORWARD

1. **RDD at adjacent rank margins** — cleanest, exploits score discontinuity
2. **PBM via EM** — classic, uses within-ad variation
3. **Within-auction FE** — simple, proper specification
4. **Stratify by placement** — heterogeneity is real
5. **Bounds for robustness** — what's knowable under minimal assumptions
