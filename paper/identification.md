# Identification Strategies

This document explains the causal identification strategy for each analysis in the paper.

---

## 1. Fold Effects (Section 7.4, near_tie_identification.tex)

### Causal Question

What is the causal effect of rank position on ad visibility and clicks?

### The Selection Problem

Higher-ranked ads get more clicks. But position is determined by an auction where ads are ranked by a score:
$$\text{AdRank}_i = \text{bid}_i \times \text{QualityScore}_i$$

Ads with higher AdRank get higher positions. This creates confounding: ads in position 1 have higher AdRank than ads in position 2, but ads with higher AdRank may also be intrinsically more clickable. A naive comparison conflates the position effect with the selection effect.

As Narayanan and Kalyanam (2015, p. 388) note: "A simple mean comparison of outcomes at two positions is likely to be biased due to these selection issues."

### Why Other Approaches Fail

Narayanan and Kalyanam (2015, Section 3) discuss three approaches and their limitations:

1. **Experimentation**: Randomizing the focal advertiser's bids does not eliminate bias induced by competitors' strategic bidding behavior. "This spurious correlation cannot be eliminated by randomizing the bids of the focal advertiser alone" (p. 395).

2. **Instrumental variables**: Hard to find instruments that are correlated with position but can be excluded from consumer outcomes. "Instruments are difficult if not impossible to find in this context; demand side factors that are correlated with position cannot typically be excluded from consumer outcome variables" (p. 389).

3. **Parametric selection models**: Require correct functional form for complex auction mechanisms. "Because position is determined through a set of highly complex processes (Jerath et al. 2011), the nature of the selection effects could vary by position with unpredictable signs" (p. 395). Furthermore, "the selection bias is likely to be highly local" with different mechanisms "operating at different times inducing biases of opposite signs" (p. 395).

### The RD Solution (Narayanan and Kalyanam 2015)

The regression discontinuity design measures causal treatment effects when treatment is based on whether an underlying continuous forcing variable crosses a threshold. The key insight is comparing outcomes when the advertiser "just barely won" versus "just barely lost" the bid.

**Treatment**: Being in position $i$ versus position $i+1$.

**Forcing variable**: The difference in AdRank between adjacent bidders. As described in Narayanan and Kalyanam (2015, p. 396, eq. 5-6), if bidder A gets position $i$ and bidder B gets position $i+1$, it must be the case that:
$$\text{AdRank}_i > \text{AdRank}_{i+1} \tag{5}$$

or in other words:
$$\Delta\text{AdRank}_i \equiv (\text{AdRank}_i - \text{AdRank}_{i+1}) > 0 \tag{6}$$

The forcing variable for the RD design is this difference in AdRank, and the threshold for treatment is 0.

**RD estimator** (p. 396, eq. 2-4): Let $y$ denote the outcome of interest, $z$ the forcing variable, and $\bar{z}$ the threshold above which there is treatment. Define the two limiting values of the outcome variable as:
$$y^+ = \lim_{\lambda \to 0} \mathbb{E}[y \mid z = \bar{z} + \lambda] \tag{2}$$
$$y^- = \lim_{\lambda \to 0} \mathbb{E}[y \mid z = \bar{z} - \lambda] \tag{3}$$

Then the local average treatment effect is:
$$d = y^+ - y^- \tag{4}$$

The RD design "compares situations when the advertiser just barely won the bid to situations when the advertiser just barely lost the bid" (p. 396). This achieves the quasi-experimental design that underlies RD.

### Why This Is "Apples to Apples"

Narayanan and Kalyanam (2015, p. 397) state: "Occasions when the advertiser barely won the bid and when he barely lost the bid can be considered equivalent in terms of underlying propensities for click-throughs, sales, etc. Any difference between the limiting values of the outcomes on the two sides of the threshold can be entirely attributed to the position."

Our design achieves this because:
1. **Same auction**: Both ads faced identical user, query, and competitive context
2. **Same score neighborhood**: $s_{\text{hi}} \approx s_{\text{lo}}$ means "ad appeal" is matched
3. **Only rank differs**: The sole source of variation is the discrete position assignment

### Our Implementation

**Score**: $s_i = \text{QUALITY}_i \times \text{FINAL\_BID}_i$ (analogous to AdRank)

**Running variable**: Normalized score gap within pair
$$z = \frac{s_{\text{hi}} - s_{\text{lo}}}{s_{\text{hi}}}$$

This is a normalized version of $\Delta\text{AdRank}$.

**Bandwidth $\tau$**: Include pairs where $z \leq \tau$ (e.g., $\tau = 0.02$ means scores within 2%).

**Lucky/Unlucky assignment**:
- Lucky = higher score (gets better rank within pair)
- Unlucky = lower score (gets worse rank within pair)
- When $z \approx 0$, assignment is as-if random

### Local Linear Regression

Following Narayanan and Kalyanam (2015, p. 397, eq. 8), the local linear regression is:
$$y_j = \alpha + \beta \cdot \mathbf{1}(\text{pos}_j = i+1) + \gamma_1 \cdot z_j + \gamma_2 \cdot z_j \cdot \mathbf{1}(\text{pos}_j = i+1) + f(j;\theta) + \varepsilon_j \tag{8}$$

where:
- $\mathbf{1}(\text{pos}_j = i+1)$ = indicator for the higher position (lower rank number)
- $\beta$ = **position effect of interest**
- $\gamma_1, \gamma_2$ = control for variation in outcome with the forcing variable, allowing the variation to be different on the two sides of the threshold through an interaction effect
- $f(j;\theta)$ = fixed effects (firm, keyword, match type, day of week)

The bandwidth $\lambda$ is selected using leave-one-out cross validation (LOOCV) to minimize the mean squared error of predictions.

### Validity Conditions

**Continuity of forcing variable** (p. 396): "A necessary condition for the validity of the RD design is that the forcing variable itself is continuous at the threshold." This is achieved in the typical marketing context if the agents are uncertain about the score or the threshold.

**No manipulation**: In Narayanan and Kalyanam (2015), the modified second-price auction mechanism used by Google ensures that "the second-price auction design eliminates incentives for advertisers to second guess their competitors bids and put in their own bids so as to have an AdRank just above the threshold" (p. 396).

**Our setting is even stronger because of autobidding**: In our context (Topsort/Poshmark), the ad-tech platform performs autobidding on behalf of advertisers. The bid formula is:
$$\text{FINAL\_BID} = \frac{\text{pCVR} \times \text{AOV}}{\text{target\_ROAS}}$$

where pCVR is predicted by the platform's ML models and AOV is approximately equal to PRICE. This means:
- **Advertisers do not control bids**: The platform determines bids algorithmically
- **No strategic manipulation possible**: Vendors cannot target being just above a threshold
- **Score variation is driven by ML predictions**: pCTR (QUALITY) and pCVR vary continuously based on product/user features

This is a stronger validity condition than Narayanan and Kalyanam's second-price auction because there is literally no advertiser strategic behavior to worry about. The score variation among near-ties is driven by small differences in ML predictions, which are as-if random conditional on observables.

### Diagnostics

1. **Balance test**: Compare QUALITY and FINAL_BID between lucky/unlucky within pairs. Define standardized difference:
$$d_Q = \frac{\bar{Q}_{\text{lucky}} - \bar{Q}_{\text{unlucky}}}{\sqrt{(\text{Var}(Q_{\text{lucky}}) + \text{Var}(Q_{\text{unlucky}}))/2}}$$
Criterion: $|d_Q| < 0.1$ and $|d_B| < 0.1$. If balanced, local randomization is credible.

2. **Density test**: Check for bunching at the threshold (McCrary-style). Smooth, monotonically decreasing density in $z$ suggests no manipulation.

3. **Rank-score alignment**: Verify platform actually ranks by score. Alignment is measured as:
$$\text{Alignment} = \frac{1}{N}\sum_{\text{pairs}} \mathbf{1}[\text{rank}_{\text{lucky}} < \text{rank}_{\text{unlucky}}]$$
Lower alignment at tight ties may indicate tie-breaking rules.

4. **Placebo boundary**: Test rank 7 vs 8 (below any reasonable fold). Expected: $\Delta_{\text{exposure}} \approx 0$, $\Delta_{\text{ctr}} \approx 0$, since both positions are off-screen.

### Two-Stage Decomposition (Our Extension)

We extend Narayanan and Kalyanam by decomposing the position effect into:

**Stage 1 (Extensive margin)**: Visibility effect
$$\Delta_{\text{exposure}} = P(\text{impressed} \mid \text{lucky}) - P(\text{impressed} \mid \text{unlucky})$$

**Stage 2 (Intensive margin)**: Conditional click effect
$$\Delta_{\text{ctr}} = P(\text{clicked} \mid \text{lucky}, \text{both exposed}) - P(\text{clicked} \mid \text{unlucky}, \text{both exposed})$$

This separates:
- **Visibility effect**: Higher rank leads to being shown to the user (scroll/fold mechanism)
- **Persuasion effect**: Higher rank leads to more clicks conditional on being seen

### Key Finding

At the fold (rank 2 vs 3) with $\tau = 0.02$: $\Delta_{\text{exposure}} \approx 0.21$, $\Delta_{\text{ctr}} \approx 0.00$.

Interpretation: Position affects visibility, not clicking conditional on visibility. The fold is a visibility threshold, not a persuasion gradient. This is consistent with Narayanan and Kalyanam's finding that "there seem to be significant effects when moving from positions 6 to 5 and 7 to 6. These positions are typically below the page fold and often require consumers to scroll down" (p. 399).

### Methodological Parallel: Cohen et al. (2016) Uber Consumer Surplus

Cohen et al. (2016) use an RD design that is structurally identical to ours. In their setting, Uber calculates a continuous "surge generator" but presents consumers with discrete surge prices (1.2x, 1.3x, 1.4x, etc.).

**Key quote (p. 7)**: "Two customers who have nearly identical surge generators (i.e., face nearly identical market conditions), but who happen to be on opposite sides of a pre-defined cut-off, face discretely different surge prices."

**Key quote (p. 9)**: "discrete pricing leads to discontinuous jumps in prices for sessions with arbitrarily small differences in the underlying demand and supply conditions"

| Aspect | Cohen et al. (Uber) | Narayanan & Kalyanam | Our Fold Effects |
|--------|---------------------|----------------------|------------------|
| Continuous variable | Surge generator (e.g., 1.249x) | AdRank = bid $\times$ QualityScore | Score = QUALITY $\times$ FINAL_BID |
| Discretization rule | Round to 1.2x, 1.3x, 1.4x | Position $i$ vs $i+1$ | Rank 1, 2, 3 |
| Near-tie example | Generator 1.249 vs 1.251 | AdRank just above/below threshold | Score gap $\leq$ 2% |
| Discrete treatment | 1.2x vs 1.3x surge price | Position $i$ vs Position $i+1$ | Position 2 vs Position 3 |
| Outcome | Purchase rate | CTR, sales | Impression, Click |
| No manipulation | Algorithmic pricing | Second-price auction | **Autobidding** |

The parallel is exact: discrete ranking leads to discontinuous position assignments for ads with arbitrarily small differences in scores, just as discrete pricing leads to discontinuous price jumps for sessions with arbitrarily small differences in the surge generator.

### Source Code Reference

R script: `analysis/position-effects-analysis-R/scripts/06_uber_near_tie_rdd_eda.R`
Result file: `analysis/position-effects-analysis-R/results/uber_near_tie_rdd_eda_round2_pl1.txt`

---

## 2. Position Bias (Section 7.1, position_bias.tex)

### Causal Question

What is the marginal effect of rank position on click probability, controlling for ad quality?

### Data Structure

- **Unit of analysis**: Impression $i$ (a winning bid that was rendered to user)
- **Population**: 71,194 impressions in a 15-minute window
- **Outcome**: $\text{clicked}_i \in \{0,1\}$

### Model Specification

$$P(\text{clicked}_i = 1) = \Lambda\left(\beta_1 \cdot \text{quality}_i + \beta_2 \cdot \text{rank}_i + \beta_3 \cdot \text{price}_i + \beta_4 \cdot \text{cvr}_i + \alpha_v + \gamma_p\right)$$

where:
- $\Lambda(\cdot)$ = logistic function
- $\text{quality}_i$ = platform predicted CTR (pCTR), the quality score used in ranking
- $\text{rank}_i$ = ad position within auction (1 = highest)
- $\text{price}_i$ = product price in cents
- $\text{cvr}_i$ = platform predicted conversion rate
- $\alpha_v$ = vendor fixed effect (absorbs 4,742 vendors)
- $\gamma_p$ = placement fixed effect (absorbs 4 placements)
- Standard errors clustered at auction level

### Identification Strategy: Selection on Observables

The platform ranks ads by $\text{score} = \text{quality} \times \text{bid}$. If we observe the quality score, we can control for the "goodness" of the ad. Any remaining rank effect, after controlling for quality, captures the causal position effect.

### Why Each Control is Needed

| Control | Why needed | What it absorbs |
|---------|------------|-----------------|
| quality | Directly controls for pCTR used in ranking | Ad relevance/appeal to this user |
| rank | Treatment of interest | Position on page |
| price | User price sensitivity | Higher prices may deter clicks |
| cvr | Predicted conversion rate | May affect user intent to click |
| $\alpha_v$ (vendor FE) | Time-invariant vendor heterogeneity | Brand reputation, product quality |
| $\gamma_p$ (placement FE) | Placement-specific click propensity | Search vs brand vs product page behavior |
| Auction clustering | Within-auction correlation | Same user, same moment, same competitive context |

### Why "Apples to Apples"?

This is a within-vendor, within-placement comparison that controls for the observable quality signal. Two impressions from the same vendor, on the same placement type, with similar quality scores, but at different ranks, are being compared.

### Assumptions

1. **Selection on observables**: Conditional on quality, price, cvr, vendor FE, and placement FE, the residual variation in rank is uncorrelated with unobserved determinants of clicks
2. **Quality score is sufficient statistic**: The platform pCTR captures all ad-specific appeal relevant to clicking
3. **No unmeasured confounding**: Within-auction competitive context does not affect click propensity beyond rank

### Limitations

This is a weaker identification than the near-tie RD design because:
- Rank is still correlated with unobserved auction-level competition
- Quality may not fully capture ad appeal
- Selection into impressions is endogenous (only winning bids are observed)

As Narayanan and Kalyanam (2015, p. 399-400) show, OLS and fixed effects estimates are generally "highly positively biased" compared to RD estimates, with "the magnitude of the bias quite significant and varies by position."

### Comparison of Identification Strength

| Design | Identifies | Assumption | Threat |
|--------|-----------|------------|--------|
| Near-Tie RD | Local causal effect at threshold | Continuity of forcing variable | Manipulation |
| Selection on Observables | Average treatment effect | No unmeasured confounders | Omitted variables |

The RD design is stronger because it only requires continuity of potential outcomes at the threshold, while selection-on-observables requires that all confounders are observed and correctly specified.

### Key Results

- $\hat{\beta}_{\text{quality}} = 0.899$ (positive: higher pCTR predicts clicks)
- $\hat{\beta}_{\text{rank}} = -0.010$ (negative: higher rank number means fewer clicks)
- $\hat{\beta}_{\text{price}} = -0.016$ (negative: higher price deters clicks)
- $\hat{\beta}_{\text{cvr}} = 3.453$ (positive: higher pCVR predicts clicks)

### Source Code Reference

R script: `analysis/position-effects-analysis-R/scripts/01_position_bias_felogit.R`
Result file: `analysis/position-effects-analysis-R/results/position_bias_felogit_fixest_round2.txt`

---

## 3. Shopping Sessions (Section 6, funnel_analysis_v2.tex)

*To be added.*

The shopping sessions analysis uses a session-level fixed effects model to estimate the average treatment effect of an ad click on purchase probability, controlling for user and session heterogeneity.

---

## 4. Vendor Panels (Section 5, vendor_week.tex)

*To be added.*

The vendor panel analysis uses a staggered difference-in-differences design (Callaway and Sant'Anna) to estimate the causal effect of campaign adoption on vendor-level outcomes.

---

## Summary: Comparison of Identification Strategies

| Analysis | Design | Unit | Comparison | Strength |
|----------|--------|------|------------|----------|
| Fold Effects | Near-tie RD | Bid pair | Within-auction, within-score | Strong (local randomization) |
| Position Bias | Selection on observables | Impression | Within-vendor, within-placement | Moderate (relies on quality control) |
| Shopping Sessions | Session FE | Session-product | Within-session | Moderate (selection into click) |
| Vendor Panels | Staggered DiD | Vendor-week | Across time, across vendors | Moderate (parallel trends) |

The fold effects design provides the cleanest identification because it exploits the discrete nature of ranking among nearly-tied competitors within the same auction. The other designs rely on stronger functional form or parallel trends assumptions.

---

## References

Cohen, P., Hahn, R., Hall, J., Levitt, S., and Metcalfe, R. (2016). "Using Big Data to Estimate Consumer Surplus: The Case of Uber." NBER Working Paper 22627.

Narayanan, S. and Kalyanam, K. (2015). "Position Effects in Search Advertising and their Moderators: A Regression Discontinuity Approach." *Marketing Science*, 34(3):388-407.
