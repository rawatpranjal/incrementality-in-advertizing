Here is the targeted EDA plan designed to validate assumptions and size the datasets before we commit to the heavy modeling computation.

### 1. Episode Definition Validation
*Goal: Confirm that the "48-hour gap" logic creates coherent shopping journeys rather than monolithic or fragmented noise.*

*   **Inter-Event Time Distribution:** Plot the log-distribution of time differences between consecutive events for a user.
    *   *Check:* Is there a natural "valley" in the distribution (e.g., at 24h or 48h) that justifies our cut-off?
*   **Episode Duration Stats:** Calculate mean, median, and 95th percentile duration of the generated episodes.
    *   *Risk:* If the median episode is 30 days, our logic is too loose. If it's 10 minutes, we are slicing the decision process too thin.
*   **Event Density:** Mean number of Impressions, Clicks, and Auctions per episode.
    *   *Sizing:* This dictates the memory requirements for the "Long Format" choice dataset.

### 2. Outcome Sparsity & "Zero-Inflation" Analysis
*Goal: Determine if we need Zero-Inflated Models (ZIP/Hurdle) or if OLS/Logit is sufficient.*

*   **The "Browser" Ratio:** What % of episodes have **zero** purchases?
    *   *Threshold:* If >90% are zero, we must use a Two-Stage model (Stage 1: Probability of Buy, Stage 2: Amount).
*   **The "Organic" Gap:** Of the episodes with a purchase, what % are purely organic (user saw ads but bought a non-promoted item)?
    *   *Relevance:* This defines the size of the "Organic" class in the Multinomial Logit. If 0%, the model collapses.
*   **GMV Distribution:** Histogram of `TOTAL_GMV` for non-zero episodes.
    *   *Check:* Is it log-normal? Extreme outliers here will skew the OLS elasticity results.

### 3. Vendor Competition & Overlap Density
*Goal: Validate that "Choice" actually exists. If users only see 1 vendor, there is no competition to model.*

*   **Share of Voice (SOV) Concentration:** Calculate the Gini coefficient of impressions *within* an episode.
    *   *Insight:* Do users see a "Monopoly" (1 vendor dominates) or "Perfect Competition" (10 vendors equally)?
*   **Unique Vendors per User:** Mean count of distinct `VENDOR_ID`s shown in an episode.
    *   *Sizing:* If Mean = 1.1, the Multinomial Logit is useless. If Mean > 3, the model is viable.
*   **The "Consideration Set" Size:** How many distinct items vs. distinct vendors are shown?

### 4. Counterfactual Validity (The "Loser" Analysis)
*Goal: Ensure the Vendor Fixed Effects model has enough variance.*

*   **Winner/Loser Intersection:** Calculate the Jaccard Index of Users for a Vendor.
    *   *Question:* Do vendors bid-and-lose on the *same* users they bid-and-win on?
    *   *Risk:* If a vendor *only* wins on User A and *only* loses on User B, the "Bid Count" control is confounded with user identity. We need within-user variation.
*   **Loss Ratios:** What % of a vendor's total auctions are losses (Rank > 1)?
    *   *Check:* If this is near 0%, we have no counterfactuals.

### 5. Semantic & Price Heterogeneity
*Goal: Verify controls for the regression models.*

*   **Price Variance within Episode:** Standard Deviation of `PRICE` shown to a user.
    *   *Check:* Does the user see a mix of cheap and expensive items? If variance is 0, `PRICE` cannot explain choice.
*   **Rank-Outcome Correlation:** Plot Conversion Rate by Ad Rank (1 to 10) *within* the episode.
    *   *Hypothesis:* We expect a sharp decay. If Rank 10 converts as well as Rank 1, the UX is non-standard or our ranking data is flawed.

### TLDR: The "Go/No-Go" Checks
1.  **If Episodes > 1 week:** Reduce gap threshold.
2.  **If 99% Zero GMV:** Switch to Hurdle Models immediately.
3.  **If 1 Vendor per User:** Cancel Multinomial Logit; stick to Binary Logit.
4.  **If No Losers:** Cancel Vendor iROAS; we cannot identify causality.