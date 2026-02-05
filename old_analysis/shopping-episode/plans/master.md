# Master Plan: Shopping Episode Incrementality Modeling

## Objective
To quantify the causal impact of sponsored ad interactions (Impressions and Clicks) on purchase behavior at the **Shopping Episode** level. This analysis moves beyond "Last Click Attribution" to estimate the marginal utility of ads for both the Platform (Marketplace GMV) and the Individual Seller (Vendor GMV), while explicitly modeling the competitive substitution effects.

---

## 1. The Unit of Analysis: "The Shopping Episode"

We reject the standard 30-minute web session as too short for high-consideration commerce and the 7-day window as too noisy.

*   **Definition:** An `EPISODE` is a contiguous sequence of user activity (Impressions, Clicks, Purchases) belonging to a single `USER_ID`.
*   **Boundary Condition:** A new episode begins after a **48-hour period of inactivity**.
*   **Universe:** The analysis is restricted to episodes containing **at least one Sponsored Impression**. (Purely organic sessions are invisible to the ad-server logs and are excluded to prevent zero-inflation bias).

---

## 2. Data Engineering Strategy

The script `05_incrementality_modeling.py` will generate three distinct analytical datasets from the raw logs.

### A. The Episode Map
*   **Process:** Sort all timestamps by User. Calculate `Time_Diff`. Cumulative Sum where `Time_Diff > 48h` to assign unique `EPISODE_ID`.

### B. Dataset 1: Marketplace Aggregate (`df_market`)
*   **Goal:** Global Elasticity Analysis.
*   **Grain:** One row per `EPISODE_ID`.
*   **Features:**
    *   `TOTAL_GMV`: Sum of all purchases (Promoted + Organic).
    *   `TOTAL_CLICKS`: Sum of sponsored clicks.
    *   `TOTAL_IMPRESSIONS`: Sum of sponsored impressions.
    *   `DURATION_HOURS`: Time between first and last event.
    *   `AD_INTENSITY`: Impressions per hour.
    *   `PRICE_TIER`: Average Price of items shown (Proxy for user budget).

### C. Dataset 2: Vendor Counterfactuals (`df_vendor`)
*   **Goal:** Seller-Level iROAS.
*   **Grain:** One row per `(EPISODE_ID, VENDOR_ID)`.
*   **Inclusion Logic:** A record exists IF:
    1.  Vendor **Bid** on this user (regardless of win/loss).
    2.  OR User **Purchased** from this Vendor.
*   **Features:**
    *   `VENDOR_GMV`: Spend on this specific vendor.
    *   `VENDOR_CLICKS`: Clicks on this specific vendor.
    *   `IS_WINNER`: Binary (Did they win $\ge 1$ impression?).
    *   `BID_COUNT`: Intensity of vendor's attempt to reach user.
    *   `COMPETITOR_CLICKS`: Clicks on *other* vendors in the same episode.

### D. Dataset 3: The Choice Menu (`df_choice`)
*   **Goal:** Competitive Substitution Modeling.
*   **Grain:** One row per `(EPISODE_ID, OPTION_ID)`.
*   **Structure:** Long-format "Menu" of options available to the user.
*   **Classes:**
    1.  `No_Buy`: Outside option (always present).
    2.  `Organic`: Purchase of non-promoted item (always present).
    3.  `Focal_Vendor`: The promoted vendor with highest Share of Voice.
    4.  `Rival_Vendor`: The promoted vendor with 2nd highest Share of Voice.

---

## 3. Modeling Specifications

### Model I: Marketplace Elasticity (OLS)
*   **Question:** Does increasing ad volume grow the total pie?
*   **Equation:**
    $$ \log(\text{GMV}+1) = \alpha + \beta_1 \log(\text{Clicks}+1) + \beta_2 \log(\text{Duration}) + \beta_3 \text{PriceTier} + \epsilon $$
*   **Hypothesis:** $\beta_1$ represents the % increase in total spend for a 1% increase in clicks. If $\beta_1 \approx 0$, ads are cannibalistic.

### Model II: Vendor iROAS (Fixed Effects)
*   **Question:** Does a vendor paying for a click generate *incremental* revenue compared to just bidding and losing?
*   **Equation:**
    $$ Y_{gmv} = \alpha + \beta_1 (\text{Clicks}) + \beta_2 (\text{CompetitorClicks}) + \beta_3 (\text{BidCount}) + \mu_{vendor} + \epsilon $$
*   **Metric:** $\text{iROAS} = \frac{\beta_1}{\text{Avg CPC}}$.
*   **Control:** `BidCount` controls for the vendor's intent; `CompetitorClicks` controls for share-of-wallet competition.

### Model III: Multinomial Logit (Competition)
*   **Question:** How does Ad Intensity shift the probability of choosing Vendor A vs. Vendor B vs. Walking Away?
*   **Equation:**
    $$ P(\text{Choice}=j) = \frac{\exp(\alpha_j + \beta_{imp}I_j + \beta_{click}C_j)}{\sum \exp(\dots)} $$
*   **Insight:** Explicitly models the "Zero Sum" nature of the session. If Ad A's probability rises, the probability of "No Buy" (Market Expansion) and "Competitor Buy" (Conquesting) must adjust.

---

## 4. Execution Plan
1.  **Script:** Single file `05_incrementality_modeling.py`.
2.  **Output:** Single file `05_incrementality_results.txt`.
3.  **Contents:**
    *   Data Quality Checks (Session lengths, Match rates).
    *   Model I Summary (Coefficients, R2, Elasticity).
    *   Model II Summary (Vendor Lift, Calc iROAS).
    *   Model III Summary (Logit Coefficients, Odds Ratios).
    *   **Recommendation Block:** Synthesized view on whether ads create value or extract rent.