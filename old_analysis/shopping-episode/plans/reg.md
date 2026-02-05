This is the rigorous statistical framework for the analysis. We will build two distinct datasets for two distinct regressions.

### 1. Marketplace Level Analysis
**Question:** Does driving more clicks globally increase the total Gross Merchandise Value (GMV) of the platform, or is it just cannibalizing organic sales?

#### The Dataset: `df_marketplace`
*   **Unit of Analysis (Row):** A single User-Shopping Episode.
*   **Definition:** A contiguous period of activity by one user, broken by a 48-hour gap.

| Type | Column Name | Definition | Hypothesis / Purpose |
| :--- | :--- | :--- | :--- |
| **ID** | `EPISODE_ID` | Unique hash of User + Start Time | Primary Key |
| **Outcome ($Y$)** | `TOTAL_GMV` | Sum of `PRICE * QTY` for *all* purchases (promoted + organic) in episode. | We want to see if this moves. |
| **Treatment ($T$)** | `TOTAL_CLICKS` | Count of sponsored clicks in episode. | Does engagement drive lift? |
| | `TOTAL_IMPRESSIONS` | Count of sponsored impressions in episode. | Exposure effect (Brand awareness). |
| **Control ($X$)** | `DURATION_HOURS` | Time between first and last event. | **Crucial:** Longer sessions have more clicks *and* more spend naturally. |
| | `INTENSITY_SCORE` | (Impressions / Duration). | High frequency of ads per hour. |
| | `IS_CONVERTER` | Binary (Did they buy anything?). | To separate "browsers" from "buyers" (Hurdle model). |

#### The Regression Model (OLS)
$$ \log(\text{TOTAL\_GMV} + 1) = \alpha + \beta_1 \log(\text{TOTAL\_CLICKS} + 1) + \beta_2 \log(\text{DURATION\_HOURS}) + \epsilon $$

*   **Interpretation:**
    *   If $\beta_1 > 0$ (and stat sig): **Additive.** Ads are growing the pie.
    *   If $\beta_1 \approx 0$: **Substitution.** Ads are just a tax on sales that would happen anyway.
    *   We log-transform to handle skew and interpret coefficients as elasticities (% change).

---

### 2. Vendor Level Analysis
**Question:** Does Vendor V paying for a click result in incremental revenue for Vendor V, compared to users who were interested but didn't click?

#### The Dataset: `df_vendor`
*   **Unit of Analysis (Row):** A `(User, Episode, Vendor)` tuple.
*   **Population Filter:** Include the row **IF AND ONLY IF**:
    1.  The Vendor **bid** on this user (Active Targeting).
    2.  **OR** The User **bought** from this Vendor (Organic conversion).
    *   *Exclude:* Random users the vendor never met. This keeps the matrix dense and relevant.

| Type | Column Name | Definition | Hypothesis / Purpose |
| :--- | :--- | :--- | :--- |
| **ID** | `EPISODE_ID` | Link to user session. | |
| **ID** | `VENDOR_ID` | The specific seller. | Fixed Effects grouping. |
| **Outcome ($Y$)** | `VENDOR_GMV` | Total spend with *this specific vendor* in this episode. | The target variable. |
| **Treatment ($T$)** | `VENDOR_CLICKS` | Count of clicks on *this* vendor's ads. | The direct cause. |
| **Instrument ($Z$)** | `WAS_SHOWN` | Binary: Did the vendor win *any* auction for this user? | **The Counterfactual Split:** Comparison of Winners vs. Losers. |
| **Control ($X$)** | `BID_INTENSITY` | Number of bids the vendor submitted for this user. | Proxy for how "badly" the vendor wanted this user. |
| **Context** | `COMPETITOR_CLICKS` | Clicks on *other* vendors in same episode. | Substitution effect. If I click your rival, do I spend less with you? |

#### The Regression Model (Fixed Effects)
$$ \text{VENDOR\_GMV}_{ij} = \alpha + \beta_1 \text{VENDOR\_CLICKS}_{ij} + \beta_2 \text{COMPETITOR\_CLICKS}_{i} + \gamma \text{BID\_INTENSITY}_{ij} + \mu_{vendor} + \epsilon $$

*   **Interpretation:**
    *   $\beta_1$: The dollar value of one additional click for the vendor.
    *   $\beta_2$: The cannibalization effect (likely negative).
    *   $\mu_{vendor}$: Vendor Fixed Effects (controls for "Nike sells more than Unknown Brand" purely due to brand equity).

---

### 3. The "Double Robust" Strategy for iROAS

To calculate the final **iROAS (Incremental Return on Ad Spend)**, we cannot just rely on the regression coefficient. We need to normalize by cost.

**Calculation Logic:**

1.  **Calculate Marginal Revenue:** Take $\beta_1$ from the Vendor Model. (e.g., 1 Click = \$5.00 incremental revenue).
2.  **Calculate Average Cost:** From `AUCTIONS_RESULTS`, calculate `AVG_CPC` (Cost Per Click) for that vendor. (e.g., \$0.50).
3.  **iROAS Formula:**
    $$ \text{iROAS} = \frac{\beta_1 (\text{Marginal Revenue})}{\text{AVG\_CPC}} $$

*   **Result:** If iROAS > 1.0, the ad spend is profitable. If < 1.0, they are losing money on every click.

### Summary of Script Logic (`05_incrementality_modeling.py`)

1.  **Ingest:** Load Imp, Click, Purch, Auction_Results, Auction_Users.
2.  **Episode Construction:** Loop/Vectorize creation of `EPISODE_ID` based on 48h breaks.
3.  **Marketplace Aggregation:** Group by Episode $\to$ Run Regression 1.
4.  **Vendor Aggregation:**
    *   Filter Auctions to relevant (User, Vendor) pairs.
    *   Join Funnel (Imp $\to$ Click $\to$ Purch).
    *   Group by (Episode, Vendor) $\to$ Run Regression 2.
5.  **Output:** One `.txt` file with:
    *   Model Summaries (Coefficients, P-values, R-squared).
    *   Derived iROAS estimates.
    *   Comparison of "Winner" vs "Loser" conversion rates.