# Data Pull Strategy: Staggered Adoption Incrementality

This strategy pivots from the previous "User Journey" extraction (sampling users) to a "Vendor Panel" extraction (aggregating vendors). To answer the research questions defined in the *Practitioner's Guide*—specifically estimating $ATT(g,t)$—we must observe the **Total GMV** for vendors, which requires capturing *all* their transactions, not just those from a sample of users.

## 1. Core Extraction Philosophy
**Shift from User-Centric to Vendor-Centric.**
Instead of hashing User IDs, we will define the universe of **Vendors** and aggregate their activity.
*   **Previous Approach:** Sample 0.1% of Users $\to$ See partial sales for many vendors. (Unsuitable for Total GMV).
*   **New Approach:** Select Target Vendors $\to$ Pull 100% of sales/ad activity for those vendors. (Required for Total GMV).

If the total data volume is too large for a full dump, we will perform **Server-Side Aggregation** (grouping by Vendor/Week inside Snowflake) rather than raw row extraction.

---

## 2. Data Pull Specifications

We will generate four distinct datasets to construct the `Vendor × Week` panel.

### Dataset A: The Vendor Universe (The "Bridge")
**Source:** `CATALOG`
**Objective:** Establish the mapping of `PRODUCT_ID` $\to$ `VENDOR_ID` and define baseline covariates.
*   **Logic:** Flatten the `VENDORS` array to create a unique mapping.
*   **Filters:** Filter for `ACTIVE` status if relevant, or include historical items to capture past organic sales.
*   **Covariates to Extract (for $X_i$):**
    *   `inventory_count`: Number of listings (proxy for "Whale" status).
    *   `avg_price_point`: Mean price of inventory (proxy for "Luxury" vs. "Budget").
    *   `main_category`: Top category by item count (for subgroup heterogeneity).

### Dataset B: The Outcome Ledger (Total Sales)
**Source:** `PURCHASES` joined with **Dataset A**
**Objective:** Construct $Y_{it}$ (Total GMV).
*   **Join Logic:** `PURCHASES.PRODUCT_ID` = `CATALOG.PRODUCT_ID`.
*   **Aggregation:** Group by `VENDOR_ID` and `WEEK` (using `DATE_TRUNC('WEEK', PURCHASED_AT)`).
*   **Metrics:**
    *   `total_gmv`: Sum of (`QUANTITY` * `UNIT_PRICE`).
    *   `transaction_count`: Count of unique `PURCHASE_ID`.
*   **Crucial QC Metric:** Calculate `unmapped_gmv_volume` (sum of GMV where `VENDOR_ID` is NULL) to validate the "Bridge."

### Dataset C: The Treatment Ledger (Ad Spend)
**Source:** `AUCTIONS_RESULTS`
**Objective:** Define Treatment Status $D_{it}$ and Cohorts $G_g$.
*   **Filter:** `IS_WINNER = TRUE` (only realized spend).
*   **Aggregation:** Group by `VENDOR_ID` and `WEEK`.
*   **Metrics:**
    *   `weekly_spend`: Sum of `PRICE` (preferred) or `FINAL_BID`.
    *   `weekly_impressions`: Count of won auctions.
    *   `avg_quality_score`: Mean `QUALITY` (to check if platform boosts high-quality vendors).
    *   `avg_pacing`: Mean `PACING` (mechanism check).

### Dataset D: The Attribution Ledger (Promoted Sales)
**Source:** `CLICKS` joined with `PURCHASES`
**Objective:** Distinguish Promoted vs. Organic GMV for Cannibalization checks.
*   **Logic:** Standard attribution window (e.g., 7-day click-to-purchase).
*   **Aggregation:** Group by `VENDOR_ID` and `WEEK`.
*   **Metrics:** `promoted_gmv`, `promoted_transactions`.
*   *Note:* `Organic GMV` will be calculated downstream as `Total GMV (Dataset B) - Promoted GMV (Dataset D)`.

---

## 3. The Construction Pipeline

The Python pipeline will assemble these extracts into the final panel.

1.  **Ingestion:** Load Parquet files for A, B, C, and D.
2.  **Scaffolding:** Create a Cartesian product of `Unique Vendors` $\times$ `All Weeks` to handle zeros. (Crucial: If a vendor makes no sales and spends nothing, they must appear as `0`, not missing).
3.  **Merging:** Left Join B, C, and D onto the Scaffold.
4.  **Imputation:** Fill NULLs with 0 for GMV and Spend.
5.  **Cohort Definition:** Calculate $G_i = \min(\text{Week} | \text{Spend} > 0)$.
    *   Vendors with no spend history are marked $G_i = \infty$ (Never-Treated Controls).

---

## 4. Required Checks (EDA Phase)

Per the *Practitioner's Guide*, we must validate specific assumptions using this data:

1.  **Orphan Rate Check:**
    *   *Question:* What % of `PURCHASES` do not match a `VENDOR_ID`?
    *   *Threshold:* If >5%, the "Total GMV" proxy is unreliable.

2.  **Treatment Stability (Flicker Analysis):**
    *   *Question:* Once a vendor starts spending, do they stop?
    *   *Action:* Calculate probability of $D_{it}=0 | D_{i,t-1}=1$. High flicker rates require using "Intent-to-Treat" estimators rather than standard Staggered DiD.

3.  **Covariate Overlap:**
    *   *Question:* Do "Never-Treated" vendors look like "Treated" vendors pre-treatment?
    *   *Action:* Check distribution of `inventory_count` for both groups. If disjoint, we must trim the sample (Assumption SO - Strong Overlap).

4.  **Zero-Inflation:**
    *   *Question:* What % of Vendor-Weeks have 0 sales?
    *   *Action:* If extremely high, we may need to aggregate to Monthly frequency or use extensive margin (Any Sale vs No Sale) as a separate outcome.