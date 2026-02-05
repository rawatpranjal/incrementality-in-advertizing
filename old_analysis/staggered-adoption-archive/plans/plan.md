# Ad-Platform Incrementality Analysis: Strategic Plan

## 1. Context & Research Questions

**Context:**
We are analyzing "Promoted Closet," a vendor-level advertising feature on a Poshmark-style social commerce marketplace. The feature automates product selection for promotion based on a weekly vendor budget. Adoption is staggered (vendors adopt at different times) and reversible (vendors can pause campaigns).

**The Data Reality:**
We observe the full promoted funnel (Impressions $\to$ Clicks) but lack organic traffic data. However, we possess the "Ground Truth": the `PURCHASES` table contains *all* transactions. By linking `PURCHASES.PRODUCT_ID` $\to$ `CATALOG.PRODUCT_ID` $\to$ `CATALOG.VENDORS`, we can reconstruct the **Total GMV** for every vendor per week. This allows us to bypass the "attribution" debate and measure true "incrementality" at the vendor wallet level.

**Research Questions:**
1.  **The Net Lift (Primary):** What is the causal effect of adopting "Promoted Closet" on a vendor's **Total GMV**? (Does the ad spend actually grow the pie, or just pay for sales that would have happened anyway?)
2.  **Efficiency (ROAS):** What is the **Incremental ROAS** (iROAS)? For every \$1.00 of *actual cost* incurred (First Price auction), how much *incremental* Total GMV is generated?
3.  **Heterogeneity:** Do "Whale" vendors (large inventory) benefit disproportionately compared to small boutique sellers?
4.  **The "Tax" Check:** Does the mechanism force low-margin vendors to participate just to maintain baseline sales (business stealing), or does it unlock new liquidity?

---

## 2. Models & Robustness

We will employ a multi-model approach to triangulate the truth, moving from simple description to rigorous causal inference.

### Model A: Dynamic Event Study (Visual Proof)
*   **Method:** Standard Event Study specification centering data around the "First Adoption Week".
*   **Goal:** Visual inspection of **Pre-trends** (Parallel Trends assumption). We need to see flat GMV *before* adoption and a clear inflection point *at* adoption.
*   **Equation:** $\log(GMV_{it}) = \alpha_i + \lambda_t + \sum_{k=-4}^{12} \beta_k D_{it}^k + \epsilon_{it}$

### Model B: Callaway & Sant'Anna (Primary Estimator)
*   **Method:** Group-Time Average Treatment Effect ($ATT(g,t)$).
*   **Why:** Standard OLS/TWFE fails with staggered adoption (negative weighting variance). CS cleanly separates "Cohorts" (groups starting in Week $X$) and compares them only to valid controls (Not-yet-treated or Never-treated).
*   **Treatment Definition:** Treatment starts at the **first week** of positive spend.
    *   *Handling Reversibility:* For the primary model, we treat adoption as an "absorbing state" to measure the *impact of joining the program*. If churn is high, this estimates the "Intent to Treat" (ITT).

### Model C: Simple TWFE (Benchmark)
*   **Method:** Standard Fixed Effects Regression.
*   **Why:** It handles reversible treatment (switching ads ON/OFF) naturally, unlike CS. While biased under heterogeneity, it provides a useful baseline for the correlation between *current* active status and sales.

### Robustness Strategy
1.  **The "Flicker" Check:** Re-estimate excluding vendors who churn ON/OFF rapidly. Focus on "Stayers."
2.  **Placebo Outcome:** Test the effect on a metric that *shouldn't* move (e.g., Number of organic listings created—unless ads incentivize listing more).
3.  **Alternative Control Groups:** Compare Treated vs. Never-Treated only (cleanest) vs. Treated vs. Not-Yet-Treated (more power).

---

## 3. Data Pull Requirements

We need to assemble a **Rectangular Panel** (Vendor $\times$ Week) covering the analysis period.

**Step 1: The Linkage Table (Product $\to$ Vendor)**
*   **Source:** `CATALOG`
*   **Action:** Flatten the `VENDORS` array. Filter for `ACTIVE` products (optional, but good for context).
*   **Output:** Map of `PRODUCT_ID` $\to$ `VENDOR_ID`.

**Step 2: The Spend Ledger (Treatment)**
*   **Source:** `AUCTIONS_RESULTS`
*   **Filters:** `IS_WINNER = TRUE`
*   **Aggregation:** Sum `FINAL_BID` (Cost) by `VENDOR_ID` and `WEEK`.
*   **Output:** `vendor_spend_weekly` (`vendor_id`, `week`, `spend_amount`, `is_active`).

**Step 3: The GMV Ledger (Outcome)**
*   **Source:** `PURCHASES` joined to **Step 1 Map**.
*   **Aggregation:** Sum `(QUANTITY * UNIT_PRICE)` by `VENDOR_ID` and `WEEK`.
*   **Output:** `vendor_gmv_weekly` (`vendor_id`, `week`, `total_gmv`, `total_items`).

**Step 4: The Master Panel (Densification)**
*   **Action:** Create a scaffold of all unique `VENDOR_IDs` $\times$ all `WEEKS`.
*   **Join:** Left join the Spend and GMV ledgers.
*   **Fill:** **CRITICAL STEP.** Fill `NULL` GMV and Spend with `0`. (A vendor exists even if they sold nothing that week).

---

## 4. The Pipeline

We will script this to be modular and file-based.

1.  **`01_linkage_and_panel_build.py`**:
    *   Executes the joins and densification.
    *   Calculates `log_gmv` and `log_spend`.
    *   **QC:** Prints match rates (What % of purchases failed to link to a vendor?).

2.  **`02_treatment_definition.py`**:
    *   Calculates `first_spend_week` for every vendor.
    *   Categorizes vendors: `Always_Treated`, `Never_Treated`, `Staggered_Adopter`.
    *   Calculates `relative_week` ($t - G_g$).

3.  **`03_descriptive_evidence.py`**:
    *   Plots raw average GMV for Treated vs Control cohorts over time.
    *   Checks for "Ashenfelter’s Dip" (Do vendors start ads because sales are tanking?).

4.  **`04_estimation_cs.py`**:
    *   Runs Callaway & Sant'Anna.
    *   Outputs the Aggregated ATT (Average Treatment Effect on Treated).
    *   Outputs the Event Study plot data (coefficients per relative week).

5.  **`05_heterogeneity_analysis.py`**:
    *   Splits panel by `Pre_Treatment_Inventory_Size` (Quartiles).
    *   Runs simple DiD or CS for each quartile to see who wins.

---

## 5. Further Ideas (Parking Lot)

*   **Cannibalization Ratio:** If we *could* heuristically flag purchases as "organic" (e.g., based on time-since-click), we could regress `Promoted GMV` against `Organic GMV`.
*   **Auction Density:** Use `AUCTION_ID` counts to measure "Share of Voice." Does a vendor with 10% impression share get 10% of sales, or is there a "winner take all" threshold?
*   **Spillover Effects:** Does treating Vendor A hurt Vendor B in the same category? (Stable Unit Treatment Value Assumption violations). Hard to test, but worth noting.

# Ad-Platform Incrementality Analysis: Strategic Plan

## 1. Context & Research Questions

**Context:**
We are analyzing a two-sided marketplace where vendors adopt "Promoted Closet," a paid advertising feature. Adoption is staggered (vendors start at different weeks) and reversible (vendors can pause). The core data challenge is the observability gap: we see promoted funnel metrics but lack organic impression/click data. However, `PURCHASES` contains the ground truth of all transactions.

**Research Questions:**
1.  **The Net Lift (Primary Estimand):** What is the Average Treatment Effect on the Treated (ATT) of adopting Promoted Closet on a vendor’s **Total GMV**? Does ad spend generate net new revenue, or does it primarily cannibalize organic sales?
2.  **Incremental ROAS (Efficiency):** For every $1.00 of actual cost incurred (First Price auction), how much *incremental* Total GMV is generated?
3.  **Heterogeneity (Mechanism):** Do "Whale" vendors (high inventory depth) derive differential benefits compared to small sellers? (Connecting to *Practitioner’s Guide* Section 4.5 on Heterogeneity).

---

## 2. Methodology & Connection to Practitioner's Guide

We will follow the "Forward-Engineering" approach recommended in the provided text, moving from target parameters to identification assumptions, rather than reverse-engineering a regression.

### The Model: Group-Time Average Treatment Effect (ATT(g,t))
*   **Reference:** *Practitioner’s Guide* Section 5.2.1.
*   **Why not TWFE?** As per Section 5.3, standard Two-Way Fixed Effects regressions bias estimates in staggered adoption designs due to "negative weighting" of cohorts. Since early adopters act as controls for late adopters, treatment effect heterogeneity over time can flip the sign of the coefficient.
*   **Chosen Estimator:** Callaway & Sant'Anna (CS).
*   **Equation:** We will estimate $ATT(g, t)$ comparing treated vendors in cohort $g$ to a control group (either "Never-Treated" or "Not-Yet-Treated") at time $t$.
    $$ATT(g, t) = E[Y_{t} - Y_{g-1} | G=g] - E[Y_{t} - Y_{g-1} | G=Control]$$

### Identification Strategy
1.  **Treatment Definition:** Week of first positive ad spend ($G_g$). We will treat the first adoption as the "event" to fit the standard DiD framework initially.
2.  **Control Group Selection:** We will test both "Never-Treated" (cleaner inference, *Guide* Section 5.2.2) and "Not-Yet-Treated" (more power).
3.  **Assumption:** **Parallel Trends (PT)**. In the absence of advertising, the trend in Total GMV for adopters would have been the same as for non-adopters.

---

## 3. The 10 Critical EDA Checks (Methodological Diagnostics)

Before modeling, we must perform the diagnostic checks outlined in the *Practitioner’s Guide* to validate our assumptions.

### A. Data Integrity & The "Bridge"
1.  **The "Orphan" Rate:** What percentage of rows in the `PURCHASES` table fail to join to a valid `VENDOR_ID` via the `CATALOG`? If this is high (>5%), our proxy for Total GMV is invalid.
2.  **Panel Balance:** What is the attrition rate of vendors? Do we have a stable "Never-Treated" control group, or do non-adopters simply leave the platform? (DiD assumes stable composition).

### B. Testing Identification Assumptions (Guide Section 4 & 5)
3.  **Pre-Trend Plausibility (The "Eye Test"):** Plot raw average Total GMV for treated vs. control groups aligned by calendar time. Do they move in parallel *before* the treated groups start spending? (*Guide* Section 3.2).
4.  **Ashenfelter’s Dip (Anticipation):** Do vendors adopt ads specifically *because* they experienced a sales slump in the prior weeks? If $Y_{t-1}$ predicts treatment, the "No Anticipation" assumption (Assumption NA, *Guide* Section 3.1) is violated.
5.  **Covariate Balance:** Are the "Treated" and "Control" vendors fundamentally different? Compare distributions of `Inventory_Size`, `Average_Price`, and `Tenure` at baseline ($t=0$). If imbalanced, we must use Conditional Parallel Trends (*Guide* Section 4.1).
6.  **Treatment Staggering:** Plot the count of new adopters per week. Is there sufficient variation in $G_g$ (start dates)? If everyone starts in Week 1, this is a cross-sectional study, not DiD.

### C. Treatment Dynamics
7.  **Flicker Rate (Reversibility):** Of vendors who start spending, what % stop spending in subsequent weeks? The standard CS estimator assumes treatment is absorbing. High churn requires defining treatment as "Intent-to-Treat" (ever-adopted) or using complex reversible estimators.
8.  **Spend Concentration:** What is the Gini coefficient of Ad Spend? If 1% of vendors drive 99% of spend, Average Treatment Effects will be non-representative.

### D. Mechanism & Outcomes
9.  **Zero-Inflation:** What % of vendor-weeks have $0 Total GMV? High zero-counts break log-linear specifications. We may need to separate Extensive Margin (probability of sale) vs. Intensive Margin (value of sales).
10. **The Cannibalization Signal:** For treated vendors, what is the raw correlation between `Promoted_GMV` and `Organic_GMV` (Total - Promoted) in post-treatment periods? A correlation of -1.0 implies pure cannibalization.

---

## 4. Data Pull Plan

We need to construct a **Rectangular Panel** (Vendor $\times$ Week).

1.  **Vendor Attribute Map:**
    *   `CATALOG`: Flatten `VENDORS` array $\to$ Unique `VENDOR_ID`.
    *   Calculate baseline covariates: `Inventory_Count`, `Avg_Price_Point`.

2.  **The "Bridge" (Total Sales Ledger):**
    *   `PURCHASES` $\to$ Join `CATALOG` on `PRODUCT_ID` $\to$ Group by `VENDOR_ID`, `WEEK`.
    *   Metric: `TOTAL_GMV` (Sum of Price * Quantity).

3.  **The Treatment Ledger:**
    *   `AUCTIONS_RESULTS`: Filter `IS_WINNER=TRUE`. Group by `VENDOR_ID`, `WEEK`.
    *   Metric: `TOTAL_SPEND` (Sum of `FINAL_BID`), `HAS_SPEND` (Binary).

4.  **The Master Panel:**
    *   Outer Join attributes, Sales, and Spend on `VENDOR_ID` and `WEEK`.
    *   **Imputation:** Fill `NULL` GMV and Spend with `0` (active vendors with no activity).

---

## 5. The Pipeline

1.  `01_panel_construction.py`: Ingests raw parquet files, executes the "Bridge" join, and outputs a `vendor_weekly_panel.parquet`.
2.  `02_eda_diagnostics.py`: Generates the summary stats to answer the 10 Checks above (Pre-trend plots, Covariate Balance tables).
3.  `03_cs_estimation.py`: Runs the Callaway-Sant'Anna estimator on `log(TOTAL_GMV + 1)`. Outputs the aggregated ATT.
4.  `04_heterogeneity.py`: Splits the panel by `Inventory_Size` quartiles and re-runs the estimator to check for effect heterogeneity (*Guide* Section 4.5).

---

## 6. Further Ideas (Parking Lot)

*   **Placement Heterogeneity:** Do ads in "Search" (High Intent) generate more incrementality than ads in "Feed" (Browsing)?
*   **Competition IV:** If DiD assumptions fail, we can try using "Auction Competition Intensity" (average bidders per auction) as an Instrument for Vendor Spend.
*   **Distributional Effects:** Does advertising help vendors clear "stale" inventory (old products)? (*Guide* Appendix A.4).