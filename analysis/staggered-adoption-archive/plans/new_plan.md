# Refined End-to-End Execution Plan

We will execute a single, streamlined pipeline. We will not create separate "EDA" and "Modeling" data pulls. We will perform **one unified server-side aggregation** that produces a robust **Vendor-Week Panel**. This single dataset will be rich enough to answer the EDA questions (Orphan rates, concentration, trends) *and* serve as the input for the Causal Models (DiD, CS).

Mechanism checks (e.g., verifying First Price Auction) that require row-level data will be done via a tiny micro-sample pulled alongside the main panel, but treated as metadata validation, not the main dataset.

---

## Phase 1: The Unified Data Pull (Snowflake-Native)

**Objective:** Produce the `vendor_weekly_panel.parquet` (The "Golden Dataset").
**Method:** Modular SQL Execution via Python.

**Step 1.1: Materialize the Vendor Universe**
*   **Action:** Flatten `CATALOG` to map `PRODUCT_ID` $\to$ `VENDOR_ID`.
*   **Result:** A clean lookup table `TEMP_VENDOR_MAP`.
*   **Covariates:** While here, aggregate static vendor traits (Inventory Size, Median Price) into `TEMP_VENDOR_TRAITS`.

**Step 1.2: Aggregate The Treatment (Ad Spend)**
*   **Action:** Aggregate `AUCTIONS_RESULTS` (Winners only) by Vendor/Week.
*   **Metrics:** `Total_Spend`, `Total_Impressions_Won`.
*   **Optimization:** Use iterative `INSERT` by week if the table scan is too slow.

**Step 1.3: Aggregate The Outcome (Sales)**
*   **Action:** Join `PURCHASES` to `TEMP_VENDOR_MAP`.
*   **Metrics:** `Total_GMV`, `Total_Transactions`.
*   **Crucial Diagnostic:** In this step, we also calculate `Global_Orphan_GMV` (sum of GMV that *didn't* match `TEMP_VENDOR_MAP`) as a single scalar value to store in metadata.

**Step 1.4: Aggregate The Mechanism (Funnel)**
*   **Action:** Aggregate `IMPRESSIONS` and `CLICKS` by Vendor/Week.
*   **Metrics:** `Promoted_Impressions`, `Promoted_Clicks`.

**Step 1.5: The Final Assembly (Densification)**
*   **Action:** Scaffold (Vendor $\times$ Week) + Left Joins + Coalesce Zeros.
*   **Download:** Pull this final, dense panel (~4M rows) to `data/vendor_weekly_panel.parquet`.

**Step 1.6: The Micro-Sample (Mechanism Check)**
*   **Action:** `SELECT * FROM AUCTIONS_RESULTS SAMPLE(0.001) LIMIT 10000`.
*   **Download:** Save to `data/auction_mechanism_sample.parquet`.

---

## Phase 2: EDA & Validation (Local Python)

**Objective:** Validate the 10 checks using the downloaded Panel and Micro-Sample.

**Script:** `02_eda_diagnostics.py`
*   **Data Integrity:** Check Orphan Rate (from Phase 1 metadata). Check Panel Balance.
*   **Assumptions:** Plot Pre-trends (Visual Parallel Trends). Check Ashenfelter's Dip.
*   **Dynamics:** Calculate Treatment Persistence (Flicker Rate). Check Adoption Velocity.
*   **Mechanism:** Regress Rank vs Bid (using Micro-Sample). Verify `Price <= Final_Bid`.
*   **Heterogeneity:** Calculate Gini Coefficient (Whales). Check Zero-Inflation %.

---

## Phase 3: Causal Modeling (Local Python)

**Objective:** Estimate $ATT$ and $iROAS$.

**Script:** `03_causal_models.py`
*   **Input:** `vendor_weekly_panel.parquet`.
*   **Filtering:** Apply logic derived from EDA (e.g., "Exclude vendors with < 5 items" or "Exclude switchers if flicker > 20%").
*   **Model A (CS):** Run Callaway-Sant'Anna. Outcome: `log(Total_GMV)`.
*   **Model B (Mechanism):** Run CS on `Promoted_GMV` and `Organic_GMV` to check cannibalization.
*   **Model C (Extensive Margin):** If Zero-Inflation is high, model `P(Sales > 0)`.

**Script:** `04_heterogeneity.py`
*   **Action:** Split panel by `Inventory_Size` quartiles (from `TEMP_VENDOR_TRAITS`).
*   **Estimation:** Re-run CS for each subgroup to see who wins.

---

## Summary of Deliverables

We will have exactly **three** primary scripts and **two** data artifacts.

1.  **`01_build_panel.py`**: The heavy lifter. Orchestrates SQL to build the panel.
    *   *Artifacts:* `vendor_weekly_panel.parquet`, `auction_mechanism_sample.parquet`.
2.  **`02_run_eda.py`**: The diagnostician. Reads artifacts, prints the 10-point scorecard.
3.  **`03_estimate_models.py`**: The economist. Reads artifacts, runs models, prints results.

This is lean, modular, and handles the "Big Data" constraint by design.