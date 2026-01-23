# Covariate Extraction Strategy for Staggered DiD

To satisfy **Assumption CPT (Conditional Parallel Trends)** described in Section 4.2 of the *Practitioner’s Guide*, we must condition on vendor characteristics that influence both the *decision to adopt ads* and the *organic sales trajectory*.

If "Whale" vendors (high inventory) naturally grow faster than "Boutique" vendors (low inventory), comparing them without conditioning introduces bias. We need to extract these time-invariant (or pre-treatment) covariates to construct propensity scores or segment the analysis.

---

## 1. Vendor-Level Covariates (Static Attributes)

These should be aggregated from the `CATALOG` and joined to the Master Panel.

| Covariate | Source | Aggregation Logic | Economic Justification (Why?) |
| :--- | :--- | :--- | :--- |
| **Inventory Depth** | `CATALOG` | `COUNT(PRODUCT_ID)` grouped by `VENDOR_ID`. <br>*(Filter: `ACTIVE=TRUE`)* | **Scale Effects.** Large sellers likely have higher organic visibility and different growth rates than small sellers. Crucial for **Assumption SO (Overlap)** checks. |
| **Price Point (AOV Proxy)** | `CATALOG` | `AVG(PRICE)` and `MEDIAN(PRICE)` grouped by `VENDOR_ID`. | **Market Segment.** Luxury vendors (High Price) operate in a lower-velocity, high-margin regime compared to fast-fashion vendors. Parallel trends likely fail across these segments. |
| **Brand Concentration** | `CATALOG` | Count of unique `brand-X` tags in `CATEGORIES` array per vendor. | **Reseller vs. Boutique.** Distinguishes between mono-brand boutiques and generalist resellers. |
| **Dominant Category** | `CATALOG` | Mode of `department-X` tag in `CATEGORIES` array. | **Sector Shocks.** Fashion trends affect "Shoes" differently than "Handbags." We may need to control for category-specific time shocks. |
| **Tenure (Platform Age)** | `CATALOG` | `MIN(CREATED_AT)` of any product associated with the vendor. *(Proxy)* | **Learning Curve.** New vendors grow rapidly (start-up phase); mature vendors have stable sales. Comparing new vs. old vendors violates parallel trends. |

---

## 2. Dynamic Pre-Treatment Covariates (Behavioral)

These must be calculated from the `PURCHASES` and `AUCTIONS_RESULTS` tables *before* the vendor enters the treatment group (Weeks $t < G_g$).

**Warning per *Guide* Section 4.1:** We must be careful not to control for "Bad Controls" (outcomes affected by treatment). These metrics must be strictly defined using pre-treatment data.

| Covariate | Source | Definition | Justification |
| :--- | :--- | :--- | :--- |
| **Pre-Trend Momentum** | `PURCHASES` | Slope of `TOTAL_GMV` in the 4 weeks prior to adoption ($t_{-4}$ to $t_{-1}$). | **Ashenfelter’s Dip.** Did the vendor start ads because sales were crashing? If so, we must control for this to avoid mean-reversion bias. |
| **Pre-Treatment Volatility** | `PURCHASES` | Standard Deviation of weekly `TOTAL_GMV` (Pre-treatment). | **Risk Profile.** Highly volatile vendors might adopt ads as insurance. |
| **Organic Baseline** | `PURCHASES` | Average Weekly `TOTAL_GMV` (Pre-treatment). | **Base Rate.** Higher volume sellers have different elasticities than lower volume sellers. |

---

## 3. Implementation in Data Pull

We do not need to pull these as separate time-series for every week. Instead, we generate a **Vendor Feature Store** (one row per vendor) that captures these attributes at the time of the snapshot or relative to their specific entry point.

**Refined Pipeline Step 1 (Vendor Attributes):**
```sql
SELECT 
    v.VENDOR_ID,
    COUNT(DISTINCT c.PRODUCT_ID) as inventory_depth,
    MEDIAN(c.PRICE) as median_price_point,
    MODE(c.CATEGORY_ID) as dominant_category
FROM VENDOR_PRODUCT_MAP v
JOIN CATALOG c ON v.PRODUCT_ID = c.PRODUCT_ID
GROUP BY 1
```

These covariates will be used in **Script 04 (Heterogeneity)** to split the sample (e.g., "High Inventory" vs "Low Inventory") and in **Script 03 (Propensity Scores)** if we use the Doubly Robust estimator (*Guide* Section 4.4).