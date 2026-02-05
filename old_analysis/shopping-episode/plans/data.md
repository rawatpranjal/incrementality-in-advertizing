Here is the data extraction plan. Since the database is massive and the analysis requires complex joins (Auction Losers, Catalog Metadata, Funnel Events), we cannot simply "select *". We must use a **Targeted Sampling** approach.

We will use the existing Snowflake connector framework but modify the logic to ensure we get a coherent "Vertical Slice" of the marketplace (User + All their Exposures + All their Actions).

### 1. Sampling Strategy: Deterministic User Hashing
We cannot sample random rows (e.g., `LIMIT 1000`). We must sample **User Journeys**.
*   **Method:** `MOD(ABS(HASH(USER_ID)), 100)`
*   **Rate:** **2.0%** (Sample 2 buckets out of 100).
    *   *Why:* 0.1% (current code) yields ~100 users. We need ~5,000+ users to get statistical significance for the Vendor Choice model, specifically to capture enough "rare" events where a user clicks Vendor B but buys Vendor A.
*   **Persistence:** The hash ensures that if we run the query for Auctions and then for Purchases, we get the *same* users.

### 2. Time Window Definition
*   **Analysis Window:** 30 Days.
    *   *Why:* We need to capture the full "48-hour gap" episodes. A 14-day window risks cutting off the head or tail of a complex shopping episode.
*   **Lookback Window:** +30 Days prior (Total 60 days).
    *   *Why:* To calculate the `PRIOR_VENDOR_AFFINITY` feature. We need to know if they bought from Nike *last month* to control for loyalty.

### 3. The Extraction Logic (SQL via Python)

We will execute 5 distinct queries. Each query filters strictly on the `SAMPLED_USER_IDS` Common Table Expression (CTE).

#### A. The Anchor CTE (Python String Injection)
```sql
WITH SAMPLED_USERS AS (
    SELECT DISTINCT OPAQUE_USER_ID as USER_ID
    FROM AUCTIONS_USERS
    WHERE CREATED_AT BETWEEN '{start_date}' AND '{end_date}'
    AND MOD(ABS(HASH(OPAQUE_USER_ID)), 100) < 2  -- 2% Sample
)
```

#### B. Table-Specific Extraction Plans

| Table | Filtering Logic | Purpose |
| :--- | :--- | :--- |
| `AUCTIONS_USERS` | `JOIN SAMPLED_USERS` | Establishes the timestamp of every potential exposure. |
| `AUCTIONS_RESULTS` | `JOIN AUCTIONS_USERS` (filtered) | **Biggest Table.** Contains Winners AND Losers. We need this to build the "Choice Set" (who did the user see vs. who could they have seen?). |
| `IMPRESSIONS` | `JOIN SAMPLED_USERS` | Confirms valid exposure. (The "Shown" flag). |
| `CLICKS` | `JOIN SAMPLED_USERS` | The Treatment Variable. |
| `PURCHASES` | `JOIN SAMPLED_USERS` | The Outcome Variable. Note: Pull **ALL** purchases for these users, not just attributed ones. |

#### C. The Catalog Strategy (The "Mono-CTE")
The Catalog is tricky because it only contains sponsored products.
*   **Logic:** Collect unique `PRODUCT_ID`s from the extracted `IMPRESSIONS`, `CLICKS`, and `PURCHASES` dataframes in Python *after* the initial pull.
*   **Query:** `SELECT * FROM CATALOG WHERE PRODUCT_ID IN (...)`.
*   **Handling Missing Organic Items:** If a user buys a product ID that is not in the `CATALOG` table (an organic-only item), we will flag it as `CATEGORY = 'Organic_Unknown'` and `PRICE = <Unit_Price from Purchase Table>`.

### 4. Python-Side Processing (The "Stitching")

Once data is in memory (Pandas/Polars), `05_data_pull.py` will perform the following immediately to reduce size before saving:

1.  **Composite Keys:** Create `hash(AUCTION_ID, PRODUCT_ID)` to join Auctions to Impressions/Clicks.
2.  **Episode Identification:**
    *   Sort `AUCTIONS_USERS` and `PURCHASES` by `USER_ID, TIMESTAMP`.
    *   Calculate `time_diff`.
    *   `Episode_ID = cumsum(time_diff > 48 hours)`.
3.  **Filter:** Drop all Episodes where `Total_Impressions == 0`. (We don't care about pure organic users for *this* specific ad-effectiveness study).
4.  **Feature Append:** Join `PRICE`, `QUALITY`, `PACING` from `AUCTIONS_RESULTS` to the Episode.

### 5. Sizing Estimates (For 2% Sample, 30 Days)
*   **Users:** ~2,000 - 5,000
*   **Auctions:** ~1M rows (Assuming 50 bids/user/day is too high, likely 10-20).
*   **Bids (Results):** ~40M rows (This is the bottleneck. 40 bids per auction).
    *   *Optimization:* If `AUCTIONS_RESULTS` > 5GB RAM, we will modify the SQL to `QUALIFY ROW_NUMBER() OVER (PARTITION BY AUCTION_ID ORDER BY RANKING ASC) <= 10`. We only need the top 10 losers to model competition; the 50th bidder is irrelevant.
*   **Purchases:** ~500 - 1,000 rows.

### TLDR
We will modify the existing `01_data_pull` script to create `05_data_pull_incrementality.py`. It will use a **2% deterministic user hash** to pull a vertical slice of 30 days. Crucially, it will limit `AUCTIONS_RESULTS` to the top 10 bids per auction if memory becomes an issue, ensuring we have the "Winner" and the "Best Losers" (Counterfactuals) without pulling 50 million junk bids. We will pull **all** purchases for these users to calculate the "Organic Gap."