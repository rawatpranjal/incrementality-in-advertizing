Here is the list of additional control features (covariates) derived strictly from your Data Dictionary. These are designed to isolate the causal effect of the **Click** by soaking up variance caused by **Item Quality**, **User Budget**, and **Market Dynamics**.

### 1. Economic & Pricing Controls (The "Wallet" Effect)
*Why: A user clicking a \$500 item has different purchasing potential than one clicking a \$10 item. We must control for the "Price Tier" of the session.*

*   **`AVG_SHOWN_PRICE`**: The mean `PRICE` (from `AUCTIONS_RESULTS` or joined `CATALOG`) of all items impressed in the session.
    *   *Mechanism:* Controls for the user's budget segment.
*   **`PRICE_VARIANCE`**: The standard deviation of prices shown.
    *   *Mechanism:* Captures "Discovery" (high variance) vs. "Targeted" (low variance) shopping modes.
*   **`AVG_BID_VALUE`**: The mean `FINAL_BID` of the winning ads shown.
    *   *Mechanism:* Proxy for commercial intent. High bids imply vendors believe this user is high-value.

### 2. Quality & Ranking Controls (The "Desirability" Effect)
*Why: Users click "better" items. If we don't control for quality, we conflate "Ad Effectiveness" with "Product Effectiveness."*

*   **`AVG_QUALITY_SCORE`**: The mean `QUALITY` score (from `AUCTIONS_RESULTS`) of all impressed ads.
    *   *Mechanism:* Crucial control. If high, the lift is likely due to the product, not just the ad placement.
*   **`AVG_DISPLAY_RANK`**: The mean `RANKING` of ads shown to the user (1=Top, 10=Bottom).
    *   *Mechanism:* Visibility bias. Top slots get clicked due to position, not just preference.
*   **`AVG_PACING_RATE`**: The mean `PACING` value of winners.
    *   *Mechanism:* Captures budget pressure. High pacing (0.9-1.0) means vendors are spending aggressively, potentially signaling a "hot" shopping period.

### 3. Session Intensity Controls (The "Boredom" Effect)
*Why: A user who sees 100 ads and clicks 1 is different from a user who sees 2 ads and clicks 1.*

*   **`AUCTION_DENSITY`**: Total Auctions / Duration Hours.
    *   *Mechanism:* Measures browsing speed. Fast scrolling = low attention per ad.
*   **`UNIQUE_CATEGORIES_SEEN`**: Count of distinct `CATEGORIES` (parsed from `CATALOG`) in the session.
    *   *Mechanism:* Focused intent (1 category) vs. Window shopping (10 categories).
*   **`TIME_TO_FIRST_CLICK`**: Minutes from Session Start to first `CLICK`.
    *   *Mechanism:* Impulsivity control.

### 4. Vendor-Specific Controls (For the Vendor Regression)
*Why: To separate "I love this Brand" from "This Ad worked."*

*   **`SHARE_OF_VOICE`**: (Vendor's Impressions / Total Impressions in Session).
    *   *Mechanism:* Did they buy because they were bombarded by this one vendor?
*   **`COMPETITOR_QUALITY_GAP`**: (Vendor's Avg Quality Score - Competitors' Avg Quality Score).
    *   *Mechanism:* Relative advantage. Did they win on merit?
*   **`PRIOR_VENDOR_AFFINITY`**: (Did User buy from Vendor in *previous* episodes? 0/1).
    *   *Mechanism:* Loyalty bias.

### Updated Regression Equation (Mental Draft)

$$ Y_{spend} = \beta_{click} T + \beta_{quality} \text{AvgQual} + \beta_{price} \text{AvgPrice} + \beta_{pos} \text{AvgRank} + \beta_{dur} \text{Duration} + \dots + \epsilon $$

I will create `05_incrementality_modeling.py` now.