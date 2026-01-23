# Shopping Episode Incrementality Analysis

## TLDR Convention

When completing a task, always provide a TLDR paragraph summary in full sentences, written as a senior economist or data scientist would. Include:
- Context: what problem was being addressed
- Approach: what methods/techniques were tried
- Findings: what the results showed
- Issues: any problems or limitations encountered
- Takeaways: key implications or next steps

---

## Unit of Analysis: The Shopping Episode

An **EPISODE** is a contiguous sequence of user activity (impressions, clicks, purchases) bounded by **48-hour inactivity gaps**.

- Reject 30-minute web session as too short for high-consideration commerce
- Reject 7-day window as too noisy
- Universe restricted to episodes with at least one sponsored impression (purely organic sessions are invisible to ad-server logs)

---

## Pipeline

```
01_data_pull.py        # 2% deterministic user sample from Snowflake, 30-day window
02_eda.py              # Episode validation, sparsity checks, vendor competition
03_modeling.py         # Three models: Marketplace OLS, Vendor FE, MNLogit
99_combine_results.py  # Combine all results into FULL_RESULTS.txt
```

---

## Data Extraction (01_data_pull.py)

### Sampling Strategy
- **Method**: `MOD(ABS(HASH(USER_ID)), 100) < 2` (2% deterministic hash)
- **Time Window**: 30 days analysis + 30 days lookback (60 days total)
- **Bid Limit**: Top 10 bids per auction (via `QUALIFY ROW_NUMBER()`)

### Tables Extracted
| Table | Filter | Purpose |
|-------|--------|---------|
| AUCTIONS_USERS | Sampled users | Establishes exposure timestamps |
| AUCTIONS_RESULTS | Top 10 ranks per auction | Winners + best losers (counterfactuals) |
| IMPRESSIONS | Sampled users | Confirms valid exposure |
| CLICKS | Sampled users | Treatment variable |
| PURCHASES | ALL for sampled users | Outcome variable (includes organic) |
| CATALOG | Products from events | Metadata (price, category) |

### Episode Construction
1. Sort all events by (USER_ID, timestamp)
2. Calculate time_diff between consecutive events
3. EPISODE_ID = cumsum(time_diff > 48 hours)
4. Drop episodes with 0 impressions

### Expected Sizes (2% sample, 30 days)
- Users: 2,000 - 5,000
- Bids: ~40M rows (limited to top 10)
- Purchases: 500 - 1,000

---

## EDA Validation (02_eda.py)

### 1. Episode Definition Check
- Inter-event time distribution (look for valley at 24h/48h)
- Episode duration: mean, median, 95th percentile
- Event density per episode (impressions, clicks, auctions)

### 2. Outcome Sparsity
- Browser ratio: % episodes with zero purchases
- Organic gap: % of purchases that are non-promoted
- GMV distribution for non-zero episodes

### 3. Vendor Competition
- SOV concentration (Gini coefficient within episode)
- Unique vendors per episode (need mean > 1.5 for MNLogit)
- Consideration set size

### 4. Counterfactual Validity
- Winner/Loser intersection (Jaccard index)
- Loss ratios per vendor

### Go/No-Go Checks
| Condition | Action |
|-----------|--------|
| Episodes > 1 week | Reduce gap threshold |
| 99% zero GMV | Use Hurdle models |
| 1 vendor per user | Cancel MNLogit, use Binary Logit |
| No losers | Cancel vendor iROAS |

---

## Three Analytical Models (03_modeling.py)

### Model I: Marketplace Elasticity (OLS)

**Question**: Does increasing ad volume grow total GMV or just cannibalize organic?

**Equation**:
$$\log(\text{TOTAL\_GMV} + 1) = \alpha + \beta_1 \log(\text{TOTAL\_CLICKS} + 1) + \beta_2 \log(\text{DURATION}) + \epsilon$$

**Interpretation**:
- $\beta_1 > 0$: Ads are additive (growing the pie)
- $\beta_1 \approx 0$: Ads are substitutive (tax on organic sales)

**Dataset**: `df_marketplace` (one row per EPISODE_ID)
| Column | Definition |
|--------|------------|
| EPISODE_ID | Unique episode identifier |
| TOTAL_GMV | Sum of all purchases (promoted + organic) |
| TOTAL_CLICKS | Count of sponsored clicks |
| TOTAL_IMPRESSIONS | Count of sponsored impressions |
| DURATION_HOURS | Time between first and last event |
| AD_INTENSITY | Impressions per hour |
| PRICE_TIER | Average price of items shown |

---

### Model II: Vendor iROAS (Fixed Effects)

**Question**: Does vendor V paying for a click generate incremental revenue vs. just bidding and losing?

**Equation**:
$$\text{VENDOR\_GMV}_{ij} = \alpha + \beta_1 \text{VENDOR\_CLICKS}_{ij} + \beta_2 \text{COMPETITOR\_CLICKS}_i + \gamma \text{BID\_INTENSITY}_{ij} + \mu_{vendor} + \epsilon$$

**Interpretation**:
- $\beta_1$: Dollar value of one additional click
- $\beta_2$: Cannibalization effect (likely negative)
- iROAS = $\beta_1$ / AVG_CPC

**Dataset**: `df_vendor` (one row per (EPISODE_ID, VENDOR_ID))
| Column | Definition |
|--------|------------|
| EPISODE_ID | Link to user session |
| VENDOR_ID | Specific seller |
| VENDOR_GMV | Spend with this vendor in episode |
| VENDOR_CLICKS | Clicks on this vendor's ads |
| IS_WINNER | Binary: won any auction for this user? |
| BID_INTENSITY | Number of bids submitted |
| COMPETITOR_CLICKS | Clicks on other vendors |

---

### Model III: Multinomial Logit (Competition)

**Question**: How does ad intensity shift probability of choosing Vendor A vs. Vendor B vs. walking away?

**Equation**:
$$P(\text{Choice}=j) = \frac{\exp(\alpha_j + \beta_{imp}I_j + \beta_{click}C_j + \beta_{price}P_j)}{\sum_k \exp(\alpha_k + \beta_{imp}I_k + \beta_{click}C_k + \beta_{price}P_k)}$$

**Choice Set per Episode**:
1. **No_Buy**: Outside option (always present)
2. **Organic**: Purchase non-promoted item (always present)
3. **Focal_Vendor**: Promoted vendor with highest Share of Voice
4. **Rival_Vendor**: Promoted vendor with 2nd highest Share of Voice

**Interpretation**:
- $\beta_{click}$: Conversion efficiency of clicks
- $\beta_{imp}$: Passive value of exposure (Billboard effect)
- Denominator captures zero-sum competition

**Dataset**: `df_choice` (long format, one row per (EPISODE_ID, OPTION_ID))
| Column | Definition |
|--------|------------|
| EPISODE_ID | Group identifier |
| OPTION_ID | No_Buy, Organic, or Vendor_Hash |
| IS_CHOSEN | Target: 1 if purchased, 0 otherwise |
| HAS_ADS | 1 for promoted candidates, 0 for No_Buy/Organic |
| IMP_COUNT | Impressions shown by this vendor |
| CLICK_COUNT | Clicks on this vendor |
| MEAN_RANK | Average position (1=Top) |
| MEAN_PRICE | Average price of items shown |

---

## Control Variables

### Economic Controls
- AVG_SHOWN_PRICE: Mean price of impressed items
- PRICE_VARIANCE: Std dev of prices (discovery vs. targeted)
- AVG_BID_VALUE: Mean final bid (proxy for commercial intent)

### Quality Controls
- AVG_QUALITY_SCORE: Mean quality score of ads
- AVG_DISPLAY_RANK: Mean ranking position
- AVG_PACING_RATE: Mean pacing value (budget pressure)

### Intensity Controls
- AUCTION_DENSITY: Auctions per hour (browsing speed)
- UNIQUE_CATEGORIES_SEEN: Count of distinct categories
- TIME_TO_FIRST_CLICK: Minutes to first click (impulsivity)

### Vendor-Specific Controls
- SHARE_OF_VOICE: Vendor impressions / total impressions
- COMPETITOR_QUALITY_GAP: Vendor quality - competitor quality
- PRIOR_VENDOR_AFFINITY: Purchased from vendor in prior episodes (0/1)

---

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Episode gap | 48 hours | Inactivity threshold |
| Sample rate | 2% | Deterministic hash |
| Analysis window | 30 days | Primary period |
| Lookback window | 30 days | For affinity features |
| Bid limit | Top 10 per auction | Memory optimization |

---

## Output Convention

### File Structure
```
results/
├── 01_data_pull.txt      # Extraction stats, episode counts
├── 02_eda.txt            # Validation checks, go/no-go results
├── 03_modeling.txt       # All three model outputs
└── FULL_RESULTS.txt      # Combined output
```

### Rules
- All scripts output to `results/*.txt`
- Raw stdout logs, no opinions or interpretation
- Tables, numbers, model outputs only
- Another AI will read and interpret

---

## Data Sources

From `eda/data/` (or fresh Snowflake pull):
| File | Contents |
|------|----------|
| auctions_results_365d.parquet | Bid-level data with ranking, vendor |
| auctions_users_365d.parquet | User-auction mapping |
| impressions_365d.parquet | Promoted impressions |
| clicks_365d.parquet | Promoted clicks |
| purchases_365d.parquet | All purchases |
| catalog_365d.parquet | Product metadata |

---

## Archive

Legacy scripts from previous User-Week-Vendor approach are in `archive/legacy_utv_pipeline/`.

---

## Guidelines

- Always use tqdm with verbosity for progress tracking
- One modular .py file leads to one .txt result file
- Results in .txt must be ultra verbose (entire stdout captured)
- No opinions or interpretation in results files
- Tone: academic, like a statistician or economist
- No graphs, csv, pkl - each .py creates one .txt
