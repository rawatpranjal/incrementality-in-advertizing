# Position Effects Analysis

## Objective

Estimate the causal effect of ad position (ranking) on click-through rate and downstream conversion. The goal is to separate true position effects (examination probability decay) from product quality effects that correlate with position via the auction mechanism.

## Data Mechanics Confirmed

From Round 1 exploration:

1. **Ranking is deterministic**: RANKING = descending order of QUALITY score. Correlation between QUALITY and RANKING is -1.0 within auction. FINAL_BID, CONVERSION_RATE, and PACING do not affect ranking in this system.

2. **IS_WINNER marks impression eligibility**: Products with IS_WINNER=TRUE were shown to users. Max winner rank varies by placement (3-5 slots typical).

3. **Impressions are batched**: All impressions within an auction share the same timestamp. No sequential examination signal available.

4. **Placements are distinct contexts**: Placement values (1, 2, 3, 5) represent different page locations. Distribution varies: Placement 3 is most common (~60%), Placement 1 (~19%), Placement 5 (~11%), Placement 2 (~11%).

5. **AUCTION_ID format differs between tables**: AUCTIONS tables use binary format requiring HEX conversion; IMPRESSIONS/CLICKS use UUID with dashes requiring REPLACE for joins.

## Key Discoveries Affecting Methodology

1. **No position variation within auction**: Each product appears at exactly one rank per auction. Position effect identification requires cross-auction comparison.

2. **QUALITY perfectly determines rank**: This is both a challenge (endogeneity) and an opportunity (sharp assignment). Products with higher QUALITY always rank higher, creating a deterministic running variable for RDD.

3. **Only promoted events observed**: IMPRESSIONS and CLICKS tables contain only sponsored products. Organic clicks/impressions are unobserved, limiting counterfactual construction.

4. **PURCHASES contain all transactions**: Unlike impressions/clicks, purchases include both promoted and organic. This allows distinguishing session outcomes but not path to purchase.

5. **Slot count varies by placement**: Different placements show different numbers of ads (2-5 typical). Effects likely differ by placement context.

## Causal Inference Viability

| Method | Feasibility | Notes |
|--------|-------------|-------|
| Position-Based Model (PBM) | Limited | No sequential examination signal (batched impressions). Would require strong assumptions. |
| IPW/Doubly Robust | Moderate | Propensity is near-deterministic (QUALITY → RANK). Positivity concerns at extremes. |
| RDD at Rank Cutoffs | Strong | QUALITY provides clean running variable. Compare rank k vs k+1 around IS_WINNER cutoff. |
| Discrete Hazard | Limited | No organic clicks observed. Session-level exit unidentified. |
| Partial Identification | Applicable | Bounds on position effect using Manski-style assumptions on organic behavior. |

**Recommended approach**: Focus on RDD at winner/loser boundary, supplemented by cross-placement comparisons where same products appear at different effective positions.

## Current Status

**Round 1 Complete**:
- Data pull notebook (01_data_pull.ipynb): 5 tables, 15-minute window, 1% sample
- EDA script (01_eda.py): Comprehensive exploration of auction mechanics
- Key finding: QUALITY determines RANKING, creating RDD opportunity

**Round 2 In Progress**:
- Data pull notebook (02_data_pull.ipynb): 6 tables (+PURCHASES), 60-minute window, 3% sample
- Expected yield: ~240K auctions, ~22K users, ~600K impressions

**Next Steps**:
1. Run 02_data_pull.ipynb to obtain expanded dataset
2. Build RDD analysis at winner/loser cutoff
3. Estimate position effects within placement strata
4. Assess magnitude of position effect relative to product quality effect

## Analysis Files

| File | Purpose | Output |
|------|---------|--------|
| 01_data_pull.ipynb | Initial data pull (15min, 1%) | *_all.parquet |
| 02_data_pull.ipynb | Expanded data pull (60min, 3%) | *_r2.parquet |
| 01_eda.py | Auction mechanics exploration | 01_eda_results.txt |
