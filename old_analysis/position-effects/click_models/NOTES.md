# Click Models - Data Notes

## Session Log Analysis (February 2026)

### Examination Patterns

Based on 10 sampled click sessions:

1. **Batched Impressions**: Items load in groups of 2-4 with shared timestamps
   - Typical scroll interval: 3-30 seconds
   - Users see approximately 2 items per "view"
   - Pattern breakdown: 78.5% MIXED, 18.5% SEQUENTIAL, 2.3% SINGLE batches

2. **Non-Monotonic CTR**: Position effects are NOT strictly decreasing
   - Rank 1: 2.93% CTR
   - Rank 5: 3.27% CTR (higher than rank 1)
   - Rank 13: 3.25% CTR
   - Suggests complex examination patterns or selection effects

3. **Multi-Click Rate**: 3.7% of sessions have 2+ clicks
   - SDBN cascade assumption violated for these sessions
   - DBN/UBM better suited for handling multi-click behavior

4. **Ranking vs Display Position**:
   - Products shown by RANKING order (1, 2, 3, ...)
   - Impressions may arrive out of order (e.g., rank 2 before rank 1 in same batch)

5. **Session Durations**: 0 seconds to approximately 3 minutes typical
   - Quick clicks: 2-10 seconds (decisive users)
   - Browse clicks: 30-180 seconds (exploratory users)

6. **Click Timing Anomalies**:
   - Clicks can occur on items not yet "impressed" (logging latency)
   - Multiple clicks on same item observed (e.g., rank 13 clicked twice)
   - Clicks span wide rank ranges (e.g., ranks 24 and 29 in same session)

### Model Assumptions vs Reality

| Assumption | PBM | DBN | SDBN | UBM | Data Reality |
|------------|-----|-----|------|-----|--------------|
| Linear traversal (top to bottom) | Yes | Yes | Yes | Yes | Mostly true (78% sequential scroll) |
| Position-only examination | Yes | No | No | No | Violated: CTR non-monotonic |
| Single click per session | No | No | YES | No | Violated: 3.7% multi-click |
| Independence across positions | Yes | Partial | No | No | Violated: batched viewing |
| Distance-to-last-click matters | No | Partial | No | YES | Unknown, UBM can test |

### Model Applicability Assessment

| Model | Literature Claim | Data Compatibility | Notes |
|-------|------------------|-------------------|-------|
| **PBM** | Robust baseline | Good | Independence assumption may be violated (batched items) |
| **DBN** | Primary workhorse | Good | Satisfaction vs attractiveness separation valuable |
| **SDBN** | Fast baseline | Moderate | Single-click assumption violated (3.7% multi-click rate) |
| **UBM** | Best log-likelihood | Good | Models distance-to-last-click |
| **Cascade** | Historical only | Poor | Strictly single-click; violated by data |

### Recommended Usage

**For position effect estimation**: Use PBM or feature-based model
- Simpler interpretation
- Position elasticity beta_1 approximately -0.06 from neural model

**For click prediction**: Use UBM or DBN
- Better handles multi-click sessions
- Accounts for sequential examination

**For unbiased LTR training**: Use DBN
- Separates attractiveness (snippet) from satisfaction (content)
- Can inform relevance ranking

## Data Summary

- 31,501 sessions, 222,890 impressions, 6,279 clicks
- CTR: 2.82%
- Placements: 1 (Homepage), 2 (PDP), 3 (Category), 5 (Cart)
- Max rank: 64, Mean items/session: 7.07

## References

From mc2015-clickmodels.pdf (Chuklin et al.):
- Chapter 4: PBM, DBN, SDBN formulations
- Chapter 5: UBM formulation and estimation
- Chapter 7: Experimental comparison showing UBM achieves best log-likelihood
- Chapter 8: Extensions (personalization, task models)
