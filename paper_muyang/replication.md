# Replication Guide

This document maps paper tables and figures to their source code and result files.

**Verification Note**: All table values were cross-checked against source files on 2026-02-03.

---

## Section 4: Data

| Table | Label | Source Script | Result File |
|-------|-------|---------------|-------------|
| Data Sources | `tab:data_structure` | Manual/descriptive | N/A |
| Average Daily Platform Statistics | `tab:summary_stats` | `analysis/staggered-adoption/scripts/02_run_eda.py` | `analysis/staggered-adoption/results/MASTER_RESULTS.txt` |

---

## Section 5: Analysis of Vendor Panels (vendor_week.tex)

**Important**: This section uses TWO different vendor panels:
1. **Descriptive panel** (979,290 obs, 150,075 vendors) from `analysis/panel/vendor_week.ipynb`
2. **Causal/DiD panel** (846,430 obs, 142,920 vendors) from `analysis/staggered-adoption/`

### Descriptive Tables (from analysis/panel/vendor_week.ipynb)

| Table | Label | Source Script | Result File |
|-------|-------|---------------|-------------|
| Vendor-Week Panel Dimensions | `tab:vendor_panel_dims` | `analysis/panel/vendor_week.ipynb` | Cell outputs (979,290 obs, 150,075 vendors) |
| Distribution of Observations per Vendor | `tab:vendor_obs_dist` | `analysis/panel/vendor_week.ipynb` | Cell outputs |
| Distribution of Key Metrics per Vendor-Week | `tab:vendor_dist` | `analysis/panel/vendor_week.ipynb` | Cell outputs |
| Weekly Platform Aggregates | `tab:weekly_agg` | `analysis/panel/vendor_week.ipynb` | Cell outputs |
| Fixed-Effects Model of Vendor-Week Revenue | `tab:vendor_fe_results` | `analysis/panel/vendor_week.ipynb` | Cell outputs (β=0.6422) |
| Summary Statistics of Vendor Elasticities | `tab:vendor_beta_v_dist` | `analysis/panel/vendor_week.ipynb` | Cell outputs (Mean=0.77) |
| iROAS Calculation Parameters | (no label) | Manual calculation | N/A |
| iROAS Sensitivity Analysis | (no label) | Manual calculation | N/A |

### Causal Tables (from analysis/staggered-adoption/)

| Table | Label | Source Script | Result File |
|-------|-------|---------------|-------------|
| Staggered DiD Estimates | `tab:staggered_main` | `analysis/staggered-adoption/scripts/03_callaway_santanna.py` | `analysis/staggered-adoption/results/MASTER_RESULTS.txt` (846,430 obs) |
| Pre-Trends Assessment | `tab:staggered_pretrends` | `analysis/staggered-adoption/scripts/03_callaway_santanna.py` | `analysis/staggered-adoption/results/MASTER_RESULTS.txt` (ATT impressions=+1.06) |

### Figures

| Figure | Label | Source Script | Result File |
|--------|-------|---------------|-------------|
| Event Study: Impressions | `fig:event_study_impressions` | `analysis/staggered-adoption/scripts/03_callaway_santanna.py` | `analysis/staggered-adoption/figures/event_study_impressions.png` |
| Event Study: Clicks | `fig:event_study_clicks` | `analysis/staggered-adoption/scripts/03_callaway_santanna.py` | `analysis/staggered-adoption/figures/event_study_clicks.png` |
| Event Study: Total GMV | `fig:event_study_total_gmv` | `analysis/staggered-adoption/scripts/03_callaway_santanna.py` | `analysis/staggered-adoption/figures/event_study_total_gmv.png` |

---

## Section 6: Analysis of Shopping Sessions (funnel_analysis_v2.tex)

**Source**: `eda/` directory (NOT `analysis/shopping-episode/` or `analysis/shopping-sessions/`)

### Tables

| Table | Label | Source Script | Result File |
|-------|-------|---------------|-------------|
| Macro-Session Summary Statistics | `tab:macro_session_stats` | `eda/06_macro_sessions.ipynb` | Cell outputs (32,111 sessions, 7,385 users) |
| Descriptive Statistics | `tab:descriptives` | `eda/06_macro_sessions.ipynb` | Cell outputs (1,786,179 obs) |
| Session-Level Conversion Funnel | `tab:funnel` | `eda/06_macro_sessions.ipynb` | Cell outputs |
| Variable Definitions | `tab:variable_definitions` | Manual/descriptive | N/A |
| Average Treatment Effect of Click | `tab:main_results` | `eda/07_fixed_effects.ipynb` | `eda/data/fixed_effects_summary.txt` (ATE Purchase=0.0074, ATE Log-Revenue=0.0575) |
| Robustness (DML rows) | `tab:robustness` | `eda/08_double_ml.ipynb` | `eda/data/double_ml_log.txt` |

**Verification**:
- Paper: ATE Purchase = 0.0074 → Result file: 0.007388 ✓
- Paper: ATE Log-Revenue = 0.0575 → Result file: 0.057471 ✓
- Paper: N = 1,786,179 → Result file: 1,786,179 ✓
- Paper: Sessions = 32,111 → Result: 32,111 ✓
- Paper: Users = 7,385 → Result: 7,385 ✓

**Note**: The `analysis/shopping-episode/` and `analysis/shopping-sessions/` directories contain DIFFERENT analyses (smaller subsamples) and are NOT the source for these paper tables.

### Figures

| Figure | Label | Source Script | Result File |
|--------|-------|---------------|-------------|
| Macro-Session Construction | `fig:session_construction` | `paper/figures/session_diagram.py` | `paper/figures/session_construction.png` |
| Panel Construction Diagram | `fig:panel_construction` | `paper/figures/panel_construction_diagram.py` | `paper/figures/panel_construction_diagram.png` |

---

## Section 7: Position Effects (09-position-effects/*.tex)

### Position Bias (position_bias.tex)

| Table | Label | Source Script | Result File |
|-------|-------|---------------|-------------|
| Position Bias: FE Logit | `tab:position-bias-felogit` | `analysis/position-effects-analysis-R/scripts/01_position_bias_felogit.R` | `analysis/position-effects-analysis-R/results/position_bias_felogit_fixest_round2.txt` |
| Placement Fixed Effects | `tab:placement-fe` | `analysis/position-effects-analysis-R/scripts/01_position_bias_felogit.R` | `analysis/position-effects-analysis-R/results/position_bias_felogit_fixest_round2.txt` |
| Position Bias: Placement 1 | `tab:position-bias-pl1` | `analysis/position-effects-analysis-R/scripts/01_position_bias_felogit.R` | `analysis/position-effects-analysis-R/results/position_bias_felogit_fixest_round2_pl1.txt` |

**Verification**:
- Paper: Quality=0.899 → Result: 0.899488 ✓
- Paper: Rank=-0.010 → Result: -0.010110 ✓
- Paper: Price=-0.016 → Result: -0.015983 ✓
- Paper: CVR=3.453 → Result: 3.453367 ✓
- Paper: P1=1.112, P2=1.448, P3=1.055, P5=0.588 → All match ✓

### Latency Effects (latency_effects.tex)

| Table | Label | Source Script | Result File |
|-------|-------|---------------|-------------|
| Latency Effects | `tab:latency-effects` | `analysis/position-effects-analysis-R/scripts/02_latency_load_vs_dwell.R` | `analysis/position-effects-analysis-R/results/latency_load_vs_dwell_fixest_round2_pl1.txt` |

### Congestion Thresholds (congestion_thresholds.tex)

| Table | Label | Source Script | Result File |
|-------|-------|---------------|-------------|
| Congestion Elasticity | `tab:congestion-elasticity` | `analysis/position-effects-analysis-R/scripts/03_congestion_thresholds.R` | `analysis/position-effects-analysis-R/results/congestion_thresholds_round2.txt` |
| Congestion Summary Statistics | `tab:congestion-summary` | `analysis/position-effects-analysis-R/scripts/03_congestion_thresholds.R` | `analysis/position-effects-analysis-R/results/congestion_thresholds_round2.txt` (SUMMARY STATISTICS section) |

**Verification**:
- Paper: P1 β(log S2)=0.65 → Result: 0.6531 ✓
- Paper: P1 β(log threshold)=0.65 → Result: 0.6485 ✓
- Paper: P1 Mean Bidders=35.2 → Result: 35.2 ✓
- Paper: P1 Median Q1=0.027 → Result: 0.027250 ✓
- Paper: P1 Median S2=0.179 → Result: 0.1792 ✓
- Paper: P1 Median Threshold=6.96 → Result: 6.9617 ✓

**Note**: Python script `analysis/position-effects/eda/scripts/13ap_congestion_thresholds.py` produces equivalent results in `13ap_congestion_thresholds_round2.txt`.

### Fold Effects / Near-Tie Identification (near_tie_identification.tex)

| Table | Label | Source Script | Result File |
|-------|-------|---------------|-------------|
| Exposure and Click by Rank Boundary | `tab:rdd_main` | `analysis/position-effects-analysis-R/scripts/06_uber_near_tie_rdd_eda.R` | `analysis/position-effects-analysis-R/results/uber_near_tie_rdd_eda_round2_pl1.txt` |
| Bandwidth Sensitivity | `tab:rdd_bandwidth` | `analysis/position-effects-analysis-R/scripts/06_uber_near_tie_rdd_eda.R` | `analysis/position-effects-analysis-R/results/uber_near_tie_rdd_eda_round2_pl1.txt` |
| Device Heterogeneity | `tab:rdd_device` | `analysis/position-effects-analysis-R/scripts/04_viewport_fold_rdd.R` | `analysis/position-effects-analysis-R/results/viewport_fold_rdd_round2_pl1_mobile.txt`, `viewport_fold_rdd_round2_pl1_desktop.txt` |

**Verification**:
- Paper: tau=0.02, boundary 2 vs 3, N_pairs=1686 → Result: 1686 ✓
- Paper: tau=0.02, boundary 2 vs 3, N_both_exposed=709 → Result: 709 ✓
- Paper: Balance at tau=0.005, boundary=2 → Result: FAIL ✓
- Paper: Balance at tau=0.010, boundary=2 → Result: PASS ✓

**Additional R Scripts for Fold Effects**:
| Script | Purpose | Result File |
|--------|---------|-------------|
| `05_near_tie_pairs_felogit.R` | LPM within both-exposed pairs | Used for robustness |
| `07_uber_near_tie_rdd_model.R` | Pair-clustered LPM with trends | `uber_near_tie_rdd_model_round2.txt` |

**Note**: Python script `analysis/position-effects/eda/scripts/13as_near_tie_pairs_lpm_all_placements.py` produces supporting analysis in `13as_near_tie_pairs_lpm_all_placements_round2.txt`.

---

## R Scripts Summary (analysis/position-effects-analysis-R/scripts/)

| # | Script | Paper Section | Result File(s) |
|---|--------|---------------|----------------|
| 01 | `01_position_bias_felogit.R` | position_bias.tex | `position_bias_felogit_fixest_round2.txt`, `*_pl1.txt` |
| 02 | `02_latency_load_vs_dwell.R` | latency_effects.tex | `latency_load_vs_dwell_fixest_round2_pl1.txt` |
| 03 | `03_congestion_thresholds.R` | congestion_thresholds.tex | `congestion_thresholds_round2.txt` |
| 04 | `04_viewport_fold_rdd.R` | near_tie_identification.tex | `viewport_fold_rdd_round2_pl1_mobile.txt`, `*_desktop.txt` |
| 05 | `05_near_tie_pairs_felogit.R` | near_tie_identification.tex (robustness) | (supports 06/07) |
| 06 | `06_uber_near_tie_rdd_eda.R` | near_tie_identification.tex | `uber_near_tie_rdd_eda_round2_pl1.txt`, `*_pl2.txt`, `*_pl3.txt`, `*_pl5.txt` |
| 07 | `07_uber_near_tie_rdd_model.R` | near_tie_identification.tex | `uber_near_tie_rdd_model_round2.txt` |

---

## Supporting Python Scripts (Position Effects EDA)

| Script | Purpose | Result File |
|--------|---------|-------------|
| `13ae_felogit_click_quality_rank_vendor.py` | Click model with quality, rank, vendor | `13ae_*.txt` |
| `13af_impression_lpm_quality_rank.py` | Impression LPM model | `13af_impression_lpm_quality_rank_round2.txt` |
| `13ag_felogit_impression_quality_rank_variants.py` | FE logit impression variants | `13ag_felogit_impression_quality_rank_variants_round2.txt` |
| `13ah_felogit_auc_quality_vs_full.py` | AUC comparison | `13ah_auc_quality_vs_full_round2.txt` |
| `13ai_contrib_decomposition.py` | Contribution decomposition | `13ai_contrib_decomposition_round2.txt` |
| `13am_latency_penalty.py` | Latency penalty analysis | `13am_latency_penalty_round2.txt` |
| `13ao_latency_load_vs_dwell.py` | Load vs dwell time | `13ao_latency_load_vs_dwell_round2.txt` |
| `13ap_congestion_thresholds.py` | Congestion analysis | `13ap_congestion_thresholds_round2.txt` |
| `13ar_near_tie_rank2_vs_rank3_eda.py` | Near-tie EDA | `13ar_near_tie_rank2_vs_rank3_eda_round2.txt` |
| `13as_near_tie_pairs_lpm_all_placements.py` | Near-tie LPM all placements | `13as_near_tie_pairs_lpm_all_placements_round2.txt` |

---

## Appendix

### Figures

| Figure | Label | Source Script | Result File |
|--------|-------|---------------|-------------|
| Decay Kernels | `fig:decay_kernels` | `paper/appendix/generate_adstock_plots.py` | `paper/appendix/figures/decay_kernels.png` |
| Adstock Accumulation | `fig:adstock_accumulation` | `paper/appendix/generate_adstock_plots.py` | `paper/appendix/figures/adstock_accumulation.png` |
| Attribution Example | `fig:attribution_example` | `paper/appendix/generate_adstock_plots.py` | `paper/appendix/figures/attribution_example.png` |
| Click Impulse Response | `fig:click_impulse` | Time series analysis | `paper/figures/click_impulse_response.png` |

### TikZ Diagrams (paper/figures/illustrative_diagrams/)

| Figure | Label | Source File |
|--------|-------|-------------|
| Macro-Session Journey | (inline) | `01_macro_session_journey.tex` |
| Activity Bias DAG | (inline) | `03_activity_bias_dag.tex` |
| Auction Process | `fig:auction_process` | `05_auction_process.tex` |
| Session Construction | (inline) | `06_session_construction.tex` |

---

## Key Analysis Directories

| Directory | Purpose | Paper Section |
|-----------|---------|---------------|
| `eda/` | **Main shopping session analysis** | Section 6 |
| `analysis/panel/` | **Vendor panel descriptives** | Section 5 |
| `analysis/staggered-adoption/` | Callaway-Sant'Anna DiD | Section 5 (causal) |
| `analysis/position-effects-analysis-R/` | R/fixest models for paper | Section 7 |
| `analysis/position-effects/eda/` | Python position EDA | Section 7 (supporting) |
| `analysis/position-effects/0_data/` | Raw parquet data files | All position analyses |
| `paper/figures/` | TikZ diagrams, Python plots | All |
| `paper/appendix/` | Supplementary model docs | Appendix |

---

## Directories NOT in Paper

| Directory | Notes |
|-----------|-------|
| `analysis/shopping-episode/` | Alternative session analysis, smaller sample |
| `analysis/shopping-sessions/` | Alternative macro-session, smaller sample |
| `analysis/time-series/` | VAR, ARDL, DFM models |
| `analysis/deep-learning/` | Neural network models |
| `analysis/holdouts/` | RCT analysis (commented out in main.tex) |
| `analysis/collaborative-filtering/` | Recommendation analysis |
| `analysis/optimization/` | Bid optimization theory |
| `analysis/ghostads-simulations/` | Ghost ad simulations |
| `analysis/regression-discontinuity/` | Additional RDD |
| `analysis/causal-attribution/` | Attribution models |

---

## Execution Order

### Vendor Panel Analysis (Section 5)

**Descriptive tables**:
```bash
jupyter execute analysis/panel/vendor_week.ipynb
```

**Causal tables (DiD)**:
```bash
cd analysis/staggered-adoption/scripts
python 01_build_panel.py
python 02_run_eda.py
python 03_callaway_santanna.py
```

### Shopping Session Analysis (Section 6)
```bash
jupyter execute eda/06_macro_sessions.ipynb
jupyter execute eda/07_fixed_effects.ipynb
jupyter execute eda/08_double_ml.ipynb
```

### Position Effects (Section 7)
```bash
cd analysis/position-effects-analysis-R/scripts
Rscript 01_position_bias_felogit.R
Rscript 02_latency_load_vs_dwell.R
Rscript 03_congestion_thresholds.R
Rscript 04_viewport_fold_rdd.R
Rscript 05_near_tie_pairs_felogit.R
Rscript 06_uber_near_tie_rdd_eda.R
Rscript 07_uber_near_tie_rdd_model.R
```

### Paper Figures
```bash
cd paper/figures
python session_diagram.py
python panel_construction_diagram.py

cd paper/appendix
python generate_adstock_plots.py
```

---

## Combined Results Files

| File | Contents |
|------|----------|
| `analysis/staggered-adoption/results/MASTER_RESULTS.txt` | All staggered DiD results |
| `eda/data/fixed_effects_summary.txt` | Main session-level ATE results |
| `eda/data/double_ml_log.txt` | Double ML robustness results |

---

## Notes

1. All paths are relative to repository root (`/Users/pranjal/Code/topsort-incrementality/`)
2. Python scripts output to `.txt` files; one script produces one result file
3. R scripts in `position-effects-analysis-R/` use fixest for high-dimensional fixed effects
4. Manual/descriptive tables are written directly in LaTeX without source scripts
5. Round2 suffix indicates second data pull (expanded sample)
6. **Critical**: Section 5 uses TWO different panels - descriptive (979K obs) vs causal/DiD (846K obs)
7. **Critical**: Section 6 sources from `eda/` directory, NOT `analysis/shopping-episode/` or `analysis/shopping-sessions/`
8. **Critical**: Near-tie/fold effects (Section 7) use R scripts 06 and 07, not script 05
