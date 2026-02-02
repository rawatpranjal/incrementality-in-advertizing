# Position Effects Analysis

## Directory Index

```
position-effects/
├── 0_data/                     # Raw data files
├── 1_data_pull/                # Data extraction scripts
│
├── click_models/               # Click modeling (13 scripts)
│   ├── classic/                # IR literature models
│   │   └── scripts/
│   │       ├── 05_pbm_verification.py      # Position-Based Model
│   │       ├── 06_dbn_estimation.py        # Dynamic Bayesian Network
│   │       ├── 07_sdbn_estimation.py       # Simplified DBN (cascade)
│   │       └── 12_ubm_estimation.py        # User Browsing Model
│   │
│   ├── feature_based/          # Discriminative models with features
│   │   └── scripts/
│   │       ├── 11_feature_click_model.py   # PBM + product features (neural)
│   │       ├── 15_gcm_feature_model.py     # General Click Model (logistic)
│   │       ├── 16_ctr_feature_importance.py # XGBoost CTR with importance
│   │       └── 17_ctr_nlp_features.py      # CTR with NLP features
│   │
│   ├── spatial/                # Grid/viewport-aware models
│   │   └── scripts/
│   │       └── 14_grid_pbm.py              # Grid Position-Based Model
│   │
│   ├── microecon/              # Behavioral economics hypotheses
│   │   └── scripts/
│   │       └── 13_viewport_models.py       # Price anchor, attention gate, vampire effect
│   │
│   ├── comparison/             # Model comparison & benchmarking
│   │   └── scripts/
│   │       ├── 08_model_comparison.py      # PBM vs DBN vs SDBN
│   │       └── 13_model_comparison_full.py # All models head-to-head
│   │
│   └── data_prep/              # Data preparation
│       └── scripts/
│           └── 04_click_model_data_prep.py # Session formatting
│
├── eda/                        # Exploratory Data Analysis (13 scripts)
│   ├── data_quality/           # Field coverage, inventory
│   │   └── scripts/
│   │       └── 09_data_inventory.py
│   │
│   ├── position/               # Position effects exploration
│   │   └── scripts/
│   │       ├── 02_position_effects_eda.py  # Method viability
│   │       ├── 03_causal_eda.py            # R-squared diagnostics
│   │       ├── 03_hypothesis_tests.py      # Ranking mechanics
│   │       └── 04_display_position_eda.py  # Display vs bid rank
│   │
│   ├── placement/              # Placement characterization
│   │   └── scripts/
│   │       ├── 10_placement_eda.py
│   │       ├── 10_placement_verification.py
│   │       ├── 11_placement_mapping_report.py
│   │       ├── 12_p1_vs_p3_comparison.py
│   │       ├── 12_search_pagination_evidence.py
│   │       ├── 13_placement_page_mapping.py
│   │       └── 14_cross_placement_cofiring.py
│   │
│   └── session/                # User journey exploration
│       └── scripts/
│           └── 12_session_raw_logs.py
│
├── iv_models/                  # Instrumental Variables (2 scripts)
│   └── scripts/
│       ├── 10_auction_pressure.py
│       └── 11_iv_strategies.py
│
├── iv-analysis/                # Discrete choice IV analysis
│   └── 14_discrete_choice.py
│
├── nlp_models/                 # Text embeddings pipeline (3 scripts)
│   └── scripts/
│       ├── 05_catalog_nlp_eda.py
│       ├── 06_nlp_position_models.py
│       └── 07_nlp_followup.py
│
├── rdd_models/                 # Regression Discontinuity (1 script)
│   └── scripts/
│       └── 05_causal_models.py
│
├── robustness/                 # Sensitivity analysis (1 script)
│   └── scripts/
│       └── 07_sensitivity_analysis.py
│
├── docs/                       # Documentation
├── papers/                     # Reference papers
├── archive/                    # Deprecated scripts
└── FINAL_REPORT.md
```

## Script Counts by Category

| Category | Subfolder | Scripts | Focus |
|----------|-----------|---------|-------|
| click_models | classic | 4 | IR literature models (PBM, DBN, SDBN, UBM) |
| click_models | feature_based | 4 | ML feature models |
| click_models | spatial | 1 | Grid layout |
| click_models | microecon | 1 | Price anchor, attention gate, vampire effect |
| click_models | comparison | 2 | Benchmarking |
| click_models | data_prep | 1 | Data formatting |
| eda | data_quality | 1 | Field coverage |
| eda | position | 4 | Position effects |
| eda | placement | 7 | Placement types |
| eda | session | 1 | User journey exploration |
| iv_models | - | 2 | IV identification |
| iv-analysis | - | 1 | Discrete choice |
| nlp_models | - | 3 | Text features |
| rdd_models | - | 1 | RDD estimation |
| robustness | - | 1 | Sensitivity |
| **Total** | | **34** | |

## Convention

- Each `.py` script produces one `.txt` result file in the corresponding `results/` folder
- Run scripts via CLI: `python script.py > results/script.txt 2>&1`
