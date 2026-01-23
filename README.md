# Incrementality in Advertising

A collection of causal inference methods for measuring advertising incrementality in online marketplaces.

## Start Here

1. **Presentation** (`docs/presentation-2026/`) - 15-minute overview of methods and findings
2. **Main analysis** (`analysis/shopping-episode/`) - Primary causal inference work
3. **Ghost Ads paper** (`docs/ghostads-paper/`) - Methodology for counterfactual measurement

## Key Findings

Advertising drives exposure but not immediate sales lift. Using staggered adoption (Callaway-Sant'Anna) across 139k vendors, we find clicks and impressions increase when vendors start advertising, but GMV shows no significant change. At the vendor level, clicks correlate with sales (elasticity ~0.64), but this is association—at the user-vendor level, more clicks on a vendor predict *less* spending, suggesting clicks signal browsing/comparison shopping, not purchase intent. Heterogeneity matters: mid-priced vendors and medium-activity vendors see the strongest effects. Top-ranked ads get higher CTR, but middle positions show surprising engagement.

## Maturity Status

| Status | Folders |
|--------|---------|
| **Production-ready** | `shopping-episode/`, `staggered-adoption/`, `panel/`, `time-series/` |
| **Complete analysis** | `holdouts/`, `ghostads-simulations/` |
| **Exploratory** | `regression-discontinuity/`, `sequential-models/`, `causal-attribution/`, `collaborative-filtering/`, `deep-learning/` |
| **Theory/reference** | `optimization/` |

## Repository Structure

```
incrementality-in-advertising/
├── analysis/           # All analysis methods
│   ├── shopping-episode/       # Main causal analysis (choice models, share of voice)
│   ├── panel/                  # Panel fixed effects models
│   ├── time-series/            # VAR and time-series models
│   ├── staggered-adoption/     # Difference-in-differences
│   ├── holdouts/               # Holdout experiment analysis
│   ├── regression-discontinuity/
│   ├── sequential-models/      # Funnel and sequential models
│   ├── causal-attribution/     # Attribution modeling
│   ├── collaborative-filtering/
│   ├── deep-learning/          # Neural network experiments
│   ├── optimization/           # Bid optimization theory
│   └── ghostads-simulations/   # Ghost ads methodology simulations
├── data/               # Data extraction pipelines
│   ├── daily-summaries/        # Daily aggregation scripts
│   └── data-pull/              # Raw data extraction
├── docs/               # Papers and presentations
│   ├── paper/                  # Main academic paper (LaTeX)
│   ├── ghostads-paper/         # Ghost ads methodology paper
│   ├── ghostads-slides/        # Presentation slides
│   └── presentation-2026/      # Recent presentation
└── eda/                # Exploratory data analysis
```

## Data Requirements

This repository does not include data. See `CLAUDE.md` for the complete data schema and table definitions.

Required tables:
- AUCTIONS_USERS (auction/query events)
- AUCTIONS_RESULTS (bid events)
- IMPRESSIONS (promoted product impressions)
- CLICKS (promoted product clicks)
- PURCHASES (all purchases)
- CATALOG (product metadata)

## Key Methods

1. **Ghost Ads** (`docs/ghostads-paper/`, `analysis/ghostads-simulations/`)
   - Counterfactual methodology for measuring ad incrementality

2. **Shopping Episode Analysis** (`analysis/shopping-episode/`)
   - Choice models, semantic halo effects, share of voice analysis

3. **Panel Methods** (`analysis/panel/`, `analysis/staggered-adoption/`)
   - Fixed effects, difference-in-differences

4. **Time Series** (`analysis/time-series/`)
   - VAR, VECM, impulse response functions

## Presentations

- [Ghost Ads Slides](docs/ghostads-slides/main.pdf)
- [February 2026 Presentation](docs/presentation-2026/)

## License

See LICENSE file.
