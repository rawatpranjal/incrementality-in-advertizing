# Analysis Methods

This folder contains all incrementality analysis methods organized by approach.

## Folders

| Folder | Method | Status |
|--------|--------|--------|
| `shopping-episode/` | Choice models, SOV, semantic halo | Primary |
| `panel/` | Panel fixed effects | Complete |
| `time-series/` | VAR, VECM, IRF | Complete |
| `staggered-adoption/` | Diff-in-diff | Complete |
| `holdouts/` | Holdout experiments | Complete |
| `regression-discontinuity/` | RD designs | Exploratory |
| `sequential-models/` | Funnel models | Exploratory |
| `causal-attribution/` | Attribution | Exploratory |
| `collaborative-filtering/` | CF methods | Exploratory |
| `deep-learning/` | DNN experiments | Exploratory |
| `optimization/` | Bid optimization | Theory |
| `ghostads-simulations/` | Ghost ads sims | Supporting |

## Data Requirements

All analysis folders expect data in their local `data/` subfolder. See main `CLAUDE.md` for schema.
