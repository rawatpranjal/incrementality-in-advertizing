# Staggered Adoption Archive

## Overview

Archived/legacy versions of staggered adoption analysis scripts. These represent earlier iterations of the methodology before refinement. Preserved for historical reference and reproducibility of prior results.

## Status

**ARCHIVED** — For current staggered adoption methodology, see `staggered-adoption/` folder.

## Historical Contents

| File | Original Purpose |
|------|------------------|
| `01_build_panel.py` | Initial panel construction |
| `02_run_eda.py` | Exploratory data analysis |
| `03_estimate_models.py` | Model estimation |
| `panel_eda_writeup.py` | EDA documentation |
| `callaway_santanna_analysis.py` | Callaway-Sant'Anna estimator implementation |
| `claude.md` | Original project notes |

## Method Context

Staggered adoption designs exploit variation in timing of treatment adoption across units. The Callaway-Sant'Anna estimator provides heterogeneity-robust estimates under staggered treatment timing, avoiding negative weighting issues in traditional two-way fixed effects.

## Connections

- Superseded by `staggered-adoption/` (current methodology)
- Related to `panel/` for fixed effects approaches
- See `staggered-adoption/README.md` for active documentation
