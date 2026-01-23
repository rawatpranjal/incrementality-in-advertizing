# Repository Reorganization for Handoff

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize the incrementality research repository into a clean, professional structure suitable for GitHub handoff as "incrementality-in-advertising".

**Architecture:** Restructure into four top-level folders: `analysis/` (all methods with subfolders), `data/` (extraction pipelines), `docs/` (papers, slides, presentations), `eda/` (exploratory notebooks). Remove all "marketplace" references, clean junk files, add READMEs.

**Tech Stack:** Bash for file operations, sed/grep for text replacement, Git for version control.

---

## Pre-requisites

- ~5.7GB of data folders will be excluded from git (already in .gitignore via `*.csv`, etc.)
- 51 files contain "marketplace" references to scrub
- Backup folders and venv to be deleted
- Archive folders to be kept (contain historical work)

---

### Task 1: Clean Up Junk Files

**Files:**
- Delete: `venv/` (entire folder)
- Delete: `temp/` (entire folder)
- Delete: `ghostads/latex_backup_*` (5 backup folders)
- Delete: `ghostads/slides_backup_*` (3 backup folders)
- Delete: `ghostads/temp/`
- Delete: All `__pycache__/` folders
- Delete: All `.DS_Store` files
- Delete: Root-level junk: `main.aux`, `main.log`, `main.out`, `main.pdf`, `texput.log`, `eda.md`

**Step 1: Delete venv and temp folders**

```bash
rm -rf venv temp
```

**Step 2: Delete ghostads backup folders**

```bash
rm -rf ghostads/latex_backup_* ghostads/slides_backup_* ghostads/temp
```

**Step 3: Delete pycache and DS_Store**

```bash
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name ".DS_Store" -delete
```

**Step 4: Delete root-level junk files**

```bash
rm -f main.aux main.log main.out main.pdf texput.log eda.md
```

**Step 5: Commit cleanup**

```bash
git add -A
git commit -m "chore: clean up junk files (venv, temp, backups, pycache)"
```

---

### Task 2: Create New Folder Structure

**Files:**
- Create: `analysis/` directory
- Create: `data/` directory
- Preserve: `docs/` directory (already exists)
- Preserve: `eda/` directory (already exists)

**Step 1: Create top-level directories**

```bash
mkdir -p analysis data
```

**Step 2: Commit structure**

```bash
git add analysis data
git commit -m "chore: create analysis/ and data/ directories" --allow-empty
```

---

### Task 3: Move Analysis Folders

**Files:**
- Move: `shopping-episode/` -> `analysis/shopping-episode/`
- Move: `panel/` -> `analysis/panel/`
- Move: `time-series/` -> `analysis/time-series/`
- Move: `staggered-adoption-final/` -> `analysis/staggered-adoption/`
- Move: `staggered-adoption-archive/` -> `analysis/staggered-adoption-archive/`
- Move: `holdouts/` -> `analysis/holdouts/`
- Move: `rd/` -> `analysis/regression-discontinuity/`
- Move: `sequential-models/` -> `analysis/sequential-models/`
- Move: `causal-attribution/` -> `analysis/causal-attribution/`
- Move: `collab-filtering/` -> `analysis/collaborative-filtering/`
- Move: `dnn/` -> `analysis/deep-learning/`
- Move: `ghostads/simulations/` -> `analysis/ghostads-simulations/`
- Move: `optimization/` -> `analysis/optimization/`
- Move: `shopping-sessions/` -> `analysis/shopping-sessions/`
- Move: `appendices/` -> `analysis/appendices/`

**Step 1: Move main analysis folders**

```bash
git mv shopping-episode analysis/
git mv panel analysis/
git mv time-series analysis/
git mv staggered-adoption-final analysis/staggered-adoption
git mv staggered-adoption-archive analysis/
git mv holdouts analysis/
git mv rd analysis/regression-discontinuity
git mv sequential-models analysis/
git mv causal-attribution analysis/
git mv collab-filtering analysis/collaborative-filtering
git mv dnn analysis/deep-learning
git mv optimization analysis/
git mv shopping-sessions analysis/
git mv appendices analysis/
```

**Step 2: Move ghostads simulations separately**

```bash
git mv ghostads/simulations analysis/ghostads-simulations
```

**Step 3: Commit moves**

```bash
git commit -m "refactor: move analysis folders to analysis/"
```

---

### Task 4: Move Data Extraction Folders

**Files:**
- Move: `daily_summaries/` -> `data/daily-summaries/`
- Move: `data_pull/` -> `data/data-pull/`

**Step 1: Move data folders**

```bash
git mv daily_summaries data/daily-summaries
git mv data_pull data/data-pull
```

**Step 2: Commit moves**

```bash
git commit -m "refactor: move data extraction folders to data/"
```

---

### Task 5: Move Documentation Folders

**Files:**
- Move: `latex/` -> `docs/paper/`
- Move: `ghostads/latex/` -> `docs/ghostads-paper/`
- Move: `ghostads/slides/` -> `docs/ghostads-slides/`
- Move: `presentation_feb_2026/` -> `docs/presentation-2026/`
- Delete: `ghostads/` (now empty except for docs subfolder)
- Keep: `info/` -> `docs/info/`

**Step 1: Move latex and presentations**

```bash
git mv latex docs/paper
git mv ghostads/latex docs/ghostads-paper
git mv ghostads/slides docs/ghostads-slides
git mv presentation_feb_2026 docs/presentation-2026
git mv info docs/
```

**Step 2: Remove empty ghostads folder**

```bash
rm -rf ghostads
```

**Step 3: Commit moves**

```bash
git commit -m "refactor: move documentation to docs/"
```

---

### Task 6: Remove marketplace References

**Files:**
- Modify: All 51 files containing "marketplace" (case-insensitive)

**Step 1: Find and list all marketplace references**

```bash
grep -ril "marketplace" --include="*.py" --include="*.ipynb" --include="*.tex" --include="*.md" . | grep -v ".git"
```

**Step 2: Replace marketplace references with generic terms**

For Python/notebook files, replace connection strings and comments:
```bash
# Replace "marketplace" with "marketplace" in all text files
find . -type f \( -name "*.py" -o -name "*.ipynb" -o -name "*.tex" -o -name "*.md" \) -exec sed -i '' 's/marketplace/marketplace/gi' {} +
find . -type f \( -name "*.py" -o -name "*.ipynb" -o -name "*.tex" -o -name "*.md" \) -exec sed -i '' 's/marketplace/Marketplace/g' {} +
```

**Step 3: Verify no marketplace references remain**

```bash
grep -ri "marketplace" --include="*.py" --include="*.ipynb" --include="*.tex" --include="*.md" . | grep -v ".git" | wc -l
```

Expected: 0

**Step 4: Commit scrubbing**

```bash
git add -A
git commit -m "chore: remove proprietary references"
```

---

### Task 7: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` - update paths, remove marketplace references

**Step 1: Update CLAUDE.md to reflect new structure**

Update folder references from old paths to new paths:
- `shopping-episode/` -> `analysis/shopping-episode/`
- `daily_summaries/` -> `data/daily-summaries/`
- etc.

**Step 2: Commit CLAUDE.md update**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for new folder structure"
```

---

### Task 8: Update .gitignore

**Files:**
- Modify: `.gitignore`

**Step 1: Add data folder exclusions**

Add to .gitignore:
```
# Data files (too large for git)
**/data/*.parquet
**/data/*.pkl
**/data/*.feather
**/results/*.png
**/results/*.pdf
```

**Step 2: Commit gitignore update**

```bash
git add .gitignore
git commit -m "chore: update gitignore for data exclusions"
```

---

### Task 9: Create Main README.md

**Files:**
- Modify: `README.md` (overwrite existing)

**Step 1: Write comprehensive README**

```markdown
# Incrementality in Advertising

A collection of causal inference methods for measuring advertising incrementality in online marketplaces.

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
```

**Step 2: Commit README**

```bash
git add README.md
git commit -m "docs: update main README for reorganized structure"
```

---

### Task 10: Create Folder READMEs

**Files:**
- Create: `analysis/README.md`
- Create: `data/README.md`
- Create: `docs/README.md`
- Create: `eda/README.md`

**Step 1: Create analysis/README.md**

```markdown
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
```

**Step 2: Create data/README.md**

```markdown
# Data Extraction

Scripts for extracting and aggregating data from the marketplace database.

## Folders

- `daily-summaries/`: Daily aggregation notebooks for auctions, impressions, clicks, purchases
- `data-pull/`: Raw data extraction and funnel analysis

## Output

Data files are not committed to git due to size. Expected outputs:
- Parquet files in each `data/` subfolder
- CSV summaries

## Schema

See `CLAUDE.md` in repository root for complete data dictionary.
```

**Step 3: Create docs/README.md**

```markdown
# Documentation

Academic papers and presentation materials.

## Folders

- `paper/`: Main incrementality paper (LaTeX)
- `ghostads-paper/`: Ghost ads methodology paper
- `ghostads-slides/`: Beamer presentation slides
- `presentation-2026/`: Recent presentation materials
- `info/`: Background information

## Building LaTeX

```bash
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
```

**Step 4: Create eda/README.md**

```markdown
# Exploratory Data Analysis

Initial data exploration and analysis notebooks.

## Notebooks

Jupyter notebooks for understanding the data structure, distributions, and relationships.

## Data

EDA data files are not committed. Run data extraction from `data/` folder first.
```

**Step 5: Commit all READMEs**

```bash
git add analysis/README.md data/README.md docs/README.md eda/README.md
git commit -m "docs: add README files to all main folders"
```

---

### Task 11: Final Verification and Commit

**Step 1: Verify structure**

```bash
ls -la
ls -la analysis/
ls -la data/
ls -la docs/
ls -la eda/
```

**Step 2: Verify no marketplace references**

```bash
grep -ri "marketplace" . --include="*.py" --include="*.ipynb" --include="*.tex" --include="*.md" | grep -v ".git" | wc -l
```

Expected: 0

**Step 3: Check git status**

```bash
git status
```

**Step 4: Final commit if needed**

```bash
git add -A
git commit -m "chore: final cleanup for handoff"
```

**Step 5: Create zip for handoff**

```bash
cd .. && zip -r incrementality-in-advertising.zip marketplace-incrementality -x "*.git*" -x "*data/*.parquet" -x "*data/*.pkl" -x "*data/*.csv" -x "*data/*.feather" -x "*.DS_Store"
```

---

## Summary

After completing all tasks:
- Repository reorganized into `analysis/`, `data/`, `docs/`, `eda/`
- All "marketplace" references removed
- Junk files cleaned
- READMEs added to each main folder
- Ready for handoff as zip file
