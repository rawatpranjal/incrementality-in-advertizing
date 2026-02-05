
# Illustrative Diagrams

This folder contains the source code for the illustrative diagrams used in the paper.

## Requirements

- **LaTeX**: `pdflatex` with `tikz` and `standalone` packages.
- **R**: `Rscript` with `ggplot2` and `grid` packages.

## How to Compile

### TikZ Diagrams (LaTeX)
To compile the TikZ diagrams into PDFs:

```bash
pdflatex 01_macro_session_journey.tex
pdflatex 03_activity_bias_dag.tex
pdflatex 05_auction_process.tex
```

This will output the `.pdf` files in the current directory (`paper/figures/illustrative_diagrams/`).

### R Diagrams (ggplot2)
To generate the R plots:

```bash
Rscript 02_mobile_viewport.R
Rscript 04_dwell_vs_load.R
```

These scripts are configured to save the PDFs directly to `paper/figures/`.

## List of Diagrams

1.  **01_macro_session_journey.tex**: Flowchart of the user session (Search -> Click -> Product -> Return).
2.  **02_mobile_viewport.R**: Schematic of "Above the Fold" vs "Below the Fold" on a mobile screen.
3.  **03_activity_bias_dag.tex**: Causal diagram showing the confounding effect of User Intent.
4.  **04_dwell_vs_load.R**: Timeline distinguishing "Page Load" time from "Dwell" time.
5.  **05_auction_process.tex**: Detailed flow from Query to Revenue.
