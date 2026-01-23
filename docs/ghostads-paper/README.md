# Ghost Ads Methodology Document

This directory contains a comprehensive technical note on the ghost ads methodology for measuring advertising incrementality.

## File Structure

- `main.tex` - Main document that compiles all sections
- `introduction.tex` - The incrementality problem with DAG, numerical confounding example, and confounders table
- `approaches.tex` - Traditional experimental approaches (ITT, PSA, placebo ads)
- `ghostads.tex` - Core ghost ads methodology with mathematical formulation
- `example.tex` - Implementation example using marketplace data schema with IS_GHOST_AD flag
- `extensions.tex` - Extensions to heterogeneous effects, panel methods, and practical considerations
- `references.bib` - Bibliography

## Compilation

```bash
cd /Users/pranjal/Code/marketplace-incrementality/ghostads/latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Key Features

1. **Causal DAG** showing confounding structure with unobserved user intent
2. **Numerical example** demonstrating 61.2pp bias when true effect is 5pp
3. **Comprehensive confounders table** from marketplace data
4. **Mathematical formulation** of ITT, PSA, and ghost ads estimators
5. **Surgical integration** showing how to add IS_GHOST_AD flag to existing schema
6. **SQL queries** for extracting treatment and control groups
7. **Complete worked example** with fake but realistic marketplace data
8. **Statistical inference** with standard errors and confidence intervals

## Writing Style

- Academic tone matching the main paper
- No textbf, bullets, or paragraph headers
- Footnotes for nuance
- Proper citations to Johnson, Lewis, and Nubbemeyer (2017)
- Written for PhD economist/ML engineer audience
