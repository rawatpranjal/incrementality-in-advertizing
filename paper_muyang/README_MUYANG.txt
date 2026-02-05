README for Muyang
=================

This folder (paper_muyang/) is a shared copy of the LaTeX paper for
collaborative editing. It is synced via Dropbox and tracked in git.

Folder Structure
----------------
main.tex            -- Master document; compiles the full paper
references.bib      -- BibTeX references

01-introduction/    -- Section 1: Introduction
02-literature/      -- Section 2: Literature Review
03-platform/        -- Section 3: Platform Description
04-data/            -- Section 4: Data
05-sessions/        -- Section 5: Shopping Sessions Analysis
06-event-sampling/  -- Section 6: Event Sampling
07-weekly-panels/   -- Section 7: Weekly Panels
08-conclusion/      -- Section 8: Conclusion
09-position-effects/-- Section 9: Position/Fold Effects
appendix/           -- Appendix material
figures/            -- All figures and diagrams
scripts/            -- Helper scripts (e.g., diagram generation)

How to Compile
--------------
From this directory:
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex

How to Contribute
-----------------
1. Edit section .tex files directly (e.g., 05-sessions/funnel_analysis_v2.tex).
2. Add references to references.bib.
3. Figures go in figures/.
4. Commit and push via git, or just save in Dropbox and Pranjal will commit.

Notes
-----
- The canonical analysis code lives in main_analysis/ at the repo root.
- The original paper/ folder is Pranjal's working copy; this folder is for
  joint editing. Both start identical; divergence is expected and fine.
- If you have questions, message Pranjal.
