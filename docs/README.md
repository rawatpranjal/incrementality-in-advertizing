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
