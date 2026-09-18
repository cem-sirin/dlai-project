# Report

**[Mackerel Road: Learning Neural Game Mechanics Without Transition Targets](main.pdf)**

- `main.tex`: two-page scientific report and AI-use statement.
- `appendix.tex`: game specification, objectives, protocol, complete results, and limitations.
- `results.tex` and `*-table.tex`: numerical content generated from measured results.
- `figures/`: plots and the generated route-map illustration.
- `references.bib`: cited literature.
- `dlaiml2026.sty`, `dlaiml2026.bst`, `fancyhdr.sty`: unmodified official course template files.

See the [root README](../README.md) to regenerate figures and tables. Build the
PDF from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The scientific main text occupies pages 1–2. The AI-use statement, references,
and appendix are outside the course's main-text page limit.

Remove intermediate build files while keeping the PDF:

```bash
latexmk -c
```
