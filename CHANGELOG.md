# Changelog

## v1.0.1 (2026-11-23)

Presentation fixes: links, figure legends, removed editing notes; no change to any result.

- Links in the paper use short link text and break at the margin (no line runs past it).
- Figures 4 and 5: legends moved out of the plot area; Figure 4(b): shorter axis label.
- Removed editing notes and draft-status text from the paper; references one per paragraph.
- Figure 3 (B1) gets its caption back (a missing blank line in v1.0 rendered it without one); figures are numbered 1-6 in order.

## v1.0 (2026-11-23)

Zenodo DOI: 10.5281/zenodo.24455576.

First public version, published with Kintsugi for Inkling 2 Mini.

- Paper (Markdown source and PDF), six figures.
- Evidence table (`evidence/EVIDENCE.md`), every pre-written line with its verdict (`evidence/LINES.md`), related work,
  the method.
- Data to recompute every statistic and figure on a CPU, and the scripts that recompute them (`scripts/reproduce.py`).
- All three tests of the stop rule are in (B1 with its same-box confirmation, the naive file on fresh seeds, AppWorld
  test_challenge).
