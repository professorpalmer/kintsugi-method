# The Kintsugi method

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24455575.svg)](https://doi.org/10.5281/zenodo.24455575)

**Find and repair quantization damage where tasks break, not where averages look.**
Paper and evidence by Cary Palmer, version 1.0.1 (2026-11-23).

- Paper: [`paper/kintsugi-method.pdf`](paper/kintsugi-method.pdf) (source: [`paper/kintsugi-method.md`](paper/kintsugi-method.md))
- Every number in the paper cites a row of [`evidence/EVIDENCE.md`](evidence/EVIDENCE.md) as [E#.#].
- Every claim is judged by a line written before its run: [`evidence/LINES.md`](evidence/LINES.md).
- Related work with one-line differences: [`evidence/RELATED.md`](evidence/RELATED.md); the method step by step:
  [`evidence/ALGORITHM.md`](evidence/ALGORITHM.md).

The models the paper is about:

| Model | Code and bundles | Weights |
|---|---|---|
| Kintsugi-27B (Qwen3.8-27B at 2.47 bits per weight, one 12 GB GPU) | [github.com/professorpalmer/Kintsugi-27B](https://github.com/professorpalmer/Kintsugi-27B) | [huggingface.co/CaryPalmer/Kintsugi-27B](https://huggingface.co/CaryPalmer/Kintsugi-27B) |
| Kintsugi for Inkling 2 Mini (looped MoE, 9.07 GB) | [github.com/professorpalmer/Kintsugi-Inkling-2-Mini](https://github.com/professorpalmer/Kintsugi-Inkling-2-Mini) (release [v1.0](https://github.com/professorpalmer/Kintsugi-Inkling-2-Mini/releases/tag/v1.0)) | [huggingface.co/CaryPalmer/Kintsugi-Inkling-2-Mini](https://huggingface.co/CaryPalmer/Kintsugi-Inkling-2-Mini) |

## Recompute every statistic and figure (CPU only, a few minutes)

```
python -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt          # Windows: .venv\Scripts\pip install -r scripts\requirements.txt
.venv/bin/python -X utf8 scripts/reproduce.py              # Windows: .venv\Scripts\python -X utf8 scripts\reproduce.py
```

`reproduce.py` runs the scripts below, writes `results/`, redraws `paper/figs/` and checks that every figure is
byte-identical to the one in this repository (Python 3.12 and the pinned packages of `scripts/requirements.txt`).

| What | Script | Data | Output | Paper |
|---|---|---|---|---|
| AppWorld counts, Wilson and paired bootstrap intervals, sign tests, the selection effect | `scripts/gg_stats.py` | `data/appworld/` | `results/gg-stats.txt` | Tables 1, 3, 4; E1.31, E1.32, E1.36 (test_challenge), E3.x, E4.x |
| Graded K/V margins on Inkling 2 Mini | `scripts/gc2_score.py` | `data/margins/gc2/` | `results/gc2-result.txt` | Section 4.2; E2.23 |
| Graded margins of the two baselines | `scripts/gb2_score.py` | `data/margins/gb2/` | `results/gb2-result.txt` | Section 4.1; E1.33 |
| B1 held-out recall population test | `scripts/b1/b1_score.py --dir data/b1/out` | `data/b1/` | `results/b1-result.txt` | Section 4.1, Figure 3; E1.34 |
| B1c same-box confirmation | `scripts/b1/b1c_score.py --dir data/b1c/out` | `data/b1c/` | `results/b1c-result.txt` | E1.35 |
| KL comparisons | `scripts/kl/kl_docs.py <p1/p2> <base> <map>...` | `data/kl/` | `results/kl.txt` | Table 1; E1.1-E1.3, E1.8, E1.9, E1.28-E1.30 |
| Figures 1-6 | `scripts/make_figs.py` | the above | `paper/figs/` | Figures 1-6 |

## What needs a GPU (not needed to check the paper)

These scripts produced the data in `data/`; they talk to a running llama.cpp-type server (`--base URL`) of the
engines in the model repositories above.

| Script | Produces | Needs |
|---|---|---|
| `scripts/b1/b1_probe.py` | one B1 document's margin records | the Kintsugi-27B engine and a model file (12 GB GPU, 262,144-token window) |
| `scripts/b1/b1_gen.py` | `data/b1/population.jsonl.gz` (CPU only) | the CPython 3.12.10 `Lib` folder and the Qwen3.8 `tokenizer.json`; with them it rebuilds the same file (sha256 bc46d5da...aba337) |
| `scripts/p1.py`, `scripts/margin.py` | the 44-item recall margins (G-B2) | the Kintsugi-27B engine |
| `scripts/ink_margin.py` | the Inkling graded K/V margins (G-C2) | the Kintsugi for Inkling 2 Mini engine |
| `scripts/bench/` | the recall gate items (needles, code copies) | used by the probes above; the copy corpus is `data/bench/pycode.raw` |
| `scripts/kl/kld_summary.py` | KL from the scorer's row files | the row files and the quant-challenge golden set (`QC_ROOT`); `kl_docs.py` gives the same numbers from `data/kl/` |

AppWorld scores come from the AppWorld harness (not included); `data/appworld/` holds the outcome of every task-run.

## Licences

- Paper text, figures and the evidence files: **CC BY 4.0** (`LICENSES/CC-BY-4.0.md`).
- Code in `scripts/`: **MIT** (`LICENSE`).
- Data: see `LICENSES/DATA.md`. In short: AppWorld results are task ids and pass / fail only (no task content, no test
  assertions), under AppWorld's terms; text derived from CPython (`data/b1/population.jsonl.gz`, `data/bench/pycode.raw`)
  is under the PSF License 2.0 (`LICENSES/PSF-2.0.txt`); KL sums are our measurements on the CC BY 4.0 golden set of
  the Qwen3.8-27B quantization challenge (the golden set is not redistributed here).

## Cite

Palmer, C. (2026). *Kintsugi: Find and Repair Quantization Damage Where Tasks Break, Not Where Averages Look*. Zenodo.
https://doi.org/10.5281/zenodo.24455575 (machine-readable: `CITATION.cff`; Zenodo gives each version its own DOI).
