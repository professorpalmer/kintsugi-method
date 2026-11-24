# The Kintsugi method

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24455575.svg)](https://doi.org/10.5281/zenodo.24455575)

**Find and repair quantization damage where tasks break, not where averages look.**
Paper and evidence by Cary Palmer, version 1.1 (2026-11-24).

- Paper: [`paper/kintsugi-method.pdf`](paper/kintsugi-method.pdf) (source: [`paper/kintsugi-method.md`](paper/kintsugi-method.md))
- Every number in the paper cites a row of [`evidence/EVIDENCE.md`](evidence/EVIDENCE.md) as [E#.#].
- Every claim is judged by a line written before its run: [`evidence/LINES.md`](evidence/LINES.md).
- Related work with one-line differences: [`evidence/RELATED.md`](evidence/RELATED.md); the method step by step:
  [`evidence/ALGORITHM.md`](evidence/ALGORITHM.md).
- A patch proposal for per-call seeds in AppWorld runs: [`evidence/SEED-FIX.md`](evidence/SEED-FIX.md).

**Version 1.1 corrects versions 1.0.x.** The task-runs of one AppWorld seed share one random stream, so the
task-level p-values of versions 1.0.x were too small. Version 1.1 gives seed-level statistics for every AppWorld
claim and adds a second batch of tests. Read [`CHANGELOG.md`](CHANGELOG.md) first: it lists the corrections before
the new results.

The models the paper is about:

| Model | Code and bundles | Weights |
|---|---|---|
| Kintsugi-27B (Qwen3.8-27B at 2.47 bits per weight, one 12 GB GPU) | [github.com/professorpalmer/Kintsugi-27B](https://github.com/professorpalmer/Kintsugi-27B) | [huggingface.co/CaryPalmer/Kintsugi-27B](https://huggingface.co/CaryPalmer/Kintsugi-27B) |
| Kintsugi for Inkling 2 Mini (looped MoE, 9.07 GB) | [github.com/professorpalmer/Kintsugi-Inkling-2-Mini](https://github.com/professorpalmer/Kintsugi-Inkling-2-Mini) (release [v1.0](https://github.com/professorpalmer/Kintsugi-Inkling-2-Mini/releases/tag/v1.0)) | [huggingface.co/CaryPalmer/Kintsugi-Inkling-2-Mini](https://huggingface.co/CaryPalmer/Kintsugi-Inkling-2-Mini) |

## Recompute the statistics and the figures (CPU only, a few minutes)

```
python -m venv .venv
.venv/bin/pip install -r scripts/requirements.txt          # Windows: .venv\Scripts\pip install -r scripts\requirements.txt
.venv/bin/python -X utf8 scripts/reproduce.py              # Windows: .venv\Scripts\python -X utf8 scripts\reproduce.py
```

`reproduce.py` runs the scripts below, writes `results/`, redraws `paper/figs/` and checks that every figure is
byte-identical to the one in this repository (Python 3.12 and the pinned packages of `scripts/requirements.txt`).

| What | Script | Data | Output | Paper |
|---|---|---|---|---|
| **Seed clustering and the seed-level test of every comparison on AppWorld test_challenge and of the Inkling comparisons of the second batch** (per-seed scores and differences, discordant counts, task-level p, design-effect p, cluster-robust 95 % interval, t-test across seeds; opening modes by seed) | `scripts/gg_seed.py` | `data/appworld/` (`t3-*`, `w2-*`, `w2d*`, `w2i*`), `data/appworld/openings/` | `results/gg-seed.txt`, `.json` | Sections 4.1-4.3, 4.6; Tables 1, 2, 6; Figures 2, 3; E5.1, E5.2, E5.4-E5.6, E5.12-E5.14, E5.17, E5.22, E5.23 |
| The same seed-level test for the test_normal comparisons of versions 1.0.x | `scripts/gg_seed_tn.py` | `data/appworld/` | `results/gg-seed-tn.txt`, `.json` | Sections 4.1, 4.5; Table 5; E5.28 |
| AppWorld counts and the task-level statistics of versions 1.0.x (Wilson and paired bootstrap intervals, sign tests, the selection effect) | `scripts/gg_stats.py` | `data/appworld/` | `results/gg-stats.txt` | E1.31, E1.32, E1.36, E3.x, E4.x (task-level values; the paper uses the two rows above for claims) |
| Flip maps of KL-table B on the held-out recall population (W2-C) | `scripts/w2c_score.py --dir data/w2c/out` | `data/w2c/`, `data/b1/`, `data/b1c/` | `results/w2c-result.txt` | Section 4.3; E5.15 |
| Graded recall of the Inkling baselines (W2-D) | `scripts/w2d_margin.py` | `data/margins/w2d/`, `data/margins/gc2/` | `results/w2d-margin-result.txt` | Section 4.6, Table 7; E5.21 |
| Free-generation check: thinking length per item | `scripts/freegen_summary.py` | `data/freegen/` | `results/freegen.txt` | Section 6; E5.26, E5.30 |
| Graded K/V margins on Inkling 2 Mini | `scripts/gc2_score.py` | `data/margins/gc2/` | `results/gc2-result.txt` | Section 4.4; E2.23 |
| Graded margins of the two baselines | `scripts/gb2_score.py` | `data/margins/gb2/` | `results/gb2-result.txt` | Section 4.3; E1.33 |
| B1 held-out recall population test | `scripts/b1/b1_score.py --dir data/b1/out` | `data/b1/` | `results/b1-result.txt` | Section 4.3, Figure 4; E1.34 |
| B1c same-box confirmation | `scripts/b1/b1c_score.py --dir data/b1c/out` | `data/b1c/` | `results/b1c-result.txt` | E1.35 |
| KL comparisons of the first batch | `scripts/kl/kl_docs.py <p1/p2> <base> <map>...` | `data/kl/` | `results/kl.txt` | Table 3; E1.1-E1.3, E1.8, E1.9, E1.28-E1.30 |
| Figures 1-7 | `scripts/make_figs.py` | the above | `paper/figs/` | Figures 1-7 |

### How to check each new number of version 1.1

| Number in the paper | Where to read it after `reproduce.py` |
|---|---|
| Shipped file above BF16 at temperature 1.0: +32, +24, +35, +26; +117 of 1,668; seed-level p 5.1e-4; [+51, +183]; t-test p 0.0014 | `results/gg-seed.txt`, part (2), row "shipped vs BF16 (W2-A + W2-A2)" |
| Temperature 0: 330 vs 319, discordant 41 / 30, p 0.235, +11 [-5, +28] | `data/appworld/w2-pv3acT-s106.csv`, `w2-pbf16T-s106.csv`; table `data/scores/w2t0-temperature-0.txt` |
| The shipped file in BF16's setup: 670 vs 614; p 0.014; [+11, +101]; vs the 12 GB run 0 [-40, +40] | rows "W2-S vs BF16", "W2-S vs shipped" |
| Map scores of Table 2 (680, 671, 637, 630, 627, 610, 586) with per-seed differences, p and intervals | rows "V5tc / S-fd / H1 / U1 / g3 / V5bc / V5cc vs shipped" (names: Appendix A of the paper) |
| KL-tools above the KL-only winner: +70; p 0.0017; [+26, +114] | row "V5tc vs V5bc" |
| Two-stage head off: 687; head repair: 687 vs 670, p 0.41 | rows "noH2 vs shipped", "noH2 vs BF16", "W2-K: noH2 (exact head) vs W2-K (3-bit head)" |
| Design effects 2.24, 2.75, 1.29; same first 8 words 51.2 / 23.8 / 27.9 %; opening modes by seed | `results/gg-seed.txt`, parts (1a)-(1d) |
| Inkling: M2 598 vs 579 (p 0.019, [+3, +35]) and vs the uniform map; harder split 638 / 649 / 604 | rows "Inkling W2-D ...", "Inkling W2-I ..." |
| Standard-split rows of Table 5 and Section 4.5 at the seed level | `results/gg-seed-tn.txt` |
| Held-out recall loss carried by one group: L -0.019, recovery +1.087 (98 %) | `results/w2c-result.txt` |
| Inkling graded recall: uniform map -0.971 (copies -2.728); Unsloth's file +0.865 (copies +2.341) | `results/w2d-margin-result.txt` |
| Free generation: 199-232 characters (BF16; 221 on vault 9); 197-209 (shipped); one item with 1,184 (KL-tools); 3 items each for the two one-group maps | `results/freegen.txt` |

### Numbers that this repository gives as lab result tables (not recomputed here)

The raw rows behind these are large or belong to third parties, so `data/scores/` holds the tables as the lab
scorers wrote them (machine names and private paths removed). `data/scores/gg-seed-lab.txt` and `gg-seed-tn-lab.txt`
are the lab outputs of the two seed-level scripts, for comparison with your `results/`.

| Number | File in `data/scores/` | Paper |
|---|---|---|
| Sharpness of four maps against BF16 (W3-E) | `w3e-sharpness.txt`, `w3e-sharpness.json` | Section 4.2; E5.8 |
| KL by token class for the band-only map (W3-T) | `w3t-token-classes.txt`, `w3t-class-counts.txt` | Section 4.3; E5.16 |
| Setup-artifact check, transcripts, context length (W2-A CHECK, W2-T, W2-B) | `w2a-check.txt`, `w2t-transcripts.txt`, `w2b-context-length.txt` | Sections 4.2, 4.3; E5.3, E5.9, E5.10 |
| Inkling KL of M2, the uniform map and Unsloth's file (W2-D) | `w2d-inkling-kl.txt` | Section 4.6, Table 7; E5.20 |
| Overlap of the golden set with a third party's calibration text; the old LookAtMyQuant file on the clean set (W4-L) | `w4l-overlap.txt`, `w4l-old-file-clean-set.txt` | Section 4.6; E5.25 |
| The task-level tables of the AppWorld scorers of the second batch (with their task-level bootstrap intervals) | `w2a-*.txt`, `w2a2-*.txt`, `w2s-*.txt`, `w2u-*.txt`, `w2f-*.txt`, `w2h-*.txt`, `w2k-*.txt`, `w2d-inkling-appworld.txt`, `w2i-*.txt` | E5.2-E5.6, E5.12-E5.14, E5.17, E5.22, E5.23 |

## What needs a GPU (not needed to check the paper)

These scripts produced the data in `data/`; they talk to a running llama.cpp-type server (`--base URL`) of the
engines in the model repositories above.

| Script | Produces | Needs |
|---|---|---|
| `scripts/b1/b1_probe.py` | one B1 document's margin records | the Kintsugi-27B engine and a model file (12 GB GPU, 262,144-token window) |
| `scripts/b1/b1_gen.py` | `data/b1/population.jsonl.gz` (CPU only) | the CPython 3.12.10 `Lib` folder and the Qwen3.8 `tokenizer.json`; with them it rebuilds the same file (sha256 bc46d5da...aba337) |
| `scripts/p1.py`, `scripts/margin.py` | the 44-item recall margins (G-B2) | the Kintsugi-27B engine |
| `scripts/ink_margin.py` | the Inkling graded margins (G-C2, W2-D) | the Kintsugi for Inkling 2 Mini engine |
| `scripts/free_needles.py` | the free-generation records of `data/freegen/` | the Kintsugi-27B engine and a model file |
| `scripts/gguf_tensor_diff.py` (with `scripts/ggufio.py`; CPU only) | the list of tensors that differ between two GGUF files | two model files; use it to read a bit map from the files (E5.19) |
| `scripts/bench/` | the recall gate items (needles, code copies) | used by the probes above; the copy corpus is `data/bench/pycode.raw` |
| `scripts/kl/kld_summary.py` | KL from the scorer's row files | the row files and the quant-challenge golden set (`QC_ROOT`); `kl_docs.py` gives the same numbers from `data/kl/` |

AppWorld scores come from the AppWorld harness (not included); `data/appworld/` holds the outcome of every task-run.

## Licences

- Paper text, figures and the evidence files: **CC BY 4.0** (`LICENSES/CC-BY-4.0.md`).
- Code in `scripts/`: **MIT** (`LICENSE`).
- Data: see `LICENSES/DATA.md`. In short: AppWorld results are task ids and pass / fail only (no task content, no test
  assertions), under AppWorld's terms; the openings files hold one word and one hash per task-run; text derived from
  CPython (`data/b1/population.jsonl.gz`, `data/bench/pycode.raw`) is under the PSF License 2.0
  (`LICENSES/PSF-2.0.txt`); KL sums are our measurements on the CC BY 4.0 golden set of the Qwen3.8-27B quantization
  challenge (the golden set is not redistributed here).

## Cite

Palmer, C. (2026). *Kintsugi: Find and Repair Quantization Damage Where Tasks Break, Not Where Averages Look*. Zenodo.
https://doi.org/10.5281/zenodo.24455575 (machine-readable: `CITATION.cff`; Zenodo gives each version its own DOI).
