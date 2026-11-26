# Changelog

## v1.2 (2026-11-26)

Text and evidence. No run was made for this text; the new results come from lab runs made after v1.1. Review items
come from a second LLM review of v1.1.

### Changes from the review

- **Title.** "Kintsugi: Gate Low-Bit Quantization on Tasks, Not on Average Error Alone". A class-level KL table, not a
  task probe, found the repair; the title no longer says more.
- **Temperature.** The abstract and Section 4.2 no longer say that the lead "is in how the two files behave under
  temperature-1 sampling". At temperature 0 the lead is +11 [-5, +28], and one greedy run cannot show whether it is
  smaller: the difference of the two leads is +18 of 417, scenario bootstrap [+0.8, +36.5], but with the design effect
  [-6, +43] (p 0.14), and every single-seed interval holds 0 (E6.1; `scripts/v12_t0.py`).
- **Table 1b** (a description): each arm's temperature-1.0 runs split by its own greedy result (E6.2).
- **Abstract** cut to about 260 words; the version history moved to Appendix B of the paper.
- **Vault 9:** the two early departures are damage that both files share, not damage in the new map.
- **AI use:** the lab found the shared seed stream; the review did not ask for the seed-level statistics.
- **Clean numbers for two doubled shards** (two copies of one job ran into one folder): uniform vs shipped -37
  (p 0.094; v1.1 -40, p 0.077); head off vs shipped +14 (p 0.44; v1.1 +17, p 0.36); head off vs BF16 +73
  (p 6.2e-4); head repair +15 (p 0.46). No verdict changes (E6.3).
- **Inkling 2 Mini is another lineage.** The allocation rule of the 27B model did not carry to it; the method expects
  that. New Section 4.9: readings across compressions of one base.

### New results

- **The mechanism of the lead over BF16 (Section 4.3).** On test_challenge the arms differ almost only at one
  decision: an unasked answer at `complete_task` on a task that needs an action and no answer. Every arm has the
  habit, BF16 included (BF16 258 of 1,668 task-runs, shipped 112). With these runs counted as passes (a post hoc
  class, not a score), the arms are within 697-723 of 834 and shipped and BF16 are level. Greedy decoding lowers
  BF16's rate, which is most of the temperature-0 result (E6.4).
- KL-tools on four seeds: +140 over BF16 (p 3.0e-5), level with the shipped file (E6.6). Three maps at
  temperature 0 (E6.7).
- Reasoning effort xhigh lowers the habit on a targeted set of 70 tasks (3 against 13, p 0.013); one targeted sentence
  at medium does not (10 against 13, p 0.65) (E6.9, E6.10). The coded bias of the format, switched on, raises the
  habit; it stays off (E6.12).
- **The repair split:** the 8 raised projections carry the +70 of KL-tools (+53, p 0.021); the 13 cuts alone are not
  separated (+38, p 0.097). With the band test, an interaction (E6.5).
- **A held candidate:** KL-tools-2 (2.4 % lower KL, every other gate passed) scored 666 against 670 on AppWorld and 492
  against 505 on BFCL multi-turn; held by the rule fixed before its AppWorld run (E6.14).
- A coding-agent audit, as a description (Section 6; E6.16). A gradient map stopped at its gate (E6.13).

### Repository

- New script `scripts/v12_t0.py`; new per-task data for the v1.2 arms in `data/appworld/`.
- `evidence/EVIDENCE.md` section W6 (E6.1-E6.16).

## v1.1 (2026-11-24)

Corrections first, then the new results.

### Corrections to versions 1.0.x

- **The p-values of every AppWorld comparison were overstated.** The AppWorld client sends one seed on every model
  call and the server makes a new sampler from it for each request, so the task-runs of one seed share one random
  stream. The exact sign test over task-runs that versions 1.0.x used treats them as independent. Version 1.1 gives,
  for every AppWorld claim, the difference on each seed, a design-effect p and a cluster-robust 95 % interval
  (Section 4.1; E5.1, E5.28; `scripts/gg_seed.py`, `scripts/gg_seed_tn.py`). The task-level values stay in the older
  evidence rows, marked as such.
- **Claims that do not keep p < 0.05 at the seed level:** the loss of the KL-only winner on the selection seeds of the
  standard split (task-level p 0.019, seed-level 0.13); the lead of 21 task-runs over BF16 on the selection seeds
  (0.0025, 0.057); the public head-to-head of the 27B build against Inkling 2 Mini on seeds 100 + 101 (0.0079, 0.090).
  Versions 1.0.x already reported all three as not replicated or withdrawn.
- **Claims that keep it:** the three tests of the stop rule keep their verdicts at the family alpha 0.0167 (the two
  AppWorld tests: seed-level p 0.0056 and 0.0078); the Inkling build against Unsloth's setup (0.021 and 0.012).
- **The scope sentence of v1.0.3 is replaced.** v1.0.3 said that the method did not find a better bit map than the
  published allocation. The second batch shows more: the method found and repaired damage in context (the KL-tools
  map scores 70 task-runs above the KL-only winner, seed-level p 0.0017), and the repaired map is level with the
  published allocation on agent work at 7.9 % lower KL. It changed free-generation behaviour on one deep recall item,
  so the shipped build keeps the published allocation (L0xRE-V2 by YourHighnessLA).
- **The Inkling headline is now the task result, not the KL ratio.** Versions 1.0.x credited "the same steps" for a
  27 % lower KL than a public 2-bit file. A uniform map at the same bytes keeps 93 % of that gap, so it belongs to the
  codec, the fits and the engine format (Section 4.6; E5.20).
- **The long-chat flag against the KL-tools map is two documents, not the class** (E5.11).
- **The band of MLP up-projections (layers 21-47) of Section 4.1 of v1.0.x** is still the trace of one gate item on one
  map. The held-out recall loss of KL-table B is now located in another group (the down-projections of layers 16-31),
  in that map only (E5.15).
- **LookAtMyQuant:** the measured file is now named exactly (Hugging Face revision 1318b804e687 of 2026-11-21,
  8,874,213,027 B). A newer file exists and is not scored here. Three golden documents are copies of public text that
  is in that project's calibration text; the comparison is given with and without them (E5.24, E5.25). The
  newer file is not yet measured on our golden set. The author's README (commit 92bb52d) lists eight ideas taken from
  the Kintsugi repositories.
- **An error in the project record, corrected before this version:** KL-table B has two cuts against the shipped map,
  not one (E5.14, E5.19).

### New results (second batch)

- At temperature 1.0 a 2.47-bit build scores above BF16 on AppWorld test_challenge on four seeds, and the same file
  in BF16's setup keeps the lead; the setup, the two-stage head, drafting and sharpness are ruled out. At temperature
  0 the lead is +11 of 417 and not significant (p 0.24): BF16 gains about 16 under greedy decoding, the quantized file
  does not. The lead is in how the two files behave under temperature-1 sampling; the mechanism is open (Section 4.2).
- Average KL does not rank bit maps: six maps of the same size score from 586 to 680 of 834 (Section 4.3).
- Nulls, each with its line: the flagged band alone; the uniform and HAWQ-style baselines at the seed level; a
  single-group cause; context length; token class; the head repair on tasks; sharpness; the transfer of the
  down-projection rule to Inkling 2 Mini for agent work (Table 10).
- Inkling 2 Mini with the setup equal: M2 above a public 2-bit file on agent work, not worse than the official
  weights; the down projections carry long-range copy margin, and the public file has more copy margin than M2
  (Section 4.6).
- A limit of every teacher-forced instrument: free-generation behaviour (Section 6). On one deep item the KL-tools
  map thinks for 1,184 characters, the shipped file and BF16 for about 210 (E5.26, E5.30). A forced reading under BF16
  shows that this is not damage: both quantized files leave BF16 at the same early tokens, and after that BF16 prefers
  the long reply; the shipped file's short answer is itself a departure (E5.31). A line on output length misread a
  shared departure. The shipped build keeps the published allocation because the KL-tools map failed a line written
  before the run, not because damage was shown.

### Repository

- New scripts: `gg_seed.py`, `gg_seed_tn.py`, `w2c_score.py`, `w2d_margin.py`, `freegen_summary.py`,
  `free_needles.py`, `gguf_tensor_diff.py`.
- New data: the per-task outcomes of every arm and seed of the second batch (`data/appworld/`), the openings files,
  the W2-C and W2-D margin records, the free-generation records, and the lab result tables in `data/scores/`.
- `evidence/SEED-FIX.md`: a patch proposal for per-call seeds.
- Figures are renumbered 1-7 (new Figure 2; Figures 3, 6 and 7 changed).

## v1.0.3 (2026-11-23)

Scope statement. No result changed.

- The abstract and the contributions now say plainly what the paper does and does not show: the method did not find a better bit map than the published allocation; the paper shows damage that average error hides, an instrument that measures it, and a protocol that kept our own claims honest.

## v1.0.2 (2026-11-23)

Text corrections. No result changed.

- The "recall loss is local" paragraph is rewritten against the held-out test: one gate item traces to a layer band,
  but the band does not explain the held-out loss, and the effect does not add (abstract, introduction,
  contributions, Section 4.1, E1.5). E1.5 now states the exception (V3sH1) with its log pointers.
- Table 1: the KL-group row states its held-out result.
- Section 4.1: the two B1 values the text did not state (KL-only winner and uniform map: slightly more long-range
  margin than the shipped file, inside the refit floor).
- Contributions: the KL-against-task evidence is on the dense model. On the looped MoE model the task probes did not
  change the map choice (as Section 4.4 already said).
- "External review" is now "an adversarial review by an LLM assistant, at the author's request". The AI use section
  states it.

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
