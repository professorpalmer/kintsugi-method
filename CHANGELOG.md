# Changelog

## v1.3 (2026-11-27)

The central result is now decision-point habits (new Section 4.10). The results come from studies made after v1.2.

### New results

- **A procedure in four steps:** detect a habit by a split by turn type, freeze one history and replay the deciding
  call, locate it by tensor copies, repair. This version does the first three. It states that the fourth is open: no
  build of ours was made to remove a habit.
- **The return turn of BFCL miss_func (Qwen3.8-27B).** A candidate map has level totals on two agent benchmarks and
  misses the returned tool more often (48 of 198 against 28 of 199; found after the run). The replay of the one
  deciding call repeats the gap; two miss rules are given everywhere (E7.1, E7.2).
- **The mechanism:** in most misses the model says that the tool is absent, although the updated tool list is in its
  prompt (keyword counts; E7.3). Not explained by prompt length.
- **Levers on the frozen turn:** three wordings; stripping the thinking of earlier turns removes 62 % (rule 1) and
  40 % (rule 2) of the shipped file's misses; effort xhigh; sampling on three seeds (E7.5-E7.7).
- **The parent check:** on one 80 GB card in one setup at temperature 0.001, BF16 misses 61 of 200 return turns and
  the shipped file 36 (p 1.1e-4; rule 2: 77 against 59). A narrow statement: one turn, one temperature, one history
  (E7.4).
- **Locate.** The common changes of the candidate maps: stop line "diffuse at the side level" (E7.8). Tensor copies
  between two files with one bit map: the fits carry their gap under rule 1; reported as **suggestive only**, with
  both rule-2 reads and both sets of labels (E7.9). **A fresh private set, development half:** the
  control gap of the candidate map shows (+21 of 230 units, p 0.0076); the gap between the two files is not shown
  (+12, p 0.15; rule 2 p 0.11), so the lines on the copies were not read; as a report, one of the four leads shows
  again, and one graft arm, chosen after the look, is below both files (E7.15). **One read of that arm on the sealed
  half did not confirm it:** the control did not reach its line (+16, p 0.065), so the lines were not read; as a
  report, the graft misses less than its own base and is not below the shipped file. No repair is confirmed on
  held-out turns (E7.16).
- **Inkling 2 Mini.** A turn split found no class that holds the score gap (E7.10). A pre-registered confirm on
  fresh tasks passed its three lines: the first reply of a task has no code in 46.3 % of the calls for the official
  weights, 45.1 % for a public 2-bit file and 11.4 % for M2 (E7.11). A decision probe did not locate
  the decision (E7.12). A pre-registered cross-over of the thinking places it at the step after the thinking
  (E7.18). **A correction, post hoc:** "one sentence and no code" is what the client receives; in most such replies
  the model writes the action in a second block that the server drops, so a part of the decision may be removable
  on the serving side (not tested; E7.19). The decision shows **no task cost** in this client (E7.13): it is a marker of a departure
  from the parent, not damage, and its lower rate is not a quality gain.
- **A blind recount** of both studies by a separate agent: no count of a complete cell differs (E7.14).

### Text

- Title: "Kintsugi: Task-Gated Repair Reaches Full-Precision Agent Scores at 2.5 Bits". "Repair" is the loop (diagnose, locate, repair,
  gate), not the 27B bit allocation, which is the published one. "Full precision" is defined in Section 2 of the
  paper: the weights as their authors released them (BF16 for Qwen3.8-27B; for Inkling 2 Mini the official release,
  which has no higher-precision form). The abstract leads with the mechanism.
- Limitations, Future work, Table 10 (missed and stop lines), Appendix A (names) and the AI-use statement are updated.
- The coding-agent description in the Limitations has the public board's later run on release v1.2: 18 of 20 with the
  same model file (13 of 20 on release v1.1.1). The serving stack changed between the runs and the weights did not;
  which change caused the rise was not measured. One run each: a description, not a test of the weights (E6.16b).

### Repository

- New data `data/habits/` (per-unit outcomes: ids and 0 / 1 flags; no prompt or reply text), new script
  `scripts/v13_habits.py`, Figures 8 and 9.
- `evidence/EVIDENCE.md` section W7 (E7.1-E7.19) and row E6.16b; `evidence/LINES.md` section 7.

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
