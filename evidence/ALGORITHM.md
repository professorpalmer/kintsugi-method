# The Kintsugi method as an algorithm

Status: draft in the private repo, 2026-11-21 (cleared for a public version). Written so that a reader with a quantizer, an inference engine and a few GPUs can
repeat it. Costs are the measured costs of our two runs (Qwen3.8-27B on one RTX 4070 12 GB; Inkling 2 Mini on rented
boxes), with the source. "Rig" = one RTX 4070 12 GB. Box prices are rented.ai rentals in November 2026.

## Notation

- $W = \{w_1, \dots, w_P\}$: the coded projections (Qwen: $P = 400$; Inkling: 48 expert tensors of 128 experts each).
- $K \in \{2, 3\}$: bits per weight of one projection's trellis code. A **map** $\mathbf{k} = (k_1, \dots, k_P)$.
- $B(\mathbf{k})$: bytes of the coded body; $B^\ast$: the byte budget.
- $\mathcal{R}$: the runtime settings (K/V type per cache class, head precision, drafting).
- Golden set $D$ split into a build set $D_b$ and an untouched set $D_u$.
- $\mathrm{KL}(\mathbf{k}) = \frac{1}{|R|}\sum_{r \in R} \mathrm{KL}\big(p_{\mathrm{ref}}(\cdot \mid r) \,\|\, p_{\mathbf{k}}(\cdot \mid r)\big)$
  over scored rows $R$.
- **Margin** of answer token $t$ with right token $y_t$:
  $m_t = \log p(y_t \mid \text{prefix}) - \max_{y \neq y_t} \log p(y \mid \text{prefix})$, teacher-forced on the gate's
  own prompt. Item margin $m_i = \min_{t \in i} m_t$. $m_i < 0$ means greedy decoding writes a wrong token.
- **Fragile item**: an item whose margin on the reference build is below a threshold fixed before any run
  (we used 1.0 nat).

## Inputs

1. The full-precision (or official) model and a reference forward that you trust.
2. A quantizer that can fit each projection at each width independently (we used exllamav3 `quantize_exl3`, EXL3
   trellis codes, with public Hessians).
3. An inference engine whose output is deterministic for a fixed input (batch-invariant kernels, one slot). Check
   this first (step 0).
4. A golden set for KL with a document split; task probes (below); one agent benchmark with seeds.
5. A starting map $\mathbf{k}^0$ (a published allocation or a uniform rule) and the byte budget $B^\ast$.

## Outputs

1. The shipped map $\mathbf{k}^\ast$ and runtime settings $\mathcal{R}^\ast$, or "no change" (the start stays).
2. A record: every line, written before its run, with its result, including failed lines and later corrections.
3. Confirmed numbers: each public claim re-measured on data not used to choose.

## Step 0. Make the measurement trustworthy (once per engine build)

- 0a. **Reference check.** Our forward vs an independent build on the same weights: Inkling line top-1 >= 99 % and
  KL <= 0.01 (V1). Cost: about 1 h of one A100 box (part of the $11.55 of Inkling stage 1).
  Source: lab record INKLING-MAP-STUDY # Stage 1 (a).
- 0b. **Determinism.** Run one probe twice on one file with separate server starts. Line: per-item margin spread
  < 0.2 nat; if 0, every later row runs once and noise comes from bootstrap over items. Ours: 0.0 (E1.15).
  Cost: two probe runs (about 30 min rig).
- 0c. **Engine = simulation.** The shipped file in the engine vs the same map in the reference forward: mean-KL gap
  within max(2 x the baseline's gap, 0.003); argmax agreement above the engine-to-engine floor (measured: 98.6-98.8 %).
  Cost: one full golden-set pass per arm (about 30-45 min on a rented RTX 4070). Source: lab record INKLING-2B # 2.
- 0d. **Box = rig.** If probes run on rented boxes, the box's reference probe must equal the rig's (we required every
  margin within 0.05 nat; we got 0.0000). Source: lab record TASK-DAMAGE-MAP # 03:56.

## Step 1. Fit a library once

For every projection $w_j$ and every $K \in \{2, 3\}$, fit and store the code. No new fits per map.
- Output: library $L = \{c_{j,K}\}$; a byte-exact writer $\mathrm{write}(\mathbf{k}) \to$ model file.
- Writer check: rebuilding the start map from $L$ equals the start file byte for byte.
- Cost: Qwen3.8-27B, 800 fits, about 2.0 min per layer on the rig (7,169 s of quantizer time), with the public
  Hessians (10.3 M tokens); library 15 GB. Inkling 2 Mini: Hessian capture about 1.4 h of A100 (about $3), 12,288
  expert fits on one 80 GB box; library 15 GB. Writing one map: 20 s to 6 min (disk-bound).
  Sources: lab record RESULTS # 6; lab record INKLING-MAP-STUDY # Stage 1 result; lab record V3-SENSITIVITY # 01:51.

## Step 2. Diagnose: measure KL AND task probes, per file

For each candidate file, record all of:

| Probe | What it sees | Cost (rig unless said) |
|---|---|---|
| KL on a subset of $D_b$ (T, 74 docs) | average next-token error | about 2.5 min + the write |
| KL on all of $D_b$ and on $D_u$ | the same, final | about 19 min each |
| KL by class and by named documents (long reasoning chat, non-English chat) | long-answer quality | about 1 min per document; full $D_u$ 13 min |
| Recall margins on the gate's own prompts (44 items at 117k) | deep recall, continuous | about 15 min |
| Full recall gate (24 needles + 20 code copies, free generation) at the deepest depths | deep recall, pass / fail | 25-30 min per depth |
| Agent benchmark (AppWorld test_normal, 168 tasks, one seed) | agent work | about 3 GPU-h on rented RTX 4070s (about $1-1.5) |
| Code health (HumanEval 164) | catastrophic loss only (saturated) | 73 min for BF16 on one A100 (rig time not logged) |

**Probe validation rule.** A probe may rank groups only if it ranks a known loser below the reference with an
interval that excludes 0. Our teacher-forced agent-trace probe failed this (E1.14) and was dropped as written.
**Hard items rule.** A recall probe must contain items that are hard for the reference (low margin); a probe whose
items all have margins of 2.4-7.2 nats passed a map that lost a needle (E1.7).

## Step 3. Locate: a damage map from one-group flips

- Groups $G_g$ = role x layer band (Qwen: 21 rows; same groups as the KL table so the rankings can be compared).
- Row $g$: flip group $g$ one width step in the start map, write the file, run the step-2 probes.
- Per row, per probe: change $\Delta_g$, change per MiB $\Delta_g / |\Delta B_g|$, and the hit list: items with
  $m_i < 0$ that were above 0 on the start map.
- Cost: Qwen 21 rows on one 8x RTX 4070 box: 4.79 h, about $4.27 (rig estimate: about 15 GPU-h). Inkling probe set:
  about 50 one-minute probes on one 80 GB box. Sources: lab record TASK-DAMAGE-MAP # 03:56; lab record INKLING-MAP-STUDY
  # Stage 1b.
- Report the rank correlation between KL per MiB and task damage per MiB with a bootstrap interval over groups, and
  say whether it is informative (ours: rho 0.007 [-0.558, 0.622], mostly noise; E1.10).

## Step 4. Repair: move bits to the damaged spots under the budget

1. **Hard constraints first.** A group is a source (may lose a bit) only if its row has no hit-list item and no
   named document worse than a fixed line (we used +7 %). A group is a sink only if its row raised the hard items.
2. **Exchange.** Pair the cheapest safe sources with the best sinks at equal bytes ($|B(\mathbf{k}) - B^\ast|$ within
   0.1 %), best expected net first. Plan with a discount for non-additivity (measured: two groups added to 98 %, six
   to 82 %; we planned with 0.8).
3. **Measure, do not predict.** Every candidate map is written and run through step 2. Two to five candidates.
4. **Runtime settings are repairs too.** For each runtime knob (K/V type per cache class, head precision, drafting),
   scan the options against the start on the same KL / top-1 line, and prefer spending precision where it costs
   little memory (Inkling: 40 sliding-window caches at f16 cost 240-320 MiB at any context; 8 global caches grow with
   context, so they take the cheapest type that passes: q8_0 in v1.0; q4_0 for the v1.1 default candidate, which
   passed KL, memory, speed and recall at 117k and 245k, E2.15-E2.18). Cost: one KL pass per option (minutes).

## Step 5. Gate: lines written before each run

Rules:
- **R1.** Every line is written (with a time) before the run that it judges. A wrong line is changed only for later
  runs, and the change is recorded with its reason. A line missed by any margin is a miss (+7.9 % against +8 % is a
  miss; E2.12).
- **R2.** When a line fails for a reason found after the run (e.g. a noise floor below the line), the verdict is
  recorded as "FAIL as written" next to the diagnosis; the decision to proceed is recorded as a decision, not as a pass
  (E2.6 F3, E2.2 top-1).
- **R3.** No best-subset quoting: only the pre-declared pooled number counts (+8.6 % on one pair of runs was not
  quoted; E2.12).
- **R4.** Fragile items (reference margin below the threshold) are defined before the run. They make no "must keep"
  rule, but a release gate still counts them (E1.6).
- **R5 (added 11-22).** A KL or a recall pass does not replace a gate on the shipped task path. The release gates
  include the agent harnesses through the release launcher (G6) and a bad-token check (GATE P); the Inkling engine
  fault of E1.21 passed KL and failed every harness.
- **R6 (added 11-22).** When a task gate fails on behaviour that could come from the model, write a rule that
  separates model from engine before the check: the same task repeated on our build and on a reference build, with a
  count line written first, and the model's own probabilities at the failing step with the grammar on and off. The
  failed gate stays failed in the record whatever the check says (E1.24, E1.25).

Line types we used (all with the values written before the run):
- KL non-inferiority / superiority: relative change $\rho = \mathrm{KL}(\mathbf{k}) / \mathrm{KL}(\mathbf{k}^0) - 1$
  with a paired bootstrap 95 % interval over documents; e.g. "interval below 0" or "at least 10 % lower and interval
  below 0" (10,000 resamples).
- Class lines: no class worse by more than 2 %; named documents within +25 % of the reference.
- Recall: all items right (24/24 + 20/20) at each gate depth; no non-fragile item under 0 in margin form.
- Agent: pooled count over two seeds at least a fixed number AND exact two-sided sign test on discordant pairs not
  significantly worse ($p \geq 0.05$). "Better" only with $p < 0.05$. **Correction (paper v1.1):** the task-runs of
  one seed share one random stream when the client sends one seed on every call, so this task-level test is too
  optimistic. Use the difference on each seed, the design-effect test and the cluster-robust interval (paper Section
  4.1, `scripts/gg_seed.py`), at least three seeds for a claim of a difference, and per-call seeds in the client
  (`SEED-FIX.md`).
- Free generation: for each item of the deep recall gate, the answer, how the reply ended and the thinking length,
  compared with the reference file on the same items. A fixed length limit is not a validated pass line. Do not tune
  a map until one item passes: it can move the event to other items (paper Section 6).
- Health: HumanEval at least a fixed count (saturated; never used to rank).
- Speed (for runtime repairs): at least a fixed gain over off, pooled over all pre-declared runs.

Ship rule: a candidate replaces the start only if every line holds AND it is better on at least one task gate beyond
its noise. Otherwise the start stays and the study is reported as negative or neutral.

## Step 6. Confirm on data not used to choose

- Re-run the winner and at least one never-selected reference arm (BF16 or the start file) on fresh agent seeds and
  held-out documents, with the same harness and settings. Paired test as in step 5.
- A public claim stands only if it holds on the fresh data. Otherwise it is corrected in public with the reason.
- Cost (ours): fresh AppWorld seeds 102 + 103 for the shipped file on one 8x RTX 4070 box: $3.30; for BF16 on two
  A100s: about $14; for the weights-only arm: about $14.6. Total of the fair test: $31.91. Recheck of the Inkling
  head-to-head: about $2-3. Source: lab record LOG-lab # 14:29, # 14:30.
- A result that was never selected can still be re-checked before a public claim, with a keep-or-drop rule written
  first. Inkling M2 vs Unsloth's setup: "better" on 102 + 103 held on 104 + 105 (E3.12, E3.13).

## Step 7. Publish the seams

Keep failed lines, rejected repairs and corrections in the public record (the "gold seams"). Report every candidate
that was measured, not only the winner.

## Total cost of one pass (measured, rounded)

| Model | Fits | Locate (table) | Candidates + gates | Confirmation | Total |
|---|---|---|---|---|---|
| Qwen3.8-27B (V3s + V5 + task-damage map) | about 2 h rig | about 15 rig-h (KL table) + $4.27 box (task table) | about 25 rig-h + AppWorld arms (estimate $10-15; not summed from the rentals log) | $31.91 | about 3 rig-days + about $50 of boxes (estimate) |
| Inkling 2 Mini (stages 1, 1b, 1c, 2b gate) | $11.55 (incl. Hessians, reference check) | $6.65 | $2.05 + about $1 of 4070 box | not run (gap) | about $21 of boxes + rig hours |

Sources: lab record INKLING-MAP-STUDY # Stage 1c log (track total $20.25); lab record TASK-DAMAGE-MAP # Where it runs;
lab record LOG-lab. Rig hours for the Qwen KL work are from the V3 / V5 logs; they are approximate.
