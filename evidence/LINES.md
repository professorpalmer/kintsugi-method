# Pre-written lines

Every claim of the paper is judged by a line that was written, with its time, in the project's private record before
the run it judges. This file copies each line that the paper relies on, its verdict, and the row of `EVIDENCE.md` with
the numbers. A line found to be wrong was changed only for later runs, with the reason recorded; such changes are
listed where they happened. Times are 2026, local time of the record. Costs, machine names and coordination notes of
the record are not copied.

## 1. The stop rule (owner, 11-22 19:1x; written before any of its three tests ran)

Three tests decide the framing; no other test is added until one passes. **Family alpha 0.05 / 3 = 0.0167**
(Bonferroni over the three tests). If none passes, the paper is an evaluation paper ("we tried to show that KL-only
selection costs tasks at about 2.5 bits; on held-out seeds, held-out items and a harder suite it did not"). If one or
more pass, the paper claims exactly what passed. After a pass, a new test needs its own line written first.

| Test | Line | Verdict | Rows |
|---|---|---|---|
| 1. B1 recall population test | passes only if its own line (section 2) says that >= 1 lower-KL map loses recall at depth but not at the shallow control, at the family alpha | **passes** (one map, V5c = "KL-table B"); confirmed on one box by B1c | E1.34, E1.35 |
| 2. Naive rerun | passes only if the naive Unsloth UD-Q2_K_XL file is lower than the shipped file on the same tasks and seeds (104 + 105), exact two-sided sign test, p < 0.0167 | **passes** (292 vs 256 of 336, p = 1.0e-5); it compares a trellis build with a K-quant file and does not isolate the allocation method | E1.32 |
| 3. AppWorld test_challenge | 417 tasks never used for any choice; V5bc ("KL-only winner") vs shipped, seeds 106 + 107 (the reserve), the same harness and engine setup (config diff: the dataset only); passes only if V5bc is lower, exact two-sided sign test, p < 0.0167 | **passes** (610 vs 670 of 834, p = 1.3e-5; each seed p < 0.004) | E1.36 |

## 2. B1 and B1c (paper agent, 11-22 19:0x; B1c by the coordinator 22:0x; both before any of their runs)

- **Population** (scripts/b1/b1_gen.py, fixed seed 20261123): 20 documents of held-out CPython standard-library code
  (minus every file name of the selection's recall corpora), depths 64k / 117k / 245k (nested prefixes), 4 needles and
  4 distinct functions per document and depth, and a shallow twin (~3.9k tokens) of every 117k and 245k item. The
  population file is data/b1/population.jsonl.gz (sha256 bc46d5dabc423d33b86c06c63c1f1260614eeb458c5ec2af523be4517aaba337).
- **Statistic:** m = min margin over the answer tokens, capped at 5 nats; d = m(map) - m(shipped); L = mean d over the
  117k + 245k items, S = over their twins, C = L - S item by item; bootstrap over documents, 10,000 resamples, seed
  20261122; Bonferroni over the six lower-KL maps inside the family alpha: 99.72 % intervals.
- **Floors:** run floor = mean |m(repeat) - m(first run)|; refit floor = |L| and |C| of the author's fits of the shipped
  layout ("V2 body"); F_L = max(run, |L(V2 body)|), F_C = max(run, |C(V2 body)|).
- **Line:** a map "loses recall at depth" only if (a) the L interval lies below -F_L, (b) the C interval lies below -F_C,
  (c) the S interval does not lie below -F_L. B1 passes if >= 1 of the six lower-KL maps (V3s2, V5a, V5c, V5bc, V5tc,
  U1) meets it. No verdict if fewer than 12 documents are complete, the repeat is missing or the smoke check fails.
  Verdict: **V5c meets (a), (b), (c); the other five do not** (E1.34).
- **Smoke check S0 (before the run):** template check equal; no grammar-check failure; >= 95 % of answer positions with
  a wrong token in the top 20; |gf min - per-token min| <= 0.5 nats on 6 shallow twins. Passed (E1.34).
- **B1c:** the B1 pass is confirmed only if, with both arms on one box and one GPU model, (a), (b) and (c) hold for V5c
  with F_L = max(B1c run floor, 0.212) and F_C = max(B1c run floor, 0.295). If not confirmed, C1 is not claimed from
  B1. Verdict: **confirms** (run floor 0.000) (E1.35).

B1 arms (sha256 of each model file):

| Arm | sha256 |
|---|---|
| shipped (Kintsugi-27B-12G.gguf) | 0510bae82445dc53dbc7d0d842d8b7e146b8eafce0837e39b157d411e48c8a74 |
| V3s2 (c-h32 build) | 44358805f04c12cf8bc615aa0dbc699e68863d539c52349a84cd2594c55cf14a |
| V5a (c-h32 build) | 6d9d64440be1e2ef573e265571aeb2dbeeacb0182539984c9ad233309fe24fc7 |
| V5c (c-h32 build) | b1ea2376e64325f7abc46e754d0ff79759787450835971589d36564b6c79ebc3 |
| V5bc-h32 | df82fcabced01275033716dd0317662cdc75bea2bdfcd7121ec48aa51410354b |
| V5tc-h32 | ac672835784f04bb6332daf04de849ee3aed18c286e8ebad0915d4b9f67fc8ab |
| U1c-h32 | d02dabb42d4975e5efecaff891b2fa8c5295a0a2d3f2e8d6f642af2c8c02617b |
| H1c-h32 | fa49974c4802cf87b36717c4b0a3084ffdebafaef034f0f6862357cbabfa0dc0 |
| L0xRE-V2 body (refit arm) | 07e80cfe57d55b9e2543e219f6855aff06b953dad464358b8fbebfc5a8776fb3 |

## 3. Fresh-seed batch (lines 11-21; batch rules 11-22 before any run)

| Line | Written | Verdict | Rows |
|---|---|---|---|
| Seed budget: a seed is fresh for a model if no run of that model used it; Qwen 104 + 105 go to this batch, 106 + 107 are reserved; no seed is used for a choice and then for a claim | 11-21 / 11-22 | kept | - |
| G-A1: "KL-only selection costs agent work" only if V5bc-h32 is lower than V3ac-h32 with p < 0.05 on seeds 104 + 105; else the draft says the loss did not replicate and C1 rests on recall and long-chat documents | 11-21 | **not met** (296 vs 292, p = 0.68) | E1.31 |
| G-A2: "the selection seeds inflated the winner" only if the bootstrap interval of the selection effect (winner's S - F minus the others' mean, tasks resampled with seed pairing) lies above 0; else "within seed-to-seed variation" and C3 is a procedure | 11-21 | **not met** (+13.3 [+0.0, +27.0]) | E3.15 |
| G-B uniform map: K3 on the last N layers, N the largest whose code bits do not exceed V3a's | 11-22 10:4x | N = 30 | E1.28 |
| G-B HAWQ-style map: score = (tr(H) / n_in) x squared fit error at K2 and K3; exact knapsack to V3a's bits | 11-22 10:4x | built | E1.29 |
| G-B comparison: report KL and recall; AppWorld only if a map passes the recall gate; no claim beyond better / not better per gate | 11-21 | both fail the recall gate on the fragile item only | E1.28, E1.29 |
| G-B naive file on fresh seeds: better / worse / level vs V3ac-h32 at p < 0.05, sign test | 11-21 | worse (also stop-rule test 2) | E1.32 |
| G-C: "runtime repair matters for recall" only if blanket q4_0 or blanket q8_0 misses any needle or copy at 64k, 117k or 245k while the release setting keeps full marks; else "shown on KL and memory only" | 11-21 | **not met** | E2.22 |
| Unfinished task-runs: a run still going 3 h after its shard's other tasks is stopped and scored "unfinished"; a line is met only if it holds for every outcome | 11-22 | not needed in this batch | - |
| Box gates: KL of the shipped file on the box within +-1 % of 0.056398 and top-1 within +-0.3 pt (changed 10:5x before any run from +-0.2 % / +-0.05 pt, because the box builds another engine); recall box gate 24/24 + 20/20 before any map's recall gate counts | 11-22 | passed | E1.28, E1.30 |
| Chat template: every arm runs with its own GGUF template, as its selection-seed run did (decision 13:3x, before any shard of the two arms concerned) | 11-22 | applied; first-call prompt tokens equal 336 / 336 | E3.16 |

## 4. Graded follow-ups (11-22, before any run)

| Line | Written | Verdict | Rows |
|---|---|---|---|
| G-C2 statistic: capped (5 nats) min margin, d vs the release K/V setting over 88 item-depth units, 95 % bootstrap over the 44 items, 10,000, seed 20261122 | afternoon | - | E2.23 |
| G-C2: "the per-class choice matters for recall (graded)" only if the interval for blanket q4_0 or blanket q8_0 lies below 0 | afternoon | **not met** | E2.23 |
| G-C2, 4-bit global cache: "costs recall margin" if below 0; "no measurable margin loss" only inside [-0.25, +inf); else inconclusive | afternoon | **inconclusive** | E2.23 |
| G-C2 noise floor (added 15:1x after a smoke finding, before any result): a repeat of the release arm; a line is met only if its interval also clears the mean repeat difference | 15:1x | floor 0.000 | E2.23 |
| G-C2 method change (15:2x, before any result): one grammar-forced request per item; validity: grammar checks pass, >= 95 % of positions with a wrong token in the top 20, within 0.5 nats of the per-token method on the smoke items | 15:2x | passed | E2.23 |
| G-B2: "U1 / H1 loses recall margin" only if the interval of the mean d on the 43 items without the fragile item lies below 0; above 0: "has more recall margin" (no other claim) | evening | **has more margin** (both) | E1.33 |

## 5. Earlier lines cited in the paper

| Line | Written | Verdict | Rows |
|---|---|---|---|
| Recall gate: 24/24 needles and 20/20 copies at 117,000 (and 245,000 where run) | 11-19 | pass / fail per map as in Table 1 | E1.3, E1.6-E1.9, E4.3 |
| Named long-chat check: p2 docs 93 / 529 at most 0.873 / 0.422 (written after V5bc's result, before V5t's) | 11-20 | V5t flagged | E1.9, E1.31 |
| Probe validation: a probe may rank groups only if it ranks a known loser below the reference | 11-21 | agent-trace probe dropped | E1.14 |
| Task-guided maps: (a) no recall item under 0, (b) margin sum not below -2 SE, (c) p2 KL limits incl. docs 93 / 529, (d) best by margin sum | 11-21 | each candidate failed one line | E1.13 |
| Sparse attention past 200k: speed >= 1.25x at 245k, full recall, copy-heavy decode >= 0.95x | 11-21 11:52 | **failed** (0.553x) | E2.20 |
| Inkling K/V setting at 32k: KL <= 0.005 and top-1 >= 98 % | 11-21 | KL pass; **top-1 failed as written** (97.97 %) | E2.2 |
| Inkling bit map: >= 10 % lower KL than Unsloth's UD-Q2_K_XL with the interval below zero, top-1 not lower, file not larger; engine vs simulation per-document r >= 0.95 | 11-21 | map line passed; **r failed as written** (0.919) | E2.5, E2.6 |
| 4-bit global cache: KL <= 0.005 at 32k; 245k decode >= 13.05 tok/s; 24/24 + 20/20 at 117k and 245k | 11-22 03:5x | all passed | E2.16-E2.18 |
| Harness gate hard lines: in every harness 0 HTTP 500, 0 unfinished streams, 0 malformed calls, 0 leaks, 0 loops, and >= 5/6 tasks | 11-22 ~04:40 | **failed** (one OpenCode loop) | E1.24 |
| Model vs engine: grammar or sampler bug only if the tool name is likely without the grammar but suppressed with it; engine / quant suspect only if our loop rate >= Unsloth's + 2 | 11-22 | model behaviour | E1.25 |
| Bad-token check: 0 malformed-token events in any round of 6 samples, every page renders | 11-22 | passed | E1.26 |
| Inkling vs Unsloth's setup: "better" only with p < 0.05 (seeds 102 + 103); kept only if ours >= Unsloth on seeds 104 + 105 | 11-22 00:5x / 02:2x | better; held | E3.12, E3.13 |
| Inkling vs the official weights: p >= 0.05 means level (amended 04:2x before any shard: same reasoning budget for both arms) | 11-22 04:1x | level | E4.8 |
| 4-bit global cache on AppWorld: pass if p >= 0.05 against the 8-bit cache and >= 292 / 336 on seeds 102 + 103; extension to 104 + 105 reported as level unless p < 0.05 | 11-22 10:2x / 11:1x | level | E4.12 |
| Inkling drafting: decode +8 % at 4k | 11-21 | **failed** (+7.9 %) | E2.12 |
| Inkling drafting at long context: +20 % at 117k and 245k | 11-22 | **failed** | E2.21 |
| Inkling loop guard: >= 8 / 10 runs unstuck at each stuck point; prompt-cache reuse >= 0.9 after a firing | 11-22 | **both failed** | E2.24 |
| Inkling two-slot memory: peak VRAM <= ceiling, 0 spill lines, 0 CUDA errors, shared memory <= start + 64 MiB | 11-22 | **failed** (shared 394 MiB) | E2.25 |

## 6. Second batch (2026-11-23; every line before its run or its computation)

The stop rule of section 1 permits new tests after a pass; each has its line written first. Verdicts are given at the
task level as the line was written, and at the seed level (section 6.1) where the line is about AppWorld.

### 6.1 Seed clustering (G-G, 11-23 20:4x, before any of its numbers; G-G2 the same evening, before its numbers)

| Line | Verdict | Rows |
|---|---|---|
| "The task-runs of a seed are clustered" only if the mean squared z of one arm's seed difference is above 1 and the sum of z^2 is above the chi-square 95th percentile | **met for the 27B** (2.24, p 0.0064); not shown for Inkling (1.02) | E5.1 |
| Per claim of a difference: "keeps p < 0.05 at the seed level" only if every seed's difference has the claim's sign AND the design-effect test gives two-sided p < 0.05 (design effect = mean over arm pairs of the sample variance of the per-seed differences / the mean number of discordant tasks, floored at 1) | per comparison, as in the paper's tables | E5.1 |
| A claim of "not worse" (margin -25 of 834) keeps it only if the lower end of the cluster-robust interval is above the margin | shipped vs BF16: kept; M2 vs the official Inkling weights: kept | E5.4, E5.23 |
| G-G2 (test_normal comparisons of v1.0.x): the same method; design effect = the larger of the G-G value and the test_normal value | 27B 2.75 used (test_normal value 0.19); the claims that keep and do not keep it are listed in the row | E5.28 |

### 6.2 The 27B model on AppWorld test_challenge (417 tasks; seeds 106 + 107 unless named)

| Test | Line | Verdict (task level, as written) | Seed level | Rows |
|---|---|---|---|---|
| W2-A, KL-tools and KL-table B | "loses agent work on the harder split" only if lower than shipped with p < 0.025 (Bonferroni over the two maps) | KL-tools: not shown (680 vs 670). KL-table B: loses (586 vs 670) | KL-table B keeps (p 2.5e-4) | E5.2 |
| W2-A, class line | "KL-only selection costs agent work as a class" only if >= 2 of the 3 lower-KL maps pass their own line | met (KL-only winner, KL-table B); KL-tools is the exception | both keep | E5.2 |
| W2-A, BF16 | "the 12 GB build is not worse than BF16" only if the lower end of the 95 % interval of (shipped - BF16) is above -25 of 834 | met; the interval is wholly above 0 (670 vs 614) | kept | E5.2, E5.4 |
| W2-A2, fresh seeds 108 + 109 | "the 12 GB build scores higher than BF16 on the harder split" only if p < 0.05 and shipped > BF16 | **met** (660 vs 599) | four seeds: p 5.1e-4, [+51, +183] | E5.4 |
| W2-S, weights or setup | "the lead comes from the weights" only if the shipped file in BF16's setup is above BF16 with p < 0.05 and not below the 12 GB run; "from the setup" only if below the 12 GB run and not above BF16 | **weights** (670 vs 614; 670 vs 670) | keeps (p 0.014) | E5.5 |
| W2-H, two-stage head | "the head lifts the score" only if the head-off arm is lower than shipped with p < 0.05 | **not shown** (687 vs 670) | p 0.36 | E5.6 |
| W2-K, head repair | "the head repair improves agent work" only if the exact head is above the 3-bit head with p < 0.05 | **not shown on tasks** (687 vs 670) | p 0.41 | E5.17 |
| W2-U, baselines | "loses agent work" only if lower than shipped with p < 0.025 | uniform 630 and HAWQ-style 637: met as written | **not kept** (p 0.077, 0.15; seed 107 only) | E5.13 |
| W2-F prediction (written before any score of the two baselines) | both baselines below shipped with p < 0.05 and under 640 | held as written | **not kept** | E5.13 |
| W2-F (a), the band alone | the shipped map with only ffn_down 20-27 at 2 bits is below shipped with p < 0.05 | **not met** (671 vs 670) | level | E5.12 |
| W2-F (b), the band restored | KL-table B with ffn_down 16-31 at 3 bits is NOT below shipped AND is above KL-table B with p < 0.05 | **not met** (627: below shipped) | below shipped p 0.045 (narrow); above KL-table B not kept (p 0.089) | E5.14 |
| W2-F chat check | "the chat flag holds" for KL-tools only if its chat mean KL is above the shipped file's with the interval above 0 AND the median is above | **"two documents, not the class"** | - | E5.11 |
| W2-B, context length (CPU) | "the loss grows with context length" only if the shipped-only pairs are on longer tasks (one-sided Mann-Whitney p < 0.025) and the share rises from the bottom to the top tercile | **not shown** (the direction is the opposite) | - | E5.10 |
| W2-C, bisect on the B1 population | "group g carries the held-out recall loss" only if the flip map's L interval is not entirely below the floor AND the recovery interval is above 0 (98.75 % intervals) | **ffn_down 16-31 carries it** (recovery 98 %); the other three groups do not | - | E5.15 |
| W2-T, transcripts (CPU) | T1 "BF16 fails by running out of steps more" (Fisher p < 0.05); T2 "BF16 thinks longer per call" (every seed); T3 "the opening word predicts the outcome" | T1 not shown; T2 met, small; T3 not shown | - | E5.9 |
| W3-T, token classes | "the band damage concentrates on class X" only if X has the largest increase, its interval is above 0 and above the point value of the next class | **spread** | - | E5.16 |
| W3-E, sharpness | "sharpness explains the agent ordering" only if both winners are sharper than BF16 and neither loser is | **not met** (every map is flatter than BF16) | - | E5.8 |
| W2-T0, temperature 0 | "the lead over BF16 holds at temperature 0" only if shipped > BF16 with p < 0.05 over 417 paired task-runs | **not shown** (330 vs 319, p 0.235, +11 [-5, +28]); BF16 gains about 16 over its T = 1.0 mean | one run per arm (no seed clustering at T = 0) | E5.18 |
| Card gate | another card may join only if the 50 identity prompts are token-identical to an RTX 4070 | an RTX 4080 failed (31 / 50); RTX 4070 only | - | E5.29 |

### 6.3 Inkling 2 Mini (uniform rule fixed before the build; Bonferroni over the two baselines)

| Test | Line | Verdict | Seed level | Rows |
|---|---|---|---|---|
| W2-D KL, M2 vs Unsloth's file | the gap holds on the release engine only if the 97.5 % interval of the difference is below 0 | **met** (-27.0 %) | - | E5.20 |
| W2-D KL, M2 vs the uniform map | "the per-group map lowers KL" only if the 97.5 % interval is below 0; "the uniform rule is lower" if above 0; else "level" | **level** (-2.5 %) | - | E5.20 |
| W2-D recall counts | a count difference is a recall difference only if one arm fails a depth that M2 passes | Unsloth's file misses one strict copy at 32k (lenient form full) | - | E5.21 |
| W2-D graded recall | "less recall margin than M2" only if the 97.5 % interval of the mean difference is below 0; "more" if above 0 | uniform map **less** (-0.971); Unsloth's file **more** (+0.865) | - | E5.21 |
| W2-D AppWorld test_normal, seeds 108-111 | "M2 scores higher than the baseline with the setup equal" only if p < 0.025; else "level" | vs Unsloth's file **higher** (598 vs 579); vs the uniform map level (598 vs 596) | keeps (p 0.019) | E5.22 |
| W2-I line 1, test_challenge seeds 112 + 113 | "the uniform map loses agent work on the harder split" only if U < M2 with p < 0.05 | **not met** (649 vs 638) | p 0.42 | E5.23 |
| W2-I line 2 | the uniform map's tools-class KL gap is larger than its all-rows gap with the interval above 0 | **not met** (interval holds 0) | - | E5.23 |
| W2-I line 4 | "M2 is not worse than the official weights on the harder split" only if the lower end of the interval of (M2 - official) is above -25 of 834 | **met** (638 vs 604); the two arms differ in K/V precision and card class | kept ([+6, +62]) | E5.23 |

### 6.4 Other lines of the second batch

| Test | Line | Verdict | Rows |
|---|---|---|---|
| Free generation on the 24 needles at 245k | every item at most 1,000 characters of thinking, finish "stop", an answer given | KL-tools: one item with 1,184 characters. The limit is not a validated threshold (an independent port reports that the official Inkling weights exceed it on 3 of 20 held-out items), so the paper reports the lengths against the reference run and treats the item as an observation | E5.26, E5.27 |
| Faithfulness of the free-generation item (before the BF16 run) | "KL-tools is faithful on vault 9" only if BF16's vault 9 thinking is over 1,000 characters AND it names the correct codes of at least 3 other vaults | **not faithful, as written** (BF16 221 characters, 0 other codes; all 24 items 199-232 characters). The line reads free-generation length only; the forced reading below overrules it as a statement about the KL-tools choice | E5.30 |
| Forced reading under BF16 (LN2, before the rent) | "the fork is a near tie in the reference" only if BF16's gap at the first token where the two files differ is under 0.25 nats; "file X flips against the reference" if BF16's top token there is the other file's token with a gap >= 0.25 | **"the shipped (v1.1) file flips against the reference at vault 9"** (gap 2.68). Both files leave BF16 earlier at the same tokens (22, 34); after that BF16 prefers the long KL-tools reply (478 of 480 tokens) | E5.31 |
| No map tuned to one item (rule, written when the item was found) | a later candidate must pass a fresh needle set; no map is tuned until the item passes | kept: the one-group maps are a diagnosis only (they move the event to other items) | E5.26 |
| W4-L overlap check | a golden document is "in his text" at >= 50 % shared 50-character windows, "touched" at >= 5 %; with any document >= 5 % the clean set gives the verdict and the full set is reported next to it | **overlap found** (3 in, 2 touched); clean set 293 documents | E5.25 |
| W4-L, new LookAtMyQuant file | "the gap to M2 closes in class c" only if the 95 % interval of (new - M2) includes 0 or is below 0, for code, tools, agentic, reasoning | not yet measured on our golden set (the rig port's check R0 did not pass) | E5.24 |

## 7. Version 1.3: decision-point habits (2026-11-26 and 11-27; every line before the cells or calls it judges)

"Rule 1" = the reply has no tool call (the rule of every line below). "Rule 2" (no call names the returned function)
was added after the first results and judges no line. Verdicts marked * were read by the paper agent from the
written line and the counts; the study record gives the counts.

### 7.1 The return-turn replay (Qwen3.8-27B; BFCL v3 multi_turn_miss_func)

| Test | Line | Verdict | Rows |
|---|---|---|---|
| Validity V-1 (per history) | "the gap reproduces on history h" only if D(KL-tools-3 - shipped) >= +10 AND p < 0.05 AND D > 2 x F (F = the replay noise, at least 2) | HS\*: **not met** (+8, p 0.17). HT\*: **met** (+27, p 3.5e-6) | E7.2 |
| Validity V-2 (both histories) | D > 0 on each history AND by-entry D >= +16 AND p < 0.01 AND D > 2 x F | **met** (+35, 40 / 10, p 2.4e-5). The replay is valid; the analysis set is HT\* (V-1 holds on one history); reading "mixed" | E7.2 |
| Diagonal check (an expectation, across machines after an amendment) | a file on its own history repeats the live call on 90 % or more of the units | 95.0 % and 89.4 % | E7.2 |
| Step 1, "carries" | an arm carries iff its share of the gap on the analysis set is >= 0.60 AND p < 0.05 AND D > 2 x F; clear iff share <= 0.30 AND p >= 0.05; else partial. Exactly one side must carry, or the hunt stops | raises partial (0.48), cuts clear (0.19): **stop line "diffuse at the side level"**; no step 2 line read; no candidate map built | E7.8 |
| P-1, the parent | "on identical prompts the shipped map misses the returned tool less often than the parent" HELD iff D(BF16 - shipped) > 0 on every history with complete cells AND p < 0.05 AND D > 2 x F | **held\*** (one history complete: 61 against 36, 33 / 8, p 1.1e-4) | E7.4 |
| S-1, sampling | "the gap holds when the model samples" HELD iff D per seed > 0 for each of 3 seed bases AND the by-entry sign test over the 3 seeds and both histories has p < 0.05 | **held** (+49, +49, +42; 91 / 22, p 3.7e-11) | E7.7 |
| W-a, W-b, wordings | "the gap is not tied to the BFCL line" HELD (per wording) iff D > 0 on each history AND the by-entry p < 0.05 | (a) **not held\*** (+4, +4; p 0.092); (b) **held\*** at the edge (+2, +11; p 0.049) | E7.6 |
| W-c, "ceiling" | HELD iff no call in 5 % or less of the units for both files | **not held\*** (7.0 % and 24.4 %) | E7.6 |
| Fit copies, half-gap rule (amendment, before any of its cells) | an arm meets it iff its by-entry D against shipped is >= 13 (half of the reference gap of 26) AND p < 0.05; it "carries" iff it meets the rule and its complement does not | the 400 second fits **carry** (+26, p 0.0056; the 53 other tensors -3); attention + linear-attention kinds **carry** (+31; MLP kinds +18, p 0.063); layers 0-31 **carry** (+31; layers 32-63 +3); by kind: linear-attention output **meets** (+21, p 0.044), three others do not. 6 primary and 6 secondary tests, no correction. By the plan's step-1 rule (an extension by the blind recount) the 400-fit arm and the output arm are "partial". Reported as suggestive only | E7.9 |
| Fit copies, wording (c) as "the stronger probe" | only if the by-entry gap of the second-fit file under (c) is above 26 AND p < 0.013 | **not met** (+11, p 0.14) | E7.9 |

### 7.2 Inkling 2 Mini

| Test | Line | Verdict | Rows |
|---|---|---|---|
| Turn split of the recorded runs | ONE CLASS iff the top class has half of the gap or more AND the sign test by task has p < 0.05; a gap under 10 task-runs = NO GAP TO SPLIT; else DIFFUSE | the official weights against M2: **no gap to split**; the two pairs with a gap: **diffuse** (shares 0.30, 0.31) | E7.10 |
| First-call confirm, L1 | the public 2-bit file's no-code rate on fresh tasks is 20 % or more (a cut reply counts as code) | **pass** (350 / 834 = 42.0 %) | E7.11 |
| First-call confirm, L2 | M2 is below it: exact sign test by task (seeds pooled per task), p < 0.001, AND M2's rate is half or less (cut replies counted against the claim) | **pass** (95 / 834 against 350 / 834, ratio 0.27; 223 / 22, p 4.8e-43) | E7.11 |
| First-call confirm, L3 | M2's replay on the standard split, seed 102, is inside the exact 95 % interval of the record's 50 / 672 at n 168: [6, 20] | **pass** (18 / 168) | E7.11 |
| Task-cost check (reading rule, before the count; post hoc data) | "the habit costs tasks" needs the pooled within-task sign test p < 0.05 AND a mean of -5 points or lower; "no task cost shown" if the pooled p >= 0.05; anything between = "a small cost" | label by the rule: "a small cost" (p 0.049). The rule did not foresee the sign: the runs with no code first pass MORE often (+1.5 points, [-1.8, +4.7]). **Plain reading: no task cost shown**; a cost under about 2 points is not excluded | E7.13 |
| Cross-over of the thinking, line D (validity) | each file's weights on its own thinking give a no-code count inside the 95 % binomial interval of its count in the confirm | **pass** (111 in [77, 112]; 331 in [308, 362]) | E7.18 |
| Cross-over, line T | "the thinking carries the habit" iff its share of the gap is 0.4 or more AND both thinking tests go the same way (the 2-bit file's thinking gives more no-code) with p < 0.001 | **no** (share -0.19; -4.0 and -7.7 points: the other way) | E7.18 |
| Cross-over, line W | "the step after the thinking carries it" iff its share is 0.4 or more AND both weights tests go the same way with p < 0.001 | **yes** (share 1.19; +38.8 points, 312 / 29; +35.2 points, 272 / 18). Reading: the step after the thinking decides | E7.18 |

### 7.3 A fresh private set of return turns (development half, then one read of the sealed half)

The set is generated by us, has the structure of BFCL miss_func and no overlap with the public entries. Only counts
leave the lab.

| Test | Line | Verdict | Rows |
|---|---|---|---|
| C0, the control | the gap of the candidate map (KL-tools-3 minus shipped) on the primary set is above 4 AND p < 0.05 under rule 1 ("the fresh set shows the habit gap"); rule 2 reported | **holds** (+21 of 230 units, 27 / 10, p 0.0076; rule 2: +18, p 0.017) | E7.15 |
| L-G, the gap of the two same-map files on fresh entries | "the gap reproduces on fresh entries" holds under a rule iff the by-entry difference (second-fit file minus shipped) on the primary set is above 4 AND p < 0.05; "the lead reproduces" iff it holds under BOTH rules | **not met** (rule 1: +12 of 230 units, 16 / 8, p 0.15; rule 2: +12, 17 / 8, p 0.11): not shown at this size; the locate result is not replicated | E7.15 |
| The fit copies and the graft arms on that set | read only under a rule where L-G holds | **not read** (no gap on the development half); their counts are a report: one copy arm of four shows again (attention + linear-attention kinds: +22, p 0.0070; rule 2 +19, p 0.029), and one graft arm of five, chosen after the look, is below shipped (-11, p 0.043; rule 2 -12, p 0.015) | E7.15 |
| Resolution, stated in the plan before the run | a gap of the public size is about +15 on this set, with p near 0.05 at best; a failed line reads "not shown", not "absent" | - | E7.15 |
| Sealed half, C (the control; lines written after the development results and before the sealed half was read) | KL-tools-3 misses more than shipped, p < 0.05 under rule 1; if C fails, S-1 to S-3 read "not read" | **fails** (+16 of 230 units, 24 / 12, p 0.065; rule 2: +14, p 0.18): S-1 to S-3 not read | E7.16 |
| Sealed half, S-1 | the graft arm (chosen after the look at the development half) misses less than its base, p < 0.05 under both rules, AND its misses are at most 0.75 x the base's under both rules | not read; report: -24, 7 / 29, p 3.1e-4, ratio 0.70; rule 2: -16, 9 / 24, p 0.014, ratio 0.80 (above 0.75) | E7.16 |
| Sealed half, S-2 | the graft arm misses less than shipped, p < 0.05 under both rules | not read; report: -7, p 0.42; rule 2: -10, p 0.23 | E7.16 |
| Sealed half, S-3 | the reverse copy misses more than shipped, p < 0.05 under both rules | not read; report: +13, p 0.087; rule 2: +6, p 0.70 | E7.16 |
| Sealed half, the reading rule | S-1 and S-2 pass: "a graft below both parent files, confirmed on held-out turns"; only S-1: "the graft lowers its base's habit"; neither: "the development lead did not hold" | **the development lead did not hold**; no repair is confirmed on held-out turns | E7.16 |
