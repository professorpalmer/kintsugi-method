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
