# Kintsugi: Find and Repair Quantization Damage Where Tasks Break, Not Where Averages Look

Cary Palmer

*Version 1.0.1 (2026-11-23). Every number cites a row of `evidence/EVIDENCE.md` as [E#.#]; `README.md` says how to recompute each one.*

---

## Abstract

Low-bit quantization of large language models is usually tuned and reported by an average error, such as perplexity
or the KL divergence to the full-precision model. We show, on a 27B dense hybrid model at 2.47 bits per weight,
that average error does not rank bit maps by task quality: none of our five bit maps with 7-11 % lower KL than the
published allocation could replace it. On our recall gate two lost a fact 117k tokens back (one also lost a code
copy), a third failed only on one fragile code-copy item (as two baselines did), and the other two were worse on long
chat documents and gave no agent gain on fresh seeds of the standard benchmark split. On 20 held-out documents with 480 recall items and a
short-distance twin of each long item, one of six lower-KL maps (our five and a uniform baseline) lost long-range
recall margin: about 1.1 nats on average at 117k-245k tokens and none at short distance, with no wrong answers;
a rerun on one machine confirmed it. With
teacher-forced margins we traced one recall loss to a band of MLP up-projections that KL rates as the cheapest place
to remove bits; in the other direction, KL found general-text damage that our task probes missed. We describe a
simple discipline, which we call the Kintsugi method: diagnose with task probes next to KL at the level of layer
groups and runtime settings, repair only the damaged spot, gate every change with lines written before the run, and
confirm on data that was not used to choose. We applied the last step to our own claims, and two did not hold on
fresh seeds of the same split. A KL-chosen map that lost 17 of 336 agent task-runs on the selection seeds (p = 0.019)
was level on two fresh seeds of that split (296 vs 292, p = 0.68); on the harder test_challenge split, which no
choice used, it lost 60 of 834 task-runs (610 vs 670, p = 1.3e-5, on both seeds). Our shipped 27B build led BF16 by 21 task-runs on the selection seeds; on two pairs of
fresh seeds the gap was +1 and -2. With four candidates on the second fresh pair the selection effect was +13.3
task-runs, bootstrap interval [+0.0, +27.0], so we cannot separate it from seed noise. On a looped mixture-of-experts model
(Inkling 2 Mini), the same steps gave a 9.07 GB build with 27.3 % lower KL than a public 2-bit build of the same
model at a smaller size, with full needle and code-copy recall up to 245k tokens on one 12 GB GPU, and an AppWorld
score level with the official weights (NVFP4 / MXFP8, run exactly) on four fresh seeds (596 vs 592 of 672 task-runs,
p = 0.64). So on both a dense and a looped-MoE model the 12 GB build is not significantly different from its reference
weights on agent work (BF16; the official release), within a resolution of about 15 task-runs of 336 and 14 of 672
(the smallest difference the sign test detects at the observed number of discordant pairs). On the same model the harness gates also caught an engine fault (HTTP
500s in four agent harnesses, from one missed upstream code hunk) that the KL gate did not see: the faulty and the
fixed build had the same KL. We did not invent the quantizer, the trellis codec, or the 27B bit allocation; the contribution is the
diagnose-repair-confirm loop and evidence that it changes what ships and what is claimed.

## 1. Introduction

Post-training quantization (PTQ) makes a 27B model fit a 12 GB consumer GPU at about 2.5 bits per weight. Two
questions decide the result: which parts of the model keep more bits, and which runtime settings (K/V-cache type,
head precision, drafting) are safe. Most allocation methods answer the first question with a proxy (Hessian
curvature [HAWQ], activation salience [AWQ], reconstruction error [GPTQ]) or with an average output error (perplexity,
KL divergence [Unsloth; LookAtMyQuant]). Task accuracy is then measured once, at the end.

This paper reports what happened when we gated every bit map and every runtime setting on tasks that users of a
12 GB build actually run: recall of an exact fact or a code function 117k-245k tokens back, long chat answers, and
a multi-step agent benchmark. Three observations shaped the method.

1. **Average error hides task damage, and the reverse is also true** (Section 4.1). Maps with lower KL than the
   shipped map failed recall gates or long-chat checks, and on held-out documents one of six lost long-range recall
   margin that short-range probes did not show; the KL-only winner lost agent work on a harder suite. The damage was
   local: one band of layers. KL also found damage that our
   task probes missed. The same blindness holds for faults outside the weights: a build that passed the KL gate failed
   every agent harness.
2. **Local repair beats blanket precision** (Section 4.2). The cheapest safe fix was often a small, targeted change
   (one cache class, one projection role), and several plausible repairs did not pay and were rejected by the gates.
3. **Confirm on fresh data before any claim** (Section 4.3). The build we chose on two AppWorld seeds lost its lead
   over BF16 on two fresh seeds, and we withdrew a public head-to-head claim for the same reason. The same check,
   applied before a later public claim, held it. Applied to our own paper claims on fresh seeds of the same split, it
   removed two: an agent loss of a KL-chosen map did not replicate there (it did show on a harder split), and the size
   of the selection effect could not be separated from seed noise. So we state this step as a procedure, not as a
   measured effect.

**Contributions.**
- A reproducible loop (Section 3; full version in the supplement `evidence/ALGORITHM.md`): fit once, diagnose with KL and
  task probes, locate with one-group flips, repair under a byte budget, gate with pre-written lines, confirm on
  fresh data, and keep failures in the record.
- Evidence on two architectures (dense hybrid; looped MoE) that KL rank and task rank disagree at 2-3 bits, with
  margins that localize a recall loss to a layer band.
- On both architectures, a 12 GB build whose agent score is not significantly different from its reference weights
  on fresh seeds (BF16: 294 vs 293 and 292 vs 294 of 336 on two seed pairs; the official Inkling weights: 596 vs 592
  of 672).
- Ablations of the method on fresh seeds (without task probes, without confirmation, without per-class runtime
  repairs), with the result that two of our own claims did not hold on fresh seeds of the same split; we report
  them as such.

**Scope.** We did not invent the quantizer (exllamav3), the trellis codec (QTIP / EXL3), or the 27B bit map, which
is the L0xRE-V2 allocation by YourHighnessLA [L0xRE-V2], shipped unchanged. On the 27B model our method changed what did *not*
ship and what we claimed; on Inkling 2 Mini it chose the shipped map and runtime settings.

**Availability.** Kintsugi-27B: [GitHub](https://github.com/professorpalmer/Kintsugi-27B) and
[Hugging Face](https://huggingface.co/CaryPalmer/Kintsugi-27B). Kintsugi for Inkling 2 Mini:
[release v1.0](https://github.com/professorpalmer/Kintsugi-Inkling-2-Mini/releases/tag/v1.0) and
[Hugging Face](https://huggingface.co/CaryPalmer/Kintsugi-Inkling-2-Mini). Evidence, data and the scripts that
recompute every statistic and figure: the [evidence repository](https://github.com/professorpalmer/kintsugi-method)
(DOI [10.5281/zenodo.24455576](https://doi.org/10.5281/zenodo.24455576)).

## 2. Setting

**Models.** (a) Qwen3.8-27B: 64 layers, 16 full-attention and 48 Gated DeltaNet (linear-attention) layers [GDN]; 400
coded projections, each at 2 or 3 bits per weight (K2 / K3) [E4.1]. (b) Inkling 2 Mini: 24 layers, 128 routed experts
(top-8) per layer, layers 6-17 run three times with shared weights (48 layer applications, each with its own K/V
cache), 4 global-attention and 20 sliding-window layers; the official release is NVFP4 / MXFP8, with no BF16 release
[E4.2]. Its reference is the official weights themselves (NVFP4 / MXFP8, not BF16), run as Unsloth's UD-Q8_K_XL file,
whose every tensor equals our exact dequantization of the official release (max abs diff 0) [E4.10]. We call them
"the official weights", not "full precision".

**Codec and fits.** Every coded projection is an EXL3 trellis code [QTIP; EXL3] in the Escha W2 layout (EXL3 code
with Hadamard input / output scales) [Escha W2], fitted with exllamav3's quantizer.
For Qwen we used the public Hessians and golden set of a community quantization challenge (CC BY 4.0). For each
projection we keep a fit at K2 and at K3 (the *fit library*); a bit map selects one per projection, so no map needs
new fits.

**Engine.** A llama.cpp fork, with the Escha op for ggml [llama.cpp-escha], batch-invariant kernels and one slot. One file and one prompt give the same
logits on every run: two server starts gave identical margins to 0.0 nats, and a rented box matched our workstation
to 0.0000 nats [E1.15]. This makes single-run probes meaningful. For Inkling the engine adds our ports of the
upstream Inkling 2 model code (Piotr Wilkin, llama.cpp #32217), its chat parser (Aldehir Rojas, #32381) and Unsloth's
loop layout [E1.23].

**Measurements.**
- *KL.* Mean over scored rows of $\mathrm{KL}(p_{\mathrm{ref}} \,\|\, p_{\mathrm{quant}})$, with the challenge's
  top-512 + tail arithmetic. Reference: BF16 (Qwen) or the official weights dequantized exactly (Inkling). Qwen uses
  295 build documents (p1) and 294 untouched documents (p2); Inkling 298 documents. Intervals are 95 % paired
  bootstrap intervals over documents [Koehn].
- *Recall gate.* 24 needle questions (a planted 4-digit code) and 20 exact copies of a code function, at 117,000 and
  245,000 tokens, thinking off, greedy. Pass = 24/24 and 20/20.
- *Margin.* For answer token $t$ with right token $y_t$,
  $m_t = \log p(y_t \mid \cdot) - \max_{y \ne y_t} \log p(y \mid \cdot)$, teacher-forced on the gate's own prompt; the
  item margin is $\min_t m_t$. A negative item margin means greedy decoding fails that item.
- *Agent.* AppWorld test_normal [AppWorld], 168 tasks, temperature 1.0, two seeds pooled (336 task-runs). Exact
  two-sided sign test on discordant (task, seed) pairs. *Resolution* means one number here: the smallest difference
  with p < 0.05 at the observed number of discordant pairs. For our two-seed comparisons (45-52 discordant pairs) it is
  about 15 task-runs of 336 [E3.17]. The Inkling comparison with the official weights pools four seeds (672
  task-runs); at its 40 discordant pairs it is 14. Seed pairs: 100 + 101 chose the Qwen maps ("selection
  seeds"); 102 + 103 and 104 + 105 were never used for a choice ("fresh"). Intervals on a difference are 95 % paired
  bootstrap intervals over tasks [E3.17].
- *Engine and harness gates.* The KL gate: KL of the release build against a fixed reference. The harness gate: six
  coding tasks in each of four agent harnesses (Claude Code, Codex, Pi, OpenCode) through the release launcher, with
  hard lines (0 HTTP 500, 0 unfinished streams, 0 malformed calls, 0 loops). The bad-token check: broken UTF-8, stray
  special tokens, repeat runs and mixed-script words over about 80k tokens of long generated HTML.
- *Names.* Maps and probes have short descriptive names in the text (for example "KL-only winner"); Appendix A maps
  them to the names in the records and in `evidence/EVIDENCE.md`.
- *Code.* HumanEval [HumanEval] is saturated for every file we built (158-162 / 164) and serves only as a health gate
  [E4.9].

## 3. Method

The loop has seven steps (full inputs, outputs, costs and gate rules in `evidence/ALGORITHM.md`).

1. **Fit a library** of per-projection codes at each width (once; about 2 GPU-hours for the 27B model on one RTX 4070).
2. **Diagnose.** For each candidate file measure KL (build and untouched documents, by class, and a few named long
   documents) *and* task probes (recall margins at depth, the recall gate, the agent benchmark).
3. **Locate.** Flip one group (role x layer band) one width step from the start map; record each probe's change per
   MiB and the list of recall items that go below zero.
4. **Repair.** Move bits from groups that no probe flags to groups that raise the hard items, at equal bytes; treat
   runtime settings the same way (one cache class, one head type at a time). Measure every candidate; do not predict.
5. **Gate.** Write each pass line, with a time, before the run it judges. A missed line is a miss; a line found to be
   wrong is changed only for later runs, with the reason recorded.
6. **Confirm.** Re-run the winner and a never-selected reference on fresh seeds and held-out documents. Public claims
   follow the fresh result.
7. **Publish the seams.** Keep failed lines, rejected repairs and corrections in the public record.

![The loop. Every candidate (bit map or runtime setting) goes through steps 1-4; a missed line sends it back. Only
step 5, on data no choice used, decides what is claimed.](figs/fig1-workflow.png){width=100%}

Three rules came from failures and are now part of the method. *Probe validation:* a probe may rank groups only if it
ranks a known loser below the reference; our teacher-forced agent-trace probe did not (+0.0021 nats per token,
interval [-0.0030, +0.0072]) and was dropped [E1.14]. *Hard items:* a recall probe needs items that are hard for the
reference build; a probe whose items all had margins above 2.4 nats passed a map that lost a needle [E1.7]. *Model or
engine:* when a task gate fails on behaviour that could come from the model, write the rule that separates the two
before the check (the same task repeated on our build and on a reference build, and the model's own token
probabilities at the failing step), and keep the failed gate in the record whatever the check says [E1.25].

## 4. Experiments

### 4.1 Average error hides task damage (C1)

**Lower KL, failed tasks (Qwen3.8-27B).** Table 1 lists every map we built that beat the shipped allocation on KL.
None shipped.

*Table 1. Bit maps with lower KL than the shipped allocation, at the same size (within 0.1 %). AppWorld: S = selection
seeds 100 + 101; fresh = seeds 104 + 105.*

| Map | How chosen | KL vs start | Recall at 117k | AppWorld vs start, S; fresh | Verdict | Ref. |
|---|---|---:|---|---|---|---|
| KL-group | measured KL per group | -9.3 % [-10.5, -8.3]; p2 -8.2 % | 23/24 + 19/20 (a needle lost) | not run | fail | [E1.3] |
| KL-table A | KL table | -8.5 % (subset) | 24/24 + 19/20 (fragile item only) | not run | fail as written (weak) | [E1.6] |
| KL-table B | KL table | -10.6 % (subset) | needle margin -1.94; held-out (B1): long-range margin loss | not run | fail | [E1.7, E1.34] |
| KL-only winner | KL table | -7.0 % [-8.3, -5.6] | 24/24 + 20/20 (also at 245k) | 285 vs 302 (p = 0.019); 296 vs 292 (p = 0.68); harder split: 610 vs 670 of 834 (p = 1.3e-5) | long-chat flag; loses on the harder split | [E1.8, E1.31, E1.36] |
| KL-tools | KL table, tools-aimed | -7.9 % [-9.1, -6.9] | 24/24 + 20/20 | 301 vs 302 (p = 1.00); 299 vs 292 (p = 0.37) | no gain; long-chat flag | [E1.9, E3.16] |
| Uniform (baseline) | uniform rule: 3 bits on the last 30 layers | -3.8 % [-6.0, -1.6]; p2 -1.2 % [-6.6, +4.4] | 24/24 + 19/20 (fragile item only); more margin on the other 43 | not run (rule) | fail as written (weak); chat +13.6 % | [E1.28, E1.33] |

KL-group was better than the start on every KL line, including the untouched half, every document class, and every depth
band up to 256k tokens in both KL and the model's own NLL of the true next token [E1.4]. It still lost one needle and
one code copy at 117k. The KL-only winner passed every KL, recall and code gate. On the selection seeds it lost the agent
benchmark by 17 task-runs (difference -17 [-30, -3]); on two fresh seeds it was level (+4 [-10, +18], p = 0.68), so
that loss did not replicate on that split [E1.8, E1.31, E3.17]. On the harder test_challenge split (417 tasks that no
choice used; the third of the author's three tests, with its line written before the run) it lost 60 of 834 task-runs:
610 against 670, 63 against 123 discordant, p = 1.3e-5, difference -60 [-87, -33], and on each seed (106: 304 against
335, p = 0.0017; 107: 306 against 335, p = 0.0035) [E1.36, E3.17]. Both results stand: on the standard split the two
files were level on fresh seeds; on the harder split the KL-only winner is clearly worse. The cost of KL-only selection
on agent work shows on the harder tasks. The other thing against it is long chat:
on two long chat documents of the untouched half its KL was 1.120 and 1.051 against 0.698 and 0.337 for the start
[E1.31]. So the C1 evidence on the 27B model rests on recall (KL-group and KL-table B lost a needle at 117k; KL-table
A failed only on the fragile item, which we count as weak evidence for it as for the baselines) and on long-chat
documents, and on the harder agent split, not on the standard one. The held-out test below (B1) checks the recall
part on material no choice used.

![Average error against task outcome on the 27B model. (a) Recall-gate items lost at 117k for every map with lower
KL than the shipped allocation, and for the two baselines; open circles mark the one fragile code-copy item, counted
as weak evidence. KL changes marked * are on the build subset T, the others on the full build half p1 [E1.3,
E1.6-E1.9, E1.28, E1.29]. (b) AppWorld difference to the shipped file for the two lower-KL maps that ran it: on the
selection seeds (filled) and on fresh seeds 104 + 105 (open), with 95 % paired bootstrap intervals over tasks. The
agent loss of the KL-only winner did not replicate on this standard split; on the harder test_challenge split (not
shown) it lost 60 of 834 task-runs [-87, -33] [E1.8, E1.31, E1.36, E3.16, E3.17].](figs/fig2-kl-vs-task.png){width=100%}

**Held-out recall population (B1).** After an external review, the author fixed three tests and their lines before
any of them ran, with a family alpha of 0.0167 (Bonferroni over the three); no other test is added until one of them
passes. The first asks whether a lower-KL map loses recall on material that no choice used. It has 20 haystack
documents built from standard-library code that no recall gate of the selection used, 240 needles and 240 distinct
functions at 64k, 117k and 245k tokens, and a shallow twin of every 117k and 245k item: the same question on a
haystack of about 3.9k tokens around the item, so that only the distance changes [E1.34]. Per map we compare the capped
item margin with the shipped file: L over the long items, S over their twins, and the depth contrast C = L - S, with
99.72 % bootstrap intervals over documents (Bonferroni over the six lower-KL maps inside the family alpha). The floors
come from a repeat run and from a second fit of the shipped map (0.212 nats for L, 0.295 for C).

One of the six lower-KL maps meets the line. KL-table B (KL -10.6 % on the build subset) has L = -1.07 nats [-1.28,
-0.86] and C = -1.14 [-1.36, -0.94], with no loss at short distance (S = +0.07) (Figure 3) [E1.34]. The loss is on
needles at depth: -1.23 nats at 117k and -3.26 at 245k, against -0.16 at 64k; the copies are flat. It is a loss of
margin, not of answers: no long item went below zero, so greedy decoding still answers every item. The other five
lower-KL maps do not meet the line. This includes KL-group, whose needle loss on the gate's one document (Table 1) is
small on held-out documents (L = -0.13 [-0.24, -0.03], inside the refit floor). So C1 on the 27B model is this: one
of six maps with lower average KL lost long-range recall margin on held-out documents, which short-range probes and
average KL did not show; the evidence is a margin, not a wrong answer, and it rests on the depth contrast.

Confirmed on one box: the first run paired the two files across two GPU types (RTX 4070 SUPER and RTX 4070), so we
reran both on one machine with one GPU model, with the same line written first (B1c). There the run floor was 0.000
(a repeat on another GPU gave the same margin on 40 of 40 records), and KL-table B again met every condition: L =
-1.11 [-1.34, -0.88], C = -1.18 [-1.40, -0.96], S = +0.07 [+0.03, +0.12]; needles -1.38 at 117k and -3.25 at 245k,
copies flat, no wrong answer [E1.35]. The shipped file gave the same margins on 800 of 800 records on the two machines
of the same type.

![Held-out recall population test (B1). (a) Per map against the shipped file: L (long items, 117k + 245k), S (shallow
twins, about 3.9k tokens) and the depth contrast C = L - S, with 99.72 % bootstrap intervals over the 20 documents
and the two floors of the line; the shaded maps are not counted (higher KL; the refit arm). Only KL-table B has L and
C below their floors with S above. (b) KL-table B by depth and family: the loss is on needles at 117k and 245k;
copies and twins are flat. Confirmed on one box with one GPU model (run floor 0.000) [E1.34, E1.35].](figs/fig3-b1-recall-population.png){width=100%}

A local proxy did worse than KL: a map chosen by the sum of per-projection Hessian-weighted fit errors was 36.7 %
*worse* in KL at the same size, because it took bits from the last MLP layers, whose own fit error is small but whose
effect on the output is large (+46.2 % KL for that family alone) [E1.1, E1.2]. A HAWQ-V2-style map (average Hessian
trace times the squared fit error, exact knapsack at the same bits) was 12.7 % worse in KL [+10.6, +14.7] and also
lost one code copy [E1.29]. A uniform baseline (3 bits on the last 30 layers) beat the shipped map on KL on the build
half (-3.8 % [-6.0, -1.6]; untouched half -1.2 % [-6.6, +4.4], chat documents +13.6 %) and lost the same code copy
[E1.28]. Both baseline misses are on the one fragile item of Section 6 (an exact function body with the comment
line above it copied too), so we count them as weak evidence; the shipped file passed that item in the same run.

Graded margins on the same 44 items at 117k (the margin probe of this section, with lines written before the run)
show no general recall loss for either baseline; they point the other way [E1.33]. On the 43 items without the fragile
one, the mean difference of the capped item margin against the shipped file was +0.164 nats [+0.016, +0.379] for the
uniform map and +0.148 [+0.016, +0.356] for the HAWQ-style map: by the line written first, both have more recall
margin than the shipped map. The gain sits in few items. Most items are at the 5-nat cap in every arm; the difference
comes from the two hardest needles of the shipped file (vault 8 and vault 10, +1.7 to +3.7 nats) and from one copy
function asked three times (+0.34 each). Counted once per function, the lower end of both intervals is +0.000. On the
fragile item both baselines are below zero (-0.325 and -0.258, against +0.264 for the shipped file), so their
recall-gate misses are that item only. On KL the picture is mixed: the uniform map is level on the untouched
half (-1.2 % [-6.6, +4.4]) and 13.6 % worse on its chat documents, and the HAWQ-style map is 12.4 % worse on both
halves [E1.28, E1.29].

![Graded recall margins per item, sorted, capped at 5 nats. (a) Qwen, 117k, the 44 recall items: uniform and
HAWQ-style baselines against the shipped file; 6 of 44 items differ, the rest sit at the cap in both arms [E1.33].
(b) Inkling 2 Mini, 117k and 245k (88 item-depth units): blanket q4_0, blanket q8_0 and the 4-bit global cache
against the release K/V setting; wide spread, no clear shift of the mean [E2.23].](figs/fig4-item-margins.png){width=100%}

**A naive 2-bit file on fresh seeds.** On fresh seeds, the shipped build beats the naive 2-bit file (Unsloth
UD-Q2_K_XL, a K-quant) in the same engine by 36 task-runs of 336: 292 against 256 on seeds 104 + 105, 51 against 15
discordant, exact sign test p = 1.0e-5, difference +36 [+20, +52] [E1.32, E3.17]. This is the second of three tests
that the author fixed, with a family alpha of 0.0167, before their results existed, and it passes. It compares a
trellis build with a K-quant file: the codec, the fits and the bit allocation all differ, so it does not isolate the
allocation method and is not evidence for C1. The naive file ran with its own chat template; the first prompt of every
task-run had the same token count as the shipped file's (336 of 336). On the selection seeds the gap was 35 (302
against 267) [E1.20].

**The recall loss is local.** Margins on the gate's own prompts turned the pass / fail needle into a continuous,
exact measurement. The failed needle ("vault 8") had a margin of +1.62 nats on the start map and between -1.6 and
-2.1 on every map that moved the MLP up-projections of layers 21-47 to 2 bits; every map without that change kept it
[E1.5]. A one-group flip table on the shipped file (21 rows) agrees, and adds two small groups: the attention output
projections (15 MiB) and the attention key / value projections (20 MiB) each pushed the needle below zero [E1.11].
KL rates the layer 21-47 up-projections among the cheapest sources (0.024-0.026 % KL per MiB) [E2.13].

**The reverse also holds.** KL rates the up-projections of layers 56-63 as the *worst* source (0.148 % per MiB).
They broke no recall item. A candidate that took all of layers 48-63 kept every recall item above zero but was
8.1 % worse in KL on the untouched documents [+3.2, +13.2], and 8-17 % worse on five document classes [E1.12]. Both
signals are needed. Over the 16 groups with both numbers, the rank correlation of KL cost and task cost was
$\rho = 0.007$ [-0.56, 0.62] [E1.10]; this interval is wide because most single-group task changes are inside their
noise, so we do not read it as evidence that the rankings differ, only that KL does not predict the task ranking
here. The task-guided search itself found no map that beat the start: the best of five task-guided maps missed one line
(a 356-row non-English chat document) and the study is paused [E1.13].

**A second model.** On Inkling 2 Mini our map (M2) has 27.3 % lower KL than Unsloth's
2-bit file on the scored rows in our engine (the gate's own measurement; a PyTorch check gave 26.8-27.0 %) [E2.6], but
25 % *higher* KL on the rows that predict user and tool text past 1k tokens ("copy rows": 0.760 vs 0.608 in the
engine) [E1.16, E1.18]. Recall did not
move: every arm (official, Unsloth, ours) got 24/24 needles and 20/20 code copies at 16k, 32k and 61k. We report the
copy-row metric but do not rank by it: it saturates (0.19 with no weight change; 0.885 with only 8-bit non-expert
tensors), so it seems to measure how unstable the prediction of tool output is under any change, not how much was
quantized [E1.17]. A third-party file (LookAtMyQuant, 8.9 GB) is a second case of average KL hiding where the damage
sits: it is level with ours on average KL (-3.3 %, interval [-0.0037, +0.0016]) but 62 % higher on copy rows, and by
document group 25 % better on English prose and 29-83 % worse on code, tools, agent traces and reasoning [E1.19]. Two
files with the same average differ by class by up to a factor of almost two. The copy-row deficit did not show on the
agent benchmark against Unsloth: M2's setup scored 298 against 283 for Unsloth's setup on two seeds and 298 against
281-283 on two more [E1.27]. That comparison is setup against setup (engine, K/V type and reasoning budget differ), so
it does not isolate the weights; the weights-only comparison is with the official weights in our engine, which score
592 of 672 on the same four seeds against 596 for M2 [E4.8].

**Faults outside the weights (Inkling 2 Mini).** The release candidate engine passed its
KL gate (KL 0.003341, top-1 97.727 %) and failed the harness gate: 4/6, 4/6, 4/6 and 5/6 tasks in Claude Code, Codex,
Pi and OpenCode, with HTTP 500 "empty grammar stack" errors in every harness [E1.21]. The cause was one upstream code
hunk that our port had missed: the lazy tool-call grammar's anchored trigger fell back to a regex search, and the
MSVC regex library matches ^ and $ at line breaks, so a blank line inside an ordinary text answer started the tool
grammar [E1.22]. The upstream parser change (Aldehir Rojas, llama.cpp #32381) has the two-line guard. With it, a replay
of all 202 captured requests and 440 sampled stress requests gave 0 errors, and KL was unchanged to the last digit
[E1.23]. An audit of every ported upstream change then found a second missed hunk of the same change and an
off-path multi-slot bug [E1.23]. This is not a quantization effect, and we do not count it as evidence about bit maps.
It is the same lesson at the level of the shipped system: a gate on the average next-token distribution cannot see a
fault on the task path, so the release gates include the harnesses themselves.

The rerun on the fixed build had 0 server errors in all four harnesses, but one hard line still failed: in one
OpenCode task the model wrote a bash heredoc into a PowerShell tool and then looped on trivial commands [E1.24].
Two checks with rules written before the run separated model behaviour from an engine or quantization fault. At the
stuck turns the model itself put 0.96-1.00 of its probability on the "bash" tool name, with the grammar on or off;
and the same task ran 10 times on our build and 10 times on Unsloth's build with 0 loops on each [E1.25]. The gate
record stays "failed"; the release states the behaviour as a known limit. A separate bad-token check found 0
malformed tokens in 81,646 generated tokens on the final build [E1.26]; it is reported as a passed gate, not as
damage found.

### 4.2 Local repair beats blanket precision (C2)

**K/V cache, by cache class (Inkling 2 Mini).** The model has 48 K/V caches: 8 for global-attention layers, which
grow with context, and 40 for sliding-window layers, which are rings of about 1,024 cells. A blanket q4_0 cache was
far outside our line (KL 0.0092 at 2k against the same weights with f16 K/V) [E2.1]. Table 2 shows the scan. The
40 sliding-window caches cost 240-320 MiB at f16 at any context length, while the global caches need 32 KiB per token
at f16 and cannot hold 262k tokens on 12 GB [E2.3]. So the repair spends precision where it is cheap per byte: f16
for the sliding-window caches and the cheapest type that passes the line (q8_0) for the global ones.

*Table 2. K/V type per cache class, KL against the same weights with f16 K/V (Unsloth's file, so only the K/V type
changes; WikiText-2, 20 x 2,048 tokens) [E2.1].*

| Global caches (8) | Sliding-window caches (40) | KL | Top-1 |
|---|---|---:|---:|
| q4_0 | q4_0 | 0.00924 | 96.13 % |
| q4_0 | f16 | 0.00560 | 96.92 % |
| q8_0 | q8_0 | 0.00356 | 97.70 % |
| **q8_0** | **f16** | **0.00271** | **98.02 %** |
| K q8_0, V q4_0 | f16 | 0.00367 | 97.53 % |

At 32k tokens the chosen setting gave KL 0.00251 (line <= 0.005, pass) and top-1 97.97 % +- 0.06 %, which misses the
98 % line written before the run by less than one standard error; we report it as failed as written [E2.2]. On our
own file the setting costs +0.00016 KL against f16 K/V [E2.4]. The repair is shown on KL and memory, not on recall:
an ablation with blanket q4_0 and blanket q8_0 on all 48 caches kept 24/24 needles and 20/20 copies at 64k, 117k and
245k, as the release setting did; the line written before that run asked for a recall miss in a blanket arm [E2.22].
Because pass / fail recall hit its ceiling, we measured the same 44 items at 117k and 245k as graded margins (the
smallest margin over the answer tokens, capped at 5 nats), paired against the release setting, with lines written
before the run [E2.23]. Even graded recall margins do not separate the blanket settings from the per-class one at
this resolution: the mean difference was -0.035 nats [-0.213, +0.146] for blanket q4_0 and -0.084 [-0.224, +0.061]
for blanket q8_0. A repeat of the release arm gave the same margin on every item (noise floor 0.000 nats), so the
width comes from the spread over items, not from run noise. The intervals resolve shifts of about 0.2 nats, not less.

**The same rule, one step further: 4-bit global caches.** At 245k tokens the q8_0
global caches do not fit in VRAM: 114,688 cells stay on the GPU and the rest is streamed from host memory, so decode
falls to 8.7 tokens per second. q4_0 halves the bytes per cell (9,216 B against 17,408 B for the 8 global
applications), so 250,112 cells fit in the same 2.2 GB and the host tail disappears up to 250k tokens; the 40
sliding-window caches stay f16 [E2.15]. The gates were written before any run. On Unsloth's file this setting had
missed the KL line (0.005135 at 32k); on M2 it passed (0.004552 +- 0.000047 against the line 0.005) [E2.16]. Decode
at 245k rose from 8.7 to 30.2 tokens per second (3.47x, line 13.05) and at 117k by 1.34x, with no change at 32k
(0.97x, inside the run-to-run band) [E2.17]. A KL pass is not enough to ship it: the setting also needs full recall at
117k and 245k, and it got 24/24 needles and 20/20 copies at both [E2.18]. The measured gain came from removing the
host tail, not from faster 4-bit attention, as the byte model predicted [E2.17]. It became the default of the published release (v1.0); below, "the release setting" means the q8_0-global
setting that the tests compared against. On AppWorld it scored 595 of 672 against 592 for the official weights and 596 for the 8-bit
cache (p = 0.78 and 1.0) [E4.12]; its KL against the official weights is measured only on a 10-document subset. On
graded recall margins the result is inconclusive: -0.120 nats [-0.320, +0.090] against the 8-bit global cache. The
interval is not below 0, but its lower end is below the non-inferiority margin of -0.25 nats that we wrote before the
run. One copy item (`_decimal_lshift_exact`, asked twice at 117k) went from +1.76 and +1.14 nats to -0.10 and -0.41,
at the third-last answer token [E2.23]; the pass / fail gate had counted every copy at this depth as correct [E2.18].

**Bit map (Inkling 2 Mini).** Our engine supports 2- and 3-bit trellis codes. Two maps were named before any result:
an expert-layer map (both expert matrices at 3 bits in 11 layers) and M2 (all down projections at 3 bits, gate/up at 3 bits only in
the 4 global layers). Both passed the line (at least 10 % lower KL than Unsloth's UD-Q2_K_XL, interval below zero,
top-1 not lower, file not larger); M2 was better and smaller [E2.5]. In our engine M2 has mean KL 0.0310 against
0.0426 for Unsloth's file in the same engine, -27.3 % [diff -0.0133, -0.0102], lower on 281 of 298 documents and in
every document class, with top-1 93.6 % against 92.7 %, at 9.07 GB against 9.14 GB [E2.6, E2.7]. One structural line
(per-document correlation of engine and simulation >= 0.95) failed as written (r = 0.919); the same statistic for
Unsloth's file was 0.912, and on documents with at least 500 scored rows it was 0.997, so the line sat below the
noise floor of short documents. We record the miss and the decision to proceed separately.

**Shipped runtime settings cost nothing measurable on tasks.** The shipped 27B build (q4_0 K/V tiered to host
memory, two-stage head, drafting) scored 294 of 336 on fresh AppWorld seeds; the same weights with f16 K/V all in GPU
memory and no head or drafting changes scored 295 (19 vs 20 discordant, p = 1.0; difference -1 [-13, +11])
[E2.14, E3.17]. On Inkling 2 Mini, M2 with
the release K/V setting (q8_0 global + f16 sliding-window) against the official weights with f16 K/V everywhere scored
596 against 592 of 672 [E2.19]; that test covers the weights and the K/V setting together and does not separate them.

**Repairs that did not pay.** See Section 5.

### 4.3 Select, then confirm on fresh data (C3)

On AppWorld seeds 100 and 101 the shipped 27B build was compared with three alternatives (two of our KL-guided maps
and the original author's own fits) and kept as the best of the four: 302 against 301, 288 and 285 [E3.8]. A comparison with BF16 on the same seeds then showed the
quantized build *ahead* of BF16 by 21 task-runs (p = 0.0025) [E3.6]. Rather than report that, we re-ran
both on two seeds never used for any choice, and later all four candidates on two more (Table 3).

*Table 3. AppWorld test_normal, pooled task-runs of 336 per seed pair. S = selection seeds; the other two pairs were
never used for any choice. Gap intervals: 95 % paired bootstrap over tasks [E3.17].*

| Arm | S: 100 + 101 | 102 + 103 | 104 + 105 | Ref. |
|---|---:|---:|---:|---|
| Kintsugi-27B as shipped (12 GB GPU; the selected candidate) | 302 | 294 | 292 | [E3.1, E3.2, E3.15] |
| KL-tools (candidate) | 301 | not run | 299 | [E3.8, E3.16] |
| Author's fits of the shipped layout (candidate; its own chat template) | 288 | not run | 289 | [E3.8, E3.16] |
| KL-only winner (candidate) | 285 | not run | 296 | [E3.8, E1.31] |
| Kintsugi-27B weights, f16 K/V (80 GB GPU) | not run | 295 | not run | [E3.3] |
| BF16 (never selected) | 281 | 293 | 294 | [E3.4, E3.5, E3.16] |
| Inkling 2 Mini, Unsloth UD-Q2_K_XL (never selected) | 283 | 283 | 281-283 | [E3.9, E3.13] |
| **Gap shipped - BF16** | **+21 [+8, +34], p = 0.0025** | **+1 [-14, +17], p = 1.0** | **-2 [-16, +11], p = 0.89** | [E3.6, E3.7, E3.16, E3.17] |

On seeds 102 + 103 the quantized build is not significantly different from BF16 (294 vs 293; weights-only 295 vs
293, 23 vs 21 discordant, p = 0.88) [E3.7]. This is a non-significant difference, not a proof of equivalence: two
seeds resolve about 15 task-runs (the resolution defined in Section 2). On seeds 104 + 105 it was 292 vs 294 (27 vs 25 discordant, p = 0.89) [E3.16]:
again not significantly different. The +21 on the selection seeds is not a result.

![The selection-seed lead. (a) AppWorld score per seed pair for the shipped (selected) file, BF16 (never selected)
and the three other candidates of the selection; candidates other than the shipped file have no run on 102 + 103.
(b) The shipped - BF16 gap with 95 % paired bootstrap intervals: +21 on the selection seeds, +1 and -2 on the two
fresh pairs [E3.1-E3.8, E3.15-E3.17].](figs/fig5-seed-pairs.png){width=100%}

How much of it was selection? We wrote the test before the run: on seeds 104 + 105, the selection effect
$\Delta_{\mathrm{sel}}$ is the winner's drop from S minus the mean drop of the other three candidates, and we claim
"the selection seeds inflated the winner" only if its bootstrap interval over tasks is above 0. The winner fell by 10;
the others moved by -2, +1 and +11 (the KL-only winner rose from 285 to 296). So $\Delta_{\mathrm{sel}} = +13.3$ task-runs,
95 % interval [+0.0, +27.0] [E3.15]. The interval is not above 0. The drop is within seed-to-seed variation; we cannot
separate selection bias from noise with this data. The never-selected BF16 arm also moved: by +12 from the selection
seeds to 102 + 103 [E3.10] and by +13 to 104 + 105 [E3.15].

The same check overturned a public claim. Our published head-to-head against Inkling 2 Mini (Unsloth's 2-bit file on
its own build) said the 27B build was better on AppWorld (302 vs 283, p = 0.0079). On fresh seeds it was 294 vs 283,
27 vs 16 discordant, p = 0.13, and we withdrew the claim on the public page the same day [E3.9]. Inkling's score was
the same on both seed pairs.

**The check also holds a real result.** Our Inkling build against Unsloth's setup gave
298 vs 283 on seeds 102 + 103 (p = 0.0135) [E3.12]. No choice for M2 used AppWorld, so this was not a selected
winner, but because of the withdrawal above we re-ran both setups on seeds 104 + 105 before any public claim, with the
rule written first: keep "better" only if ours is at least equal there. It was 298 vs 281-283 (two Unsloth task-runs
did not finish at the output cap, so we give a bound; p = 0.0081 on the 334 finished pairs) [E3.13]. Table 4 shows
that no Inkling arm moved by more than 4 task-runs between seed pairs [E3.14], against +12 for the never-selected BF16
arm on Qwen. This is a setup-against-setup result (engine, K/V type and reasoning budget differ), so it is a product
comparison, not a weight-only one; against the official weights in the same engine M2 is level (596 vs 592 of 672)
[E4.8].

*Table 4. Inkling 2 Mini, AppWorld test_normal, task-runs of 336 per seed pair. No arm was chosen on AppWorld. The
Unsloth row is setup against setup; the official-weights row is the weights-only comparison (same engine).*

| Arm | 100 + 101 | 102 + 103 | 104 + 105 | Ref. |
|---|---:|---:|---:|---|
| Ours (M2, our engine, release K/V) | not run | 298 | 298 | [E3.12, E3.13] |
| Official weights (NVFP4 / MXFP8 as UD-Q8_K_XL, f16 K/V, our engine) | not run | 298 | 294 | [E4.8] |
| Unsloth's setup (UD-Q2_K_XL, Unsloth's build) | 283 | 283 | 281-283 | [E3.9, E3.13] |

**Lesson (a procedure, not a measured effect).** This is itself a result of the method. Before the last batch we
held two claims that came from the selection seeds: that KL-only selection costs agent work (Section 4.1), and that
the selection seeds inflated the winner by a measurable amount. We tested both on seeds of the same split that no
choice had used, and neither held. The first then held on a harder split that was named, with its line, before it
ran (Section 4.1); the second we keep as a procedure: a configuration chosen by its score on a benchmark needs a held-out
confirmation before any comparison with another model or with its reference weights is claimed, because of the winner's
curse [Thaler; Cawley & Talbot], whose size our data could not measure. The check removed one public claim and two
of our own paper claims on the standard split (one of them held on the harder split), and it kept one.

### 4.4 Transfer across architectures (C4)

The same loop ran on both models with the same tools: fit library, KL gate with paired bootstrap, recall gate at depth,
pre-written lines. Both builds run on one 12 GB GPU with a 262,144-token window (K/V past the GPU limit is kept in
pinned host memory and streamed; the host path matched the all-GPU path to KL < 1e-6 [S.1]).

| | Qwen3.8-27B (dense hybrid) | Inkling 2 Mini (looped MoE) |
|---|---|---|
| File | 10.55 GB | 9.07 GB |
| KL | 0.0579 vs BF16 at shipped settings [E4.5] | 0.0310 vs official; -27.3 % vs Unsloth [E2.6] |
| Recall 24 + 20 | full at 117k and 245k [E4.3] | full at 64k, 117k and 245k [E4.4] |
| Agent vs reference weights (fresh seeds) | 294 vs BF16 293 of 336, p = 1.0 [E3.7]; 292 vs 294 on seeds 104 + 105, p = 0.89 [E3.16] | 596 vs official weights 592 of 672, p = 0.64 [E4.8]; with the 4-bit global cache 595, p = 0.78 [E4.12] |
| What else differs in that test | card (12 GB vs 80 GB), K/V, head, drafting; weights-only arm 295 [E3.3] | K/V precision, card (12 GB vs 48 GB) [E4.8] |
| Code (health gate) | HumanEval 159-161 / 164 [E4.9] | HumanEval 158 / 164 [E4.11] |

![Inkling 2 Mini on AppWorld, per seed (102-105, none used for a choice): M2 with the release K/V setting, M2 with the
4-bit global cache, the official weights (NVFP4 / MXFP8) in the same engine, and Unsloth's setup (setup against
setup; two of its task-runs on seed 105 did not finish). Totals in the legend [E3.12, E3.13, E4.8, E4.12]. Recall
was full (24/24 + 20/20) at 64k, 117k and 245k [E4.4, E2.22].](figs/fig6-inkling-appworld.png){width=100%}

What transferred is the procedure and three findings: KL alone does not decide (Section 4.1), per-class runtime
repairs pay in KL and memory, though neither pass / fail recall nor graded recall margins could see the difference
(Section 4.2), and on fresh seeds the 12 GB build is not significantly different from its reference weights (BF16;
the official NVFP4 / MXFP8 release) on agent work for both a dense and a looped-MoE model. Both agent results are "not significantly different", not proofs
of equivalence: the tests resolve about 15 of 336 and about 14 of 672 task-runs. What did not transfer: the 27B
allocation is not ours, and only the Inkling map was chosen by our method. On Inkling that choice was made on KL (the
recall checks were full for every arm), so a KL-only selection would have picked the same map: C4 is the transfer of
the procedure and its gates (they set the K/V choice and caught an engine fault), not evidence that task probes change
the map choice on that model.

## 5. Negative results (repairs measured and rejected)

All lines below were written before the run. Rejected items stay in the record.

| Repair | Idea | Measured | Decision | Ref. |
|---|---|---|---|---|
| M2 with a 4-bit head | Q4_0 output head, saves 232 MB | +6.9 % KL vs M2 (diff +0.0021 [+0.0019, +0.0024]) | option note only | [E2.8] |
| Copy-aware map | bits to layers that help copy rows | +10.3 % KL on scored rows; copy rows only -2.4 % | not shipped | [E2.9] |
| Borrowed allocation | LookAtMyQuant's byte allocation on our fits | +43 % KL (multilingual +153 %) | not shipped | [E2.10] |
| Exact attention / exact experts | fix copy rows with precision | attention BF16 or F16: no gain; all downs exact: -2.6 %; all gate/ups exact: -13 % (far over budget) | no lever | [E2.11] |
| Drafting on the MoE | block drafter + a draft-only head copy | +7.9 % decode at 4k, pooled over 4 runs, against a +8 % line (one run pair: +8.6 %, not quoted) | next version | [E2.12] |
| Drafting at long context (MoE) | the same drafter past 100k | line +20 % at 117k and 245k; 117k 18.2-20.0 vs 32.3 tok/s (the drafter's memory pushes K/V to the host); 245k 8.5-9.3 vs 8.6 | not shipped | [E2.21] |
| Sparse attention past 200k (27B) | read V only for high-weight rows of the host tail | speed 1.31x at 245k and full recall (24/24 + 20/20), but 0.55x on copy-heavy text (line 0.95x) | not shipped | [E2.20] |
| Five task-guided 27B maps | bits by measured task damage | each failed one line; best missed one document | start map stays | [E1.13] |
| Agent-trace probe | NLL of fixed agent actions | could not see a known loser | probe dropped | [E1.14] |

Sparse attention shows why one task gate is not enough even for a runtime setting: it passed the speed line and both recall
lines, and lost on the third line, copy-heavy text. The likely cause (not yet profiled) is that drafting makes each
step verify about 27 rows there, not the 5 rows the kernel was built for [E2.20].

The borrowed allocation shows why copying an allocation across codecs fails: the third-party file stores cheap rows of the head and
embedding as 2-3 bit trellis codes, while the ggml types available to us at that size are much coarser, so the same
byte split costs far more [E2.10].

## 6. Limitations

- **Small n.** Agent results use one benchmark, 168 tasks (417 on the harder split) and two seeds per comparison
  (four for the Inkling test against the official weights); differences under about 15 task-runs read as "not worse". The 27B fresh-seed confirmation
  against BF16 is two separate pairs of seeds (two seeds each). The recall gate has 44 items at
  two depths, and the decisive items are single needles (we treat them as signs, not rules).
- **One fragile item.** Three failures (KL-table A and the uniform and HAWQ-style baselines) are on one code copy
  that is near the edge on the start map (margin 0.43; 0.26 in the graded probe); in the baselines the body was exact
  and the comment line above the function was copied too. The gate counted all three, as written, and the text treats
  all three the same: a failed gate, weak evidence. Graded margins support that reading for the baselines: on the
  other 43 items they have more margin than the shipped file [E1.33]; KL-table A has no graded run.
- **Selection inside the study.** The fit library, the probes and the named documents were built and chosen on the
  same golden-set half (p1). The untouched half and fresh seeds limit, but do not remove, this.
- **Confounds in the BF16 rows.** BF16 ran on 80 GB GPUs and the shipped build on 12 GB GPUs; the weights-only arm
  (same engine, same card class as BF16) removes the K/V and card difference for one comparison only. The Inkling
  official-weights arm differs from ours in the weights and the K/V precision, and ran on 48 GB cards; it has no
  weights-only arm.
- **Setup against setup.** Our Inkling build against Unsloth's differs in the engine, the K/V type and the reasoning
  budget; the "better" result is about the two shipped products, not the two weight files. Two Unsloth task-runs on
  seed 105 did not finish (bound given).
- **Copy-row KL** saturates and its meaning is unclear (Section 4.1).
- **Runtime repair on tasks.** The per-class K/V choice changed KL and memory but not pass / fail recall (blanket
  arms were full at every depth) and not graded recall margins at a resolution of about 0.2 nats. On AppWorld only
  the global cache type was varied on fixed weights (4-bit 595 vs 8-bit 596 of 672 [E4.12]); the blanket settings
  have no agent score.
- **Repeated copy items.** The copy bench can ask for the same function in adjacent slots (117k: 13 distinct
  functions in 20 slots), so the 44 items are not 44 independent questions. Two post-hoc readings (not the statistic
  fixed before the run, so not claims) put the blanket q8_0 interval below 0: each function counted once (-0.157
  [-0.285, -0.028]) and the needles alone (-0.189 [-0.335, -0.069]); the needles alone also put the 4-bit global
  cache below 0 [E2.23]. A follow-up needs more distinct items, with its lines written first.
- **A failed gate stays failed.** The Inkling harness gate on the final build failed one hard line (one OpenCode loop);
  we report the model-behaviour reading from checks with rules written first, not a pass.
- **Saturated code benchmark.** HumanEval cannot rank our files.
- **Hardware.** One GPU family for all speed and recall numbers; other architectures were checked by KL only. The
  Inkling release engine also ran on an Ampere card (RTX 3080 Ti): KL within 0.6 % of the RTX 4070 value once one
  near-tie token is set aside, and 24/24 needles at 32k [E4.13].
- **Ablations.** Task probes are shown to matter on recall, long chat and the harder agent split, not on the standard
  agent split [E1.31, E1.36]; the
  selection effect is not separated from noise [E3.15]; per-class runtime repairs show no recall difference [E2.22,
  E2.23]. Seeds 106 + 107 were reserved and went to the one harder test (AppWorld test_challenge). The author's three
  named tests (B1, the naive file, test_challenge) all passed at the family alpha; no other test was added.
- **Every missed line.** Table 6 lists every pre-written line that a run missed and what we did.

*Table 6. Every pre-written line that a run missed, and what we did. Lines that rejected a candidate map or repair
are in Table 1 and Section 5; this table adds the misses of our own claims, settings and gates.*

| Line (written before the run) | Measured | What we did | Ref. |
|---|---|---|---|
| Inkling K/V setting: top-1 >= 98 % at 32k | 97.97 % +- 0.06 % | failed as written; setting kept on the KL line, which passed | [E2.2] |
| Inkling engine vs simulation: per-document r >= 0.95 | 0.919 (0.997 on documents with >= 500 rows) | failed as written; proceeded, decision recorded | [E2.6] |
| Harness gate hard lines on the final Inkling build: 0 loops | 1 OpenCode loop | failed; model-behaviour checks with rules written first; stated as a known limit | [E1.24, E1.25] |
| Inkling loop guard: >= 8 / 10 runs unstuck at each stuck point | 6, 6 and 7 / 10 (0-4 / 10 without it) | failed; guard on by default, release text gives the measured effect, not a pass | [E2.24] |
| Inkling loop guard: prompt-cache reuse >= 0.9 after a firing | 0.656 and 0.735 | failed; recorded | [E2.24] |
| Inkling release engine on Ampere (RTX 3080 Ti): KL within 2 % of the RTX 4070 value | 0.003092 vs 0.003341 (7.5 % lower); without one near-tie token on the 4070, -0.6 % | failed as written; the gap was traced to summation order (one token); top-1, recall and memory lines passed | [E4.13] |
| Inkling two-slot memory: shared (host) memory <= start + 64 MiB | 394 MiB (the tiered cache's pinned tail) | failed; two slots opt-in, the default stays one slot | [E2.25] |
| KL-only selection costs agent work (fresh seeds, p < 0.05) | 296 vs 292, p = 0.68 | not met; claim removed | [E1.31] |
| Selection effect: interval of the selection effect above 0 | +13.3 [+0.0, +27.0] | not met; stated as a procedure | [E3.15] |
| Per-class K/V matters for recall (pass / fail) | every arm full | not met; repair shown on KL and memory only | [E2.22] |
| Per-class K/V matters for recall (graded) | intervals include 0 | not met; 4-bit global cache inconclusive | [E2.23] |
| Agent-trace probe sees a known loser | +0.0021 [-0.0030, +0.0072] | probe dropped | [E1.14] |
| Long-chat named check (KL-tools) | docs 93 / 529: 1.398 / 0.582 vs 0.873 / 0.422 | flagged; map not shipped | [E1.9] |
| Task-guided map search: best candidate on doc 529 | 0.728 vs line 0.422 | failed; study paused | [E1.13] |

## 7. What we did not do (honest scope)

We did not invent the quantizer, the trellis codec, or the 27B bit map, which is L0xRE-V2 by YourHighnessLA. The
contribution is the diagnose-repair-verify discipline, and evidence that it changes what ships and what is claimed:
on the 27B model it rejected five lower-KL maps, one runtime repair and one public claim, and it removed two claims
of this paper that did not hold on fresh seeds of the same split (one held on a harder split); on Inkling 2 Mini it chose
the map and the K/V setting, kept six other changes out of the release (Section 5), held one public claim on fresh
seeds, and its harness gate
caught an engine fault that the KL gate passed. The engine fault came from our own porting, and the fix came from
upstream work by others.

## 8. Related work

Mixed-precision allocation by sensitivity [HAWQ; HAWQ-V2; HAWQ-V3; SqueezeLLM], by learned or searched policies
[HAQ; EvoPress] and outlier-aware PTQ [SpQR; GPTQ; AWQ] choose bits by a proxy, a hardware signal, a calibration
fitness or a reconstruction error. Trellis and lattice codes [QuIP#; QTIP; EXL3] give the codec we use. K/V-cache
quantization [KIVI; KVQuant] proposes codecs, and KVTuner [KVTuner] searches a key / value precision per layer; we
choose existing types per cache class. MoE-specific PTQ balances calibration across experts [MoEQuant] or calibrates
the router against expert shift [EAC-MoE]; we did not change the quantizer for the MoE. Long-context evaluation moved
from single needles [NIAH] to broader suites [RULER], and long-context tasks lose more at 4 bits than at 8 [Mekala et
al.]; our probes are needle-type with continuous margins. Closest in spirit are evaluations showing that averages hide
changes in compressed models [Dutta et al.; Jaiswal et al.; Marchisio et al.; Liu et al. 2025]. Dutta et al.
recommend KL divergence (with flips) as the evaluation metric; we disagree for selection at about 2.5 bits, because
maps with lower KL lost recall at depth in our gate set (Table 1), and we keep KL as one gate among several. We put
task probes inside the allocation loop and add a held-out confirmation [Cawley & Talbot; Recht et al.; Dwork et al.].
Our lines written before each run are an informal, private form of preregistration [Nosek et al.; Prereg-ML]. Full
list with one-line differences: `evidence/RELATED.md`.

## Data and licences

- Qwen3.8-27B weights, tokenizer and template: Apache-2.0. Our release file is derived from them (quantized, rebuilt
  as GGUF) and is Apache-2.0.
- Public Hessians, golden set and KL scorer of the Qwen3.8-27B quantization challenge: CC BY 4.0; the corpora behind
  them keep their own licences. Our scorer is a port of the challenge's scorer.
- exllamav3 (quantizer, EXL3 codebook, GEMV kernels): MIT. escha-mlx (codec tables, reference): Apache-2.0.
  llama.cpp / ggml and the engine fork chain, including llama.cpp-escha: MIT.
- The 27B bit allocation is taken unchanged from L0xRE-V2. Its model card shows the licence
  "qwen-research-terms-derivative" with no licence text. We use the list of bit widths with credit; readers who reuse
  it should check that card.
- Inkling 2 Mini official weights: Apache-2.0. LookAtMyQuant: code MIT, Inkling files Apache-2.0. Unsloth's Inkling
  GGUFs (UD-Q2_K_XL, UD-Q8_K_XL) and Unsloth's llama.cpp build were used for evaluation only.
- Benchmarks: AppWorld, HumanEval, WikiText-2 (test split), used for evaluation only.

## AI use

Large language model agents, run as coding agents, ran most experiments, wrote most of the code and drafted most of
this text, under the author's direction. The pass lines were written before each run, by the author or by an agent
under his direction, and kept in a timestamped private record. The author checked the results against the logs and
decided what ships and what is claimed. Every number in this paper is read from a log or a scorer output listed in
`evidence/EVIDENCE.md`.

## Acknowledgements

This work stands on the work of many others (from the public CREDITS of the release): the Qwen team (Qwen3.8-27B);
turboderp (exllamav3, the EXL3 codec, kernels); Albert Tseng, Qingyao Sun, David Hou and Christopher De Sa (QTIP);
Escha Labs (the Escha W2 format and reference); YourHighnessLA (the L0xRE-V2 bit allocation shipped unchanged, and the
L0xRE runtime); StillDeadcode (the public Hessians, golden set and KL scorer, CC BY 4.0) and the publishers of the
corpora behind them; the ggml / llama.cpp contributors; Ajay9o9, spiritbuun, PrismML, sudoingX (batch-invariant
mode), Chirag Patnaik, alesha-pro, Mirai Labs, usmaneth, zhaoyilun, cklxx, Empressia, lenny76, abdulaziz, Scott
Guyton, Strike3D, Ankit Khandelwal, Mark Schatza, Jason Titus, Tomer Weller (engine work); syv-ai / HyperQwen, z-lab
(DFlash), Inco AI and ggml-org (drafting), Magnitude (mixed K/V idea); Thinking Machines (Inkling 2 Mini, Apache-2.0),
Unsloth (the Inkling build and GGUFs we compared against, and the UD-Q8_K_XL file we used to run the official
weights) and ShinMegamiBoson (LookAtMyQuant); Piotr Wilkin (llama.cpp Inkling 2 support and the DSpark drafter
code we ported); Aldehir Rojas (the llama.cpp Inkling chat parser and the grammar-trigger guard whose absence in our
port caused the harness failures of Section 4.1, and the lazy-grammar reasoning gating); Alexey Fateev (alesha-pro)
and Peasant Smith and the Project Maya contributors (the per-token launch-overhead measurement on Inkling 2 Mini and
the idea to fuse its glue launches); Killy (the voxel-pagoda prompt behind the bad-token check); the authors of HumanEval and
AppWorld.
No person or company named here endorses this work.

## References

[AppWorld] H. Trivedi, T. Khot, et al. AppWorld: A Controllable World of Apps and People for Benchmarking Interactive
Coding Agents. ACL 2024. arXiv:2407.18901.

[AWQ] J. Lin et al. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. MLSys 2024.
arXiv:2306.00978.

[Cawley & Talbot] G. C. Cawley, N. L. C. Talbot. On Over-fitting in Model Selection and Subsequent Selection Bias in
Performance Evaluation. JMLR 11:2079-2107, 2010.

[Dutta et al.] A. Dutta, S. Krishnan, N. Kwatra, R. Ramjee. Accuracy is Not All You Need. NeurIPS 2024.
arXiv:2407.09141.

[Dwork et al.] C. Dwork, V. Feldman, M. Hardt, T. Pitassi, O. Reingold, A. Roth. The reusable holdout: Preserving
validity in adaptive data analysis. Science, 2015.

[EAC-MoE] Y. Chen, Y. Shao, P. Wang, J. Cheng. EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts
Large Language Models. ACL 2025. arXiv:2508.01625.

[EvoPress] O. Sieberling, D. Kuznedelev, E. Kurtic, D. Alistarh. EvoPress: Accurate Dynamic Model Compression via
Evolutionary Search. ICML 2025. arXiv:2410.14649.

[EXL3] turboderp. ExLlamaV3. Software, <https://github.com/turboderp-org/exllamav3>.

[GDN] S. Yang, J. Kautz, A. Hatamizadeh. Gated Delta Networks: Improving Mamba2 with Delta Rule. ICLR 2025.
arXiv:2412.06464.

[GPTQ] E. Frantar, S. Ashkboos, T. Hoefler, D. Alistarh. GPTQ (OPTQ). ICLR 2023. arXiv:2210.17323.

[HAQ] K. Wang, Z. Liu, Y. Lin, J. Lin, S. Han. HAQ: Hardware-Aware Automated Quantization with Mixed Precision.
CVPR 2019. arXiv:1811.08886.

[HAWQ] Z. Dong, Z. Yao, A. Gholami, M. W. Mahoney, K. Keutzer. HAWQ. ICCV 2019. arXiv:1905.03696.

[HAWQ-V2] Z. Dong et al. HAWQ-V2. NeurIPS 2020. arXiv:1911.03852.

[HAWQ-V3] Z. Yao et al. HAWQ-V3: Dyadic Neural Network Quantization. ICML 2021. arXiv:2011.10680.

[HumanEval] M. Chen et al. Evaluating Large Language Models Trained on Code. arXiv:2107.03374, 2021.

[Jaiswal et al.] A. Jaiswal, Z. Gan, X. Du, B. Zhang, Z. Wang, Y. Yang. Compressing LLMs: The Truth is Rarely Pure
and Never Simple. ICLR 2024. arXiv:2310.01382.

[KIVI] Z. Liu et al. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. ICML 2024. arXiv:2402.02750.

[Koehn] P. Koehn. Statistical Significance Tests for Machine Translation Evaluation. EMNLP 2004.

[KVQuant] C. Hooper et al. KVQuant. NeurIPS 2024. arXiv:2401.18079.

[KVTuner] X. Li, Z. Xing, Y. Li, L. Qu, H.-L. Zhen, W. Liu, Y. Yao, S. J. Pan, M. Yuan. KVTuner: Sensitivity-Aware
Layer-Wise Mixed-Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference. ICML 2025.
arXiv:2502.04420.

[Liu et al. 2025] R. Liu et al. Quantization Hurts Reasoning? COLM 2025. arXiv:2504.04823.

[LookAtMyQuant] ShinMegamiBoson. LookAtMyQuant. Software, <https://github.com/ShinMegamiBoson/lookatmyquant>.

[Marchisio et al.] K. Marchisio et al. How Does Quantization Affect Multilingual LLMs? Findings of EMNLP 2024.
arXiv:2407.03211.

[Mekala et al.] A. Mekala, A. Atmakuru, Y. Song, M. Karpinska, M. Iyyer. Does quantization affect models' performance
on long-context tasks? EMNLP 2025. arXiv:2505.20276.

[MoEQuant] X. Hu, Z. Chen, D. Yang, et al. MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language
Models via Expert-Balanced Sampling and Affinity Guidance. ICML 2025. arXiv:2505.03804.

[NIAH] G. Kamradt. Needle In A Haystack. Software, 2023.

[Nosek et al.] B. A. Nosek, C. R. Ebersole, A. C. DeHaven, D. T. Mellor. The preregistration revolution. PNAS
115(11):2600-2606, 2018.

[Prereg-ML] S. Albanie, J. F. Henriques, L. Bertinetto, A. Hernandez-Garcia, H. Doughty, G. Varol (eds.). NeurIPS
2021 Workshop on Pre-registration in Machine Learning. PMLR 181, 2022.

[QTIP] A. Tseng, Q. Sun, D. Hou, C. De Sa. QTIP: Quantization with Trellises and Incoherence Processing. NeurIPS 2024.
arXiv:2406.11235.

[QuIP#] A. Tseng, J. Chee, Q. Sun, V. Kuleshov, C. De Sa. QuIP#. ICML 2024. arXiv:2402.04396.

[Recht et al.] B. Recht, R. Roelofs, L. Schmidt, V. Shankar. Do ImageNet Classifiers Generalize to ImageNet? ICML 2019.

[RULER] C.-P. Hsieh et al. RULER. COLM 2024. arXiv:2404.06654.

[SpQR] T. Dettmers et al. SpQR. ICLR 2024. arXiv:2306.03078.

[SqueezeLLM] S. Kim et al. SqueezeLLM: Dense-and-Sparse Quantization. ICML 2024. arXiv:2306.07629.

[Thaler] R. H. Thaler. Anomalies: The Winner's Curse. Journal of Economic Perspectives 2(1):191-202, 1988.

[Unsloth] Unsloth AI. Unsloth Dynamic 2.0 GGUFs. Blog, 2025.

[L0xRE-V2] YourHighnessLA. L0xRE-V2-27b-Low. Hugging Face, 2026. <https://huggingface.co/YourHighnessLA/L0xRE-V2-27b-Low>

[Escha W2] Escha Labs. escha-mlx (Escha W2 format and reference). Software, <https://github.com/EschaLabs/escha-mlx>

[llama.cpp-escha] Ajay9o9. llama.cpp-escha (the Escha op for ggml). Software, <https://github.com/Ajay9o9/llama.cpp-escha>

## Appendix A. Names in the text and in the records

| Name in this paper | Name in the records / `evidence/EVIDENCE.md` | What it is |
|---|---|---|
| shipped file, start map | V3ac-h32 (= V3a with a per-32 Q8_0 head), Kintsugi-27B-12G.gguf | the L0xRE-V2 allocation with our fits |
| KL-group | V3s2 | lower-KL map from a measured KL table per group |
| KL-table A, KL-table B | V5a, V5c | lower-KL maps from the KL table |
| KL-only winner | V5bc-h32 | the lower-KL map that passed every non-agent gate |
| KL-tools | V5tc-h32 (V5t) | KL-table map aimed at the tools class |
| uniform baseline, HAWQ-style baseline | U1c-h32, H1c-h32 | the G-B baselines |
| author's fits | L0xRE-V2 body | the original author's fits of the shipped layout |
| proxy map | V3b | Hessian-weighted fit-error map |
| five task-guided maps | C1-C5 | task-damage map candidates |
| expert-layer map, M2 | M1, M2 | the two Inkling maps named before any result |
| M2 with a 4-bit head, copy-aware map, borrowed allocation | M2h, M3, M5 | rejected Inkling repairs |
| sparse attention past 200k | S1v2 | rejected 27B runtime repair |
| KL gate, recall gate, harness gate, bad-token check | G1, G2 / G3, G6, GATE P | release gates |
| graded margin probe | P1 (Qwen), ink_margin gf (Inkling) | teacher-forced recall margins |
| fresh-seed tests (agent ablation, selection effect, K/V ablation, graded K/V, graded baselines) | G-A1, G-A2, G-C, G-C2, G-B2 | test names in `evidence/LINES.md` |
