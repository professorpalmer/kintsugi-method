# Kintsugi: Task-Gated Repair Reaches Full-Precision Agent Scores at 2.5 Bits

Cary Palmer

*Version 1.3 (2026-11-27). Every number cites a row of `evidence/EVIDENCE.md` as [E#.#]; `README.md` says how to recompute each one. Appendix B lists what each version changed.*

---

## Abstract

Low-bit quantization of language models is usually tuned on an average error such as KL divergence. The Kintsugi
method gates every change and every repair on task probes next to KL, with pass lines written before each run. On two
model families, the 12 GB builds that came out of this loop, at about 2.5 bits per coded weight, reach the AppWorld
scores of the full-precision weights: on Qwen3.8-27B the shipped build (the published bit allocation with our fits,
head and runtime) and the map that the loop repaired; on Inkling 2 Mini our map (596 against 592 of 672). On the 27B
model, average KL would have picked lower-scoring maps. Such results sit at single decisions that a benchmark total
hides: decision-point habits. We detect one by a turn-type split, replay the deciding call on a frozen history, and
locate it by tensor copies between files.
(1) Qwen3.8-27B, BFCL: when a held-back tool returns, the model often says the tool is absent although the new tool
list is in its prompt. At temperature 0.001 in one setup, BF16 misses 61 of 200 such turns and the 2.47-bit
build 36 (p = 1.1e-4). Stripping earlier thinking removes 40-62 % of the build's misses. Tensor copies pointed at the
fits of two same-map files on the public entries; on the development half of a fresh private set that gap was not
shown, and a graft chosen there was not confirmed on the sealed half.
(2) Inkling 2 Mini, a pre-registered confirm on fresh tasks: the first reply that the client receives has no code in 46 % of
calls for the official weights, 45 % for a public 2-bit file and 11 % for our map; no task cost is shown.
No build was made to change a habit. A map built to remove one, confirmed on held-out turns, is the open step. We did
not invent the quantizer, the trellis codec or the 27B bit allocation.

## 1. Introduction

Post-training quantization (PTQ) makes a 27B model fit a 12 GB consumer GPU at about 2.5 bits per weight. Two
questions decide the result: which parts of the model keep more bits, and which runtime settings (K/V-cache type,
head precision, drafting) are safe. Most allocation methods answer the first question with a proxy (Hessian
curvature [HAWQ], activation salience [AWQ], reconstruction error [GPTQ]) or with an average output error (perplexity,
KL divergence [Unsloth; LookAtMyQuant]). Task accuracy is then measured once, at the end.

This paper reports what happened when we gated every bit map and every runtime setting on tasks that users of a
12 GB build run: recall of an exact fact or a code function 117k-245k tokens back, long chat answers, and a
multi-step agent benchmark. Version 1.2 added the mechanism of the agent result: a reading of every failed run of
the second batch, and four tests that followed from it (Section 4.3). Version 1.3 makes that reading a procedure and
applies it to a second decision of the 27B model and to a second model family (Section 4.10).

**The central result of this version: decision-point habits** (Section 4.10). A benchmark total can hide a choice
that a model makes at one kind of turn. We give a procedure with four steps (detect, freeze and replay, locate,
repair) and we do the first three.

- *Detect.* A candidate map has level totals on two agent benchmarks. A split by turn type shows one turn where it
  differs. When a held-back tool comes back (BFCL), it gives no call in 48 of 198 entries, against 28 of 199 for the
  shipped file [E7.1].
- *The mechanism.* In most of these misses the model says that the tool is not there, while the updated tool list is
  in its prompt. It keeps its earlier belief over the tool block [E7.3].
- *Replay.* One frozen history and one call per entry repeat the gap in about 35 minutes for each file. On the frozen
  turn we tested the wording, the sampler, the thinking of earlier turns (stripping it removes 40 % to 62 % of the
  shipped file's misses) and the parent. In one setup at temperature 0.001, BF16 misses 61 of 200 return turns and
  the shipped file 36 [E7.4-E7.7].
- *Locate.* On the public entries, tensor copies between two files with one bit map suggest that the fits carry
  their gap, and that the head and the other tensors do not. The result does not hold under every miss rule. On a
  fresh private set the gap between the two files is not shown on the development half (same direction, smaller),
  and one of the four leads shows again. A graft that we chose on that half was not confirmed on the sealed half. We
  call the result suggestive only [E7.9, E7.15, E7.16].
- *A second family.* On Inkling 2 Mini, a pre-registered confirm on fresh tasks passed its three lines. At the first
  call of a task, the first reply that the client receives has no code in 46 % of the calls for the official weights, 45 % for a public 2-bit
  file and 11 % for our map. This decision has no measured task cost: it marks a departure from the parent that the
  task total does not show [E7.11, E7.13].
- *The open step.* Neither build was made to change a habit. A map that is built to remove a habit, and then
  confirmed on held-out turns, is the step that this paper does not have.

**What the title says.** On both families, the builds that came out of the task-gated loop reach the AppWorld
scores of the full-precision weights (Section 2 defines the term). On Qwen3.8-27B these are the shipped build and the
map that the loop repaired. The shipped build is the published bit allocation with our fits, an exact head and our
runtime settings, each change gated on tasks. It is level with BF16 on the standard split, and above BF16 at
temperature 1.0 on the harder split [E3.7, E3.16, E5.4]. The repaired map (KL-tools) was found by class-level KL and
gated on tasks. It is level with the shipped build on four seeds of the harder split, and above BF16 on the same
seeds [E6.6]; it did not ship (Section 4.4). On Inkling 2 Mini the build is M2: the loop chose its map and repaired
its K/V setting by cache class, and it scores 596 against 592 of 672 for the official weights [E4.8]. So "repair" in
the title is the loop (diagnose, locate, repair, gate), not the 27B bit allocation, which is the published one
(Scope, below). Both builds have about 2.5 bits per coded weight: 2.47 and 2.44 [E4.1, E2.5].

**The other main results.**

1. **At temperature 1.0, a 2.47-bit build scores above BF16 on the harder agent split, and at that temperature the
   lead is in the weights** (Section 4.2). The shipped 27B file leads BF16 on AppWorld test_challenge on each of
   four seeds (+32, +24, +35, +26 of 417; seed-level p = 5.1e-4) [E5.4]. The same file in the exact setup of the BF16
   run scores 670 against 614 [E5.5]. At temperature 0 (greedy, one run of 417 tasks for each arm) the lead is +11
   [-5, +28] (p = 0.24), and one greedy run cannot show whether it is smaller than the lead at 1.0 [E5.18, E6.1]. On
   the standard split the two were level (Section 4.6).
2. **The lead is one end-of-task habit of the base model** (Section 4.3). On test_challenge, 369 of 417 tasks need an
   action and no answer, and the prompt says so in every call. A run that passes an answer anyway fails the
   evaluator's answer test. Every arm does this, BF16 included; BF16 does it in 258 of 1,668 task-runs, the shipped
   file in 112. Wrong or missing actions are level over all eight arms. With the unasked-answer runs counted as
   passes (a class added after the first reading showed it), the shipped file and BF16 are level, and the bit maps
   of Section 4.4 are within 26 task-runs of each other [E6.4]. Reasoning effort xhigh lowers the habit (3 against 13
   runs on a targeted set, p = 0.013); one targeted sentence at the default effort does not (10 against 13,
   p = 0.65) [E6.9, E6.10].
3. **Average KL does not rank bit maps** (Section 4.4). At the same size and in the same codec, six maps score from
   586 to 680 of 834 task-runs; the KL order and the task order do not agree [E5.2, E5.13].
4. **A class-level KL table found a repair in context, and the raised band carries it** (Section 4.4). The repaired
   map scores 70 task-runs above the KL-only winner; its 8 raised projections alone give +53 (seed-level p = 0.021),
   but the same band costs nothing when it alone goes to 2 bits in the shipped map [E5.2, E5.12, E6.5].
5. **Most of our mechanism hypotheses for the map losses failed, and we report each one** (Sections 4.4, 4.5, 4.7).
6. **Lines written before the runs held two lower-KL maps out of the release** (Sections 4.4 and 6): KL-tools, on a
   free-generation line, and a later candidate, on a rule for both agent benchmarks [E5.26, E6.14].
7. **On Inkling 2 Mini, a model of another lineage, the headline is a task result** (Section 4.7): with the setup
   equal, our build scores above a public 2-bit file (598 against 579 of 672, seed-level p = 0.019) and is not
   worse than the official weights [E5.22, E5.23]. The KL gap to that file belongs to the codec, the fits and the
   engine format, not to our per-group map [E5.20].

**Two findings about the instruments.**

- **Seed clustering** (Section 4.1). The AppWorld client sends the same seed on every model call, and the server makes
  a new sampler from that seed for each request. All task-runs of one seed thus share one random stream. Every
  AppWorld claim here has a design-effect p, a cluster-robust interval and the per-seed signs [E5.1, E5.28].
- **Teacher-forced instruments are blind to free generation** (Section 6). KL, recall margins and the held-out
  recall test score forced tokens. A free-generation check found a change in thinking length that none of them
  showed. A line on output length then misread it: a forced reading under BF16 shows a departure that both quantized
  files share, not damage in the new map [E5.26, E5.31].

**Contributions.**

- A procedure for decision-point habits (detect by turn type, freeze and replay, locate by tensor copies) on two
  model families, with a blind recount of every cell count, and a plain statement of the step that is missing
  (Section 4.10).
- A reproducible loop (Section 3; full version in the supplement `evidence/ALGORITHM.md`): fit once, diagnose with KL and
  task probes, locate with one-group flips, repair under a byte budget, gate with pre-written lines, confirm on
  fresh data, and keep failures in the record.
- A measured case in which a 2.47-bit build scores above its BF16 reference on a hard agent split at temperature
  1.0, with controls that put the lead in the weights at that temperature, a temperature-0 control, and the
  mechanism: one habit of the base model whose rate the quantized files change (Sections 4.2, 4.3).
- Evidence on a dense hybrid model that average KL does not rank bit maps by task quality at 2-3 bits, on agent
  work and on held-out long-range recall, and a list of the mechanism hypotheses that failed (Section 4.4).
- A repair in context: class-level KL pointed at one layer band, the repair was fixed before its results, and the
  repaired map (KL-tools) scores 70 task-runs above the KL-only winner (seed-level p = 0.0017) [E5.2]. A split test
  puts the gain in the raised band [E6.5]. The repaired map is level with the published allocation on agent work at
  7.9 % lower KL [E5.11], also on four seeds [E6.6]. It failed a free-generation line written before the run, so it
  did not replace the published allocation; a forced reading under BF16 then showed that the line misread a shared
  departure (Section 6) [E5.26, E5.30, E5.31].
- A seed-level statistics method for agent benchmarks that send one seed for each run, the finding that made it
  necessary, and a proposed fix for the client (Section 4.1).
- Readings across compressions of one base model: what carries from one map to the next of the same base, and what
  did not carry to a second lineage (Section 4.9).
- On a second architecture, a task result with the setup held equal, and a split of a KL gap into the part from the
  codec and the part from the bit map (Section 4.7).

**Scope.** We did not invent the quantizer (exllamav3), the trellis codec (QTIP / EXL3), or the 27B bit map, which
is the L0xRE-V2 allocation by YourHighnessLA [L0xRE-V2], shipped unchanged. On the 27B model the method found and
repaired damage in context, but the shipped build keeps the published allocation. On Inkling 2 Mini our rule chose
the shipped map and the runtime settings; a uniform map at the same bytes is level with it on KL and on agent work
and has less long-range copy margin (Section 4.7).

**Availability.** Kintsugi-27B: [GitHub](https://github.com/professorpalmer/Kintsugi-27B) and
[Hugging Face](https://huggingface.co/CaryPalmer/Kintsugi-27B). Kintsugi for Inkling 2 Mini:
[release v1.0](https://github.com/professorpalmer/Kintsugi-Inkling-2-Mini/releases/tag/v1.0) and
[Hugging Face](https://huggingface.co/CaryPalmer/Kintsugi-Inkling-2-Mini). Evidence, data and the scripts that
recompute every statistic and figure: the [evidence repository](https://github.com/professorpalmer/kintsugi-method)
(DOI [10.5281/zenodo.24455575](https://doi.org/10.5281/zenodo.24455575)).

## 2. Setting

**Models.** (a) Qwen3.8-27B: 64 layers, 16 full-attention and 48 Gated DeltaNet (linear-attention) layers [GDN]; 400
coded projections, each at 2 or 3 bits per weight (K2 / K3) [E4.1]. (b) Inkling 2 Mini: 24 layers, 128 routed experts
(top-8) per layer, layers 6-17 run three times with shared weights (48 layer applications, each with its own K/V
cache), 4 global-attention and 20 sliding-window layers; the official release is NVFP4 / MXFP8, with no BF16 release
[E4.2]. Its reference is the official weights themselves (NVFP4 / MXFP8), run as Unsloth's UD-Q8_K_XL file,
whose every tensor equals our exact dequantization of the official release (max abs diff 0) [E4.10]. In this
paper *full precision* means the weights as their authors released them: BF16 for Qwen3.8-27B, and for Inkling 2 Mini
the official NVFP4 / MXFP8 release, which has no higher-precision form. The text calls them "the official weights".

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
- *Free-generation check.* The 24 needle questions at 245,000 tokens with the launcher's default thinking, temperature
  0 and at most 4,096 new tokens. For each item we record the answer, how the reply ended and the length of the
  thinking, and we compare the lengths with those of the reference file on the same items. We do not use a fixed
  length limit as a pass line (Section 6) [E5.26].
- *Agent.* AppWorld [AppWorld], temperature 1.0. Two splits: test_normal (168 tasks) and the harder test_challenge
  (417 tasks). One *task-run* is one task on one seed. The client sends only the temperature and the seed, so the
  server's sampler defaults apply in every arm: top-k 20, top-p 0.95, min-p 0.05 [E5.6]. Seeds: 100 + 101 chose the
  Qwen maps ("selection seeds"); 102-105 (test_normal) and 106-109 (test_challenge) were never used for a choice
  ("fresh"). Seeds 119-124 (test_challenge) belong to the effort tests of Section 4.3; none chose a map. Inkling:
  102-105 and 108-111 (test_normal), 112 + 113 (test_challenge); no Inkling choice used AppWorld.
- *Return-turn replay.* BFCL v3 multi-turn [BFCL], category miss_func (200 entries), temperature 0.001. The unit is
  the first model call of the turn in which a held-back function comes back, sent again to each file on a recorded
  history. Two miss rules; exact sign tests on paired units (Section 4.10).
- *First-call replay.* The first call of an AppWorld task, built by the benchmark's own agent code: one call for each
  task and seed, with an output cap of 1,024 tokens. The score is whether the reply has a code block (Section 4.10).
- *Agent statistics.* For each comparison we give the difference in task-runs, the difference on each seed, a
  design-effect p and a cluster-robust 95 % interval (Section 4.1). "Seed-level p" means the design-effect p. Where we
  quote a task-level p (the exact sign test on discordant task-runs), we say so; it is too small by an unknown
  factor.
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
   documents) *and* task probes (recall margins at depth, the recall gate, free generation, the agent benchmark).
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

Four rules came from failures and are now part of the method. *Probe validation:* a probe may rank groups only if it
ranks a known loser below the reference; our teacher-forced agent-trace probe did not (+0.0021 nats per token,
interval [-0.0030, +0.0072]) and was dropped [E1.14]. *Hard items:* a recall probe needs items that are hard for the
reference build; a probe whose items all had margins above 2.4 nats passed a map that lost a needle [E1.7]. *Model or
engine:* when a task gate fails on behaviour that could come from the model, write the rule that separates the two
before the check (the same task repeated on our build and on a reference build, and the model's own token
probabilities at the failing step), and keep the failed gate in the record whatever the check says [E1.25]. *No map
tuned to one item:* when a single gate item fails, do not tune a map until that item passes; a map that passes it can
move the failure to other items (Section 6) [E5.26].

Version 1.3 adds a procedure for single decisions next to this loop: detect a turn type where two files differ,
freeze one history and replay the deciding call, and locate the difference by tensor copies (Section 4.10).

## 4. Experiments

### 4.1 The task-runs of one seed are not independent

An exact sign test on the discordant task-runs assumes that each task-run is an independent draw. On AppWorld it is
not.

**The finding.** The AppWorld client sends the run's seed on every model call. The server makes a new sampler for each
request and seeds it with that value. So every call of every task of every arm on one seed starts from the same
random stream [E5.1]. We found it while we compared the transcripts of passing and failing runs of the same task.
The data show it: the first reply starts with the same 8 words on 51.2 % of the tasks for two *different* maps on
the same seed, on 23.8 % for two maps on different seeds, and on 27.9 % for *one* map on two seeds. The seed decides
the opening more than the weights do. One map (KL-tools) opens its first reply with the word "We" on 2.4 % of the
tasks of seed 106 and on 54.2 % of the tasks of seed 107 [E5.1].

**The size.** We wrote the measures and their lines before we computed them. For one arm on two seeds of the same
tasks, the difference of the two seed totals is larger than independent draws permit: the mean squared z-value over
13 arm-seed pairs of the 27B model is 2.24 (1 is the value for independent draws; chi-square p = 0.0064). For a
paired difference between two arms the same statistic, the *design effect*, is 2.75 (mean over 55 arm pairs on
seeds 106 and 107; median 1.32, range 0-18.6). For the Inkling arms it is 1.29, and their single-arm statistic does
not show clustering (1.02, p = 0.43) [E5.1].

**The statistics of this paper.** For a comparison with per-seed differences $D_s$: (i) the differences and
their signs; (ii) the design-effect test $z = \sum_s D_s / \sqrt{\phi V}$, where $V$ is the variance of
$\sum_s D_s$ from a bootstrap that resamples tasks inside each seed and $\phi$ is the design effect (2.75 for the 27B
model, 1.29 for Inkling); (iii) the cluster-robust 95 % interval $\sum_s D_s \pm 1.96 \sqrt{\phi V}$. A claim of a
difference "keeps p < 0.05 at the seed level" only if every seed has the sign of the claim and the design-effect p is
below 0.05 [E5.1]. For the standard split of versions 1.0.x we applied the same method. On that split the measured
design effect of the 27B arms is small (0.19 over 33 arm pairs), and we used the larger value 2.75, which is
conservative there [E5.28].

Versions 1.0.x used the sign test over task-runs. Under this method the large effects keep their verdicts and
several smaller ones do not; Appendix B lists both groups [E5.1, E5.28].

**Limits of the method.** The design effect is a mean over arm pairs with a wide range, so the seed-level p is
approximate. Most comparisons have two seeds; a strict test with the seed as the only unit (a t-test of $D_s$) has
almost no power there, and we give it only where it helps. Two comparisons have four seeds: shipped against BF16
(t-test p = 0.0014) and the Inkling comparison of Section 4.7 (p = 0.059) [E5.1].

**A fix for later runs.** The client can send a seed derived from the run's seed, the task and the call number. The
draws are then independent across tasks and calls, they stay reproducible, and two arms keep the same draw on the
same task and call. We give the change as a patch proposal in the evidence repository (`evidence/SEED-FIX.md`); no run of this
paper used it [E5.1].

### 4.2 At temperature 1.0, a 2.47-bit build scores above BF16 on the harder split

On the standard AppWorld split the shipped 27B file and BF16 are level on fresh seeds (Section 4.6). At the
temperature of every AppWorld run in this paper (1.0), the harder split separates them, in the direction that we did
not expect. At temperature 0 it does not separate them significantly (below).

*Table 1. AppWorld test_challenge, 417 tasks per seed: the shipped 2.47-bit file against BF16. "12 GB setup" = the
release launcher on an RTX 4070 (4-bit tiered K/V cache, two-stage head, drafting). "BF16's setup" = an 80 GB GPU,
16-bit K/V cache, full head, no drafting [E5.2, E5.4, E5.5, E5.18].*

| Seed | Shipped file, 12 GB setup | BF16 | Difference | Only shipped passes / only BF16 passes | Shipped file in BF16's setup |
|---|---:|---:|---:|---:|---:|
| 106 | 335 | 303 | +32 | 68 / 36 | 322 |
| 107 | 335 | 311 | +24 | 58 / 34 | 348 |
| 108 | 331 | 296 | +35 | 70 / 35 | not run |
| 109 | 329 | 303 | +26 | 69 / 43 | not run |
| All four | 1,330 | 1,213 | +117 of 1,668 | 265 / 148 | 670 (BF16: 614) on 106 + 107 |
| Temperature 0, one run (seed 106; greedy) | 330 | 319 | +11 (p = 0.24) | 41 / 30 | not run |

**The result.** Seeds 106 + 107 were the first run, with a non-inferiority line written before it (the shipped file is
"not worse" only if the interval of the difference stays above -25 of 834). The result was 670 against 614. Because
this was a surprise, we wrote a new line and ran both arms again on two fresh seeds. The line was: "the 12 GB build
scores higher than BF16 on the harder split" only if it is higher with p < 0.05. Seeds 108 + 109 gave 660 against 599
[E5.4]. Over the four seeds the lead is +117 of 1,668 task-runs, with the same sign and a similar size on each seed.
The seed-level p is 5.1e-4, the cluster-robust 95 % interval is [+51, +183], and the t-test over the four seed
differences gives p = 0.0014 [E5.4].

**At temperature 1.0 it is in the weights.** The two arms of Table 1 differ in the weights and in the whole setup. So we ran the shipped
file in the exact setup of the BF16 arm, with the line written first: "the lead comes from the weights" only if this
arm is above BF16 with p < 0.05 and is not below the 12 GB run. It scored 670 against BF16's 614 on seeds 106 + 107
(+19 and +37; seed-level p = 0.014; interval [+11, +101]) and 670 against 670 for the 12 GB run (-13 and +13;
interval [-40, +40]) [E5.5]. The line is met. The -13 and +13 for one file in two setups also show how much one
seed can move.

**What we ruled out.**

- *A setup artifact.* In 123 of the 126 task-runs of seeds 106 + 107 that only the shipped file passed, BF16 finished
  the task and gave a wrong result. It did not stop at a limit: no reply was cut, no reasoning budget was hit, no
  request failed, and the step limit ended 3 of BF16's failures and 5 of the shipped file's. The first prompt had the
  same token count in 834 of 834 task-runs [E5.3]. Over four seeds, the 50-call limit ended 3.8 % of BF16's failed
  runs and 6.1 % of the shipped file's (Fisher p = 0.33) [E5.9].
- *The two-stage head.* The 12 GB launcher computes exact logits only for 512 candidate tokens. Both setups sample
  with top-k 20, top-p 0.95 and min-p 0.05, so neither arm can draw a token outside its top 20 [E5.6]. With the
  two-stage head off, the shipped file scored 687 against 670 with it on. One shard of the head-off arm ran twice
  into one folder (19 mixed task-runs of seed 107); without them the difference is +14 (+12 and +2; seed-level
  p = 0.44): the head does not lift the score. The head-off arm leads BF16 by 73 (+44 and +29; seed-level p = 6.2e-4)
  [E5.6, E6.3].
- *Drafting.* The 12 GB setup drafts tokens; the BF16 setup does not. In the engine, every emitted token is the
  target model's own sample, and a drafted token is kept only if it equals that sample. So drafting changes the
  speed, not the output distribution. This is from a reading of the code, not from a run [E5.7]. The run in BF16's
  setup has no drafting and gives the same lead [E5.5].
- *A sharper output.* A quantized model could score higher at temperature 1.0 if its distribution were more peaked.
  It is not. On 376,086 rows of the golden set, each of four maps is slightly *flatter* than BF16 after the sampler
  truncation: entropy 0.7514-0.7570 nats against 0.7389, each interval above 0. Two maps with the same sharpness
  (KL-tools and the KL-only winner; difference -0.0002 [-0.0007, +0.0003]) score 680 and 610. The line "sharpness
  explains the agent ordering" was not met [E5.8]. The text of this test is the golden set, not AppWorld traffic.
- *Longer thinking, or longer tasks.* BF16 writes 2-8 % more reasoning for each call. It needs about one call fewer
  for each task [E5.9]. The lead is spread over task lengths, step counts, difficulty levels and apps [E5.10]. It is
  not spread over the steps of a task: Section 4.3 puts it at the last step.

![AppWorld test_challenge at temperature 1.0, 417 tasks per seed (the temperature-0 control is in Table 1). (a) Score per seed for the shipped 2.47-bit file in the 12 GB setup
(seeds 106-109), for BF16 (106-109), and for the shipped file in BF16's exact setup (106 + 107). (b) Difference to
BF16 with cluster-robust 95 % intervals: the shipped file over four seeds (of 1,668 task-runs); the shipped file in
BF16's setup and the shipped file with the two-stage head off (each of 834, seeds 106 + 107)
[E5.2, E5.4-E5.6].](figs/fig2-bf16-per-seed.png){width=100%}

**At temperature 0 the lead is not significant.** Greedy decoding has no random stream, so each task is one
independent draw. We wrote the line before the run: "the lead over BF16 holds at temperature 0" only if the shipped
file is higher with an exact sign test p < 0.05 over the 417 paired task-runs. It is not shown: 330 against 319, 41
against 30 discordant, p = 0.24, difference +11 with a 95 % paired bootstrap interval of [-5, +28] [E5.18]. The two
files move differently. Under greedy decoding BF16 scores 319, which is about 16 more than its mean of 303.2 over
the four temperature-1.0 seeds; the shipped file scores 330, about the same as its mean of 332.5. Each arm's greedy
outcome agrees with the majority of its four temperature-1.0 runs on 87-88 % of the tasks where the majority is not
tied. The temperature-0 control is one run of 417 tasks for each arm; it can show a large lead, not a small one.

**Is the lead smaller at temperature 0? Not shown.** The lead at 1.0 (mean of four seeds, +29.25 of 417) minus the
lead at 0 (+11) is +18. A bootstrap over the 139 AppWorld scenarios gives [+0.8, +36.5]. That interval treats the
task-runs of a seed as independent. With the design effect of Section 4.1 on the temperature-1.0 part it is
[-6, +43] (p = 0.14), and against each single temperature-1.0 seed every interval holds 0 [E6.1]. So we state: at
temperature 0 the lead is +11 [-5, +28], and one greedy run cannot show whether it is smaller. At temperature 1.0,
the run in BF16's setup shows that the difference is in the weights.

*Table 1b. Each arm's temperature-1.0 runs (seeds 106-109) split by the arm's own greedy result on the same task. A
description, not a test [E6.2].*

| Arm | Greedy passes | T 1.0 runs that fail | Greedy fails | T 1.0 runs that pass |
|------------|------:|--------------------:|------:|--------------------:|
| BF16 | 319 | 209 of 1,276 (16.4 %) | 98 | 146 of 392 (37.2 %) |
| Shipped file | 330 | 154 of 1,320 (11.7 %) | 87 | 164 of 348 (47.1 %) |

At temperature 1.0, BF16 loses more of the tasks that its greedy run passes, and gains fewer of those that its greedy
run fails. Section 4.3 shows which step these runs lose.

**What we do not claim.** We do not claim that quantization improves this model in general, or that the 2.47-bit
file is better than BF16 without naming the temperature. Section 4.3 gives the mechanism and its limits. The result
is from one benchmark and one sampler chain. On the standard split the two files are level. BF16 ran in our engine; in the stock engine it scored 272 against 281 in ours on the
standard split (not significant), which is some evidence that our engine does not handicap BF16, but that check was
not repeated on the harder split [E3.11].

### 4.3 The mechanism: one decision at the end of the task

**How we read it.** After the temperature-0 control we read every failed run of the second batch: the transcripts,
the API-call logs and the evaluator's list of failed tests (CPU only, spent seeds; a reading with no line) [E6.4]. The
method (the first point where two runs part, and the kind of step there) was written before the script ran. The
failure classes and the counts below were added after the first run showed the class. They are post hoc, and the
record marks them as amendments.

**The decision.** On test_challenge, 369 of the 417 tasks need an action and no answer (for example, "Start my music
player"). The agent prompt states the rule in every call: if no answer is required, omit the answer argument of
`complete_task`. A run that does the task and then passes a text as the answer (an order confirmation, a number, a
name) fails the evaluator's answer test, also when every action is right. We call such a run an *unasked answer*.

*Table 1c. AppWorld test_challenge, seeds 106 + 107, 834 task-runs per arm, temperature 1.0. "Unasked answer only" =
the answer test is the only failed test of the run. "Pass + unasked only" is not an AppWorld score: it is the score if
the answer were not tested on action tasks. Post hoc classes [E6.4].*

| Arm | Pass | Wrong or missing action | Unasked answer (all) | Unasked answer only | Pass + unasked only |
|------------------------------|------:|----------:|----------:|----------:|----------:|
| KL-tools | 680 | 90 | 39 | 30 | 710 |
| Band-only map | 671 | 85 | 62 | 48 | 719 |
| Shipped (L0xRE-V2 allocation) | 670 | 94 | 58 | 46 | 716 |
| HAWQ-style baseline | 637 | 81 | 99 | 86 | 723 |
| Uniform baseline | 630 | 86 | 105 | 87 | 717 |
| BF16 | 614 | 86 | 121 | 104 | 718 |
| KL-only winner | 610 | 102 | 101 | 87 | 697 |
| KL-table B | 586 | 90 | 144 | 125 | 711 |

**What the table shows.**

- Wrong or missing actions are level over the eight arms (81 to 102). The step limit, a full context window, loops
  and length cuts do not order the arms.
- The order of the arms is the order of one class. With it, the arms score from 586 to 680; without it, from 697 to
  723.
- Shipped against BF16: in the discordant pairs, the pairs in which the losing run's only failed test is the unasked
  answer give -56 of BF16's -56. Over four seeds BF16 passes an unasked answer in 258 of 1,668 task-runs (57 % of its
  failed runs), the shipped file in 112 (33 %); "pass + unasked only" is 1,434 against 1,420 [E6.4].
- KL-tools on four seeds: 1,353 against BF16's 1,213 (+140, every seed positive, seed-level p = 3.0e-5) and level
  with the shipped file (+23, p = 0.45). On "pass + unasked only", KL-tools and BF16 are level (-2) [E6.6].

**The habit is the base model's; the quantized files change its rate.** Every arm has the habit, BF16 included. Its
share of the task-runs goes from 4.7 % (KL-tools) and 7.0 % (shipped) to 14.5 % (BF16) and 17.3 % (KL-table B). So on
this benchmark the lead over BF16, and the order of the maps, is a difference in the rate of one habit. Fidelity to
BF16 does not give that order. The reasoning of the last call shows the habit: when it says that no answer is needed,
the run passes one in 1-8 % of the cases; when it does not say so, in 15-27 % (keyword counts) [E6.4].

**Temperature 0.** Greedy decoding lowers BF16's rate and leaves the shipped file's: for BF16, 51 unasked answers at
temperature 0 against a mean of 64.5 per temperature-1.0 seed; for the shipped file, 32 against 28.0. On an action task
where the greedy run passes no answer, a temperature-1.0 run passes one in 13 % of the cases for BF16 and in 5 % for
the shipped file [E6.4]. This is most of the temperature-0 result of Section 4.2. Three maps also ran at temperature
0, one run each. KL-tools scored 352 (shipped 330: +22, p = 0.0092; no line), the KL-only winner 322 and KL-table B
312. The deficit of KL-table B is in the same class (57 unasked answers; "pass + unasked only" 360 against the shipped
file's 353), and wrong or missing actions are level (41 to 44) in all five arms [E6.7].

**Longer thinking lowers the habit; one sentence does not.** The release launcher runs the model at reasoning effort
medium, and the model's template adds no instruction at medium. At xhigh the template adds one line at the start of
the system turn, and the model thinks 2.1 to 2.9 times longer per call (a pilot of 8 tasks) [E6.8]. By a rule written
first, we took the 70 tasks where earlier medium runs of the shipped file had given an unasked answer, and we ran both
efforts on one fresh seed on one box. A set chosen this way regresses to the mean, so only the fresh medium run counts.
The line was: "xhigh lowers the habit" only if it gives fewer unasked answers with an exact McNemar p < 0.05. It is
met: 3 against 13 (discordant 2 / 12, p = 0.013) [E6.9]. "Pass or unasked only" is 56 against 56: the passes that
xhigh gains are the unasked answers that it does not give. xhigh costs 1.7 times the minutes per task, and it ended 4
runs at a full window (medium 1) and 3 by giving up (medium 0). Then we put one targeted sentence in the slot of the
xhigh line, at medium: "When the task asks only for an action, finish it without giving an answer. Give an answer only
when the task asks a question." The result was 10 against 13 (p = 0.65): not shown, with the same thinking length as
medium [E6.10]. On the 48 tasks that need an answer, the sentence stopped no answer, but it lost 4 passes against a
fresh medium control (p = 0.125); its guard line fails as written by one task. So the xhigh effect goes with the
longer thinking, not with an instruction in that place. A forced read of the end-of-task decision under both efforts
was planned; its AppWorld data were lost to a fault in the job (its cleanup deleted the files before they were
reduced), so it has no result [E6.11]. The product release v1.2 of Kintsugi-27B keeps the same weights and offers
xhigh as an opt-in [E6.15].

**A part of the format that we keep off.** Every coded projection of the format has a bias vector, learned in the
format author's own end-to-end tune. Our clean rebuild writes it as zero on purpose (the bytes in the older files were
third-party data that our fits did not make), and the engine does not add it. On the refit file that keeps the
original bytes, a forced read with the bias on moved the end-of-task decision toward the unasked answer, by about 12
times the refit floor (100 items of our own, not AppWorld) [E6.12]. So the zero bias costs nothing on this habit.

**Limits.** The classes were added after the first reading (post hoc). "Pass + unasked only" is not a score: an
unasked answer is an error by the rules of the benchmark, and the prompt states the rule. The maps have two seeds. The
417 task-runs of a seed are one cluster (Section 4.1), so the counts have no p value. Keyword counts are matches of
fixed phrases, not a reading of each text. There is one harness, one prompt (its one worked example is a question and
ends with `answer=`) and one sampler chain. The effort test is one seed on a selected set of 70 tasks, in a later
release launcher (v1.1.6, tool layer on): it is a lead for a full arm, not an AppWorld score.

**The first case of a pattern.** The unasked answer is a habit at one decision point: one kind of turn, one choice, a
rate that the files change. We found it by a reading of failed runs. Section 4.10 makes the reading a procedure
(detect, freeze and replay, locate) and gives two more cases: a second decision of this model, and a decision of a
second model family.

### 4.4 Average KL does not rank bit maps

**Same size, same codec, 586 to 680.** Table 2 gives every 27B map that ran the harder split. Six of them have the
same size (within 0.1 %) and use the same codec and the same fit library: the shipped map, three maps that our KL
tables chose, and two baselines. They score from 586 to 680 of 834. Figure 3(b) plots the score against the KL
change.

*Table 2. AppWorld test_challenge, seeds 106 + 107, 834 task-runs per map, 12 GB setup. KL change on the build half
p1 against the shipped file. Differences are against the shipped file; "p" is the seed-level (design-effect) p;
intervals are cluster-robust 95 %. Maps marked \* do not have the size of the others [E5.2, E5.12-E5.14, E5.16,
E1.8, E1.9, E1.28, E1.29, E6.3]. The uniform row is without the 32 task-runs of one doubled shard of seed 106 (two
copies of one job ran into one folder); its totals are as scored.*

| Map | KL change | Seed 106 | Seed 107 | Total | Difference (per seed) | p | 95 % interval |
|---|---:|---:|---:|---:|---|---:|---|
| KL-tools | -7.9 % | 333 | 347 | 680 | +10 (-2, +12) | 0.64 | [-32, +52] |
| Band-only map \* | +4.1 % | 341 | 330 | 671 | +1 (+6, -5) | 0.96 | [-39, +41] |
| Shipped (L0xRE-V2 allocation) | 0 | 335 | 335 | 670 | - | - | - |
| HAWQ-style baseline | +12.7 % | 334 | 303 | 637 | -33 (-1, -32) | 0.15 | [-78, +12] |
| Uniform baseline | -3.8 % | 325 | 305 | 630 | -37 (-7, -30) | 0.094 | [-80, +6] |
| KL-table B, down band restored \* | not measured | 310 | 317 | 627 | -43 (-25, -18) | 0.045 | [-85, -1] |
| KL-only winner | -7.0 % | 304 | 306 | 610 | -60 (-31, -29) | 0.0078 | [-104, -16] |
| KL-table B | -8.8 % | 292 | 294 | 586 | -84 (-43, -41) | 2.5e-4 | [-129, -39] |

The two lowest scores belong to two of the three maps with the lowest KL. The third map with low KL (KL-tools) has
the highest score. The map with the *highest* KL of the six (the HAWQ-style baseline, +12.7 %) is not significantly
below the shipped file. So on this split KL neither finds the damaged maps nor clears the sound ones.
For reference, BF16 scores 614 on these seeds (Section 4.2). Section 4.3 shows that the order of this table is the
order of one end-of-task habit; wrong or missing actions are level over all eight arms [E6.4].

**A repair in context, found by class-level KL.** No task probe found this repair. On 2026-11-20, a KL table by
document class showed that one change of the KL-only winner cost
the tool-call documents far more than the average: the MLP down-projections of layers 16-31 at 2 bits raised KL by
16.6 % on tool-call rows against 7.6 % on all rows, and layers 20-27 carried 14.6 points of the 16.6 [E5.11]. We
fixed a repaired map before we had any of its results: the KL-only winner with the down-projections of layers 20-27
back at 3 bits. This is the KL-tools map. The two maps differ in 21 projections: the 8 of the band, and 13 that
pay for it (12 attention projections of layers 16-30 and one output projection of a linear-attention layer)
[E5.11, E5.19]. On the harder split it
scores 70 task-runs above the KL-only winner (+29 and +41; seed-level p = 0.0017; interval [+26, +114]) [E5.2]. It is
level with the shipped file on agent work on every split we ran: 680 against 670 on the harder split, 301 against
302 and 299 against 292 on the standard split [E5.2, E5.11, E5.28]. It has 7.9 % lower KL [E1.9], and it keeps its
held-out recall margin (below) [E1.34]. The long-chat flag that versions 1.0.x held against it is two documents,
not the class: on the 22 chat documents of the untouched half its median KL is lower than the shipped file's, and
one document carries the whole rise of the mean [E5.11]. On four seeds it is 1,353 against the shipped file's 1,330
of 1,668 (+23, seed-level p = 0.45): level [E6.6]. It did not replace the shipped map because it failed a
free-generation line written before the run, on one deep recall item; a later forced reading under BF16 shows that
the line misread a departure that both files share, not damage in the new map (Section 6) [E5.26, E5.31].

**Which part of the repair carries the gain.** The 21 projections have two parts: 8 raises (the band) and 13 cuts that
pay for them. We built two maps from the model files, each with one part only, and wrote the lines before any run
[E6.5]. The 8 raises alone lift the KL-only winner by 53 task-runs (+19 and +34; seed-level p = 0.021; interval
[+8, +98]) and are not separated from the full repair (-17, p = 0.41). The 13 cuts alone give +38 (+20 and +18;
p = 0.097), which is not separated from 0. By the line written first, the raises carry the repair. The two shares of
the +70 (0.76 and 0.54) add to 1.30, so "both parts help" is not excluded at this power (two seeds). With the band test
below, this measures an interaction: the band lifts the KL-only winner, and the same band costs nothing when it alone
goes to 2 bits in the shipped map (671 against 670). Its need depends on its neighbours, which is what "repaired in
context" means here.

**The band alone is not the cause.** The repair suggests a rule: "the down-projections of layers 20-27 at 2 bits cost
hard agent work". We wrote that as a prediction and tested it in three ways. All three failed or were weak.

1. *The band alone.* The shipped map with only those 8 projections at 2 bits (85 MiB smaller, nothing paid back)
   scored 671 against 670 (+6 and -5; interval [-39, +41]). The prediction was "below the shipped file with
   p < 0.05". It was not met [E5.12]. The same cut raised KL on the tool-call documents by 12.7 % and on all rows by
   4.1 %. So a rise in tool-call KL is a lead to follow, not a verdict.
2. *Two baselines that have the band at 2 bits.* The uniform map and the HAWQ-style map scored 630 and 637. Both are
   below the shipped file at the task level (p = 0.0045 and 0.022). Neither is significant at the seed level
   (p = 0.094 without the doubled shard, and 0.15), and in both the loss is in seed 107 only: -7 and -30 for the
   uniform map, -1 and -32 for the HAWQ-style map [E5.13, E6.3]. From seed 106 to seed 107 the score of each map falls
   as its count of the unasked-answer class of Section 4.3 rises (uniform -20 and +27, HAWQ-style -31 and +22) [E6.4]. We do not claim that these maps lose agent work.
3. *The band restored in KL-table B.* KL-table B with the down-projections of layers 16-31 back at 3 bits scored 627:
   41 above KL-table B (+18 and +23; seed-level p = 0.089, not significant) and 43 below the shipped file (-25 and
   -18; p = 0.045, weak). The prediction was "not below the shipped file". It was not met [E5.14].

A correction belongs here. Our record first said that the down band was the only cut of KL-table B against the
shipped map. That was wrong; it came from a comparison with the wrong base map. The model files show a second cut:
32 MLP up-projections (layers 4, 5, 10, 11, 15 and 21-47) [E5.14, E5.19]. So the map of item 3 still has that cut,
and the cause of its remaining gap is not located. No single group explains the losses. The repair of the band
holds in the map where we made it, and we do not state it as a rule.

**A candidate held by its rule.** After KL-tools failed its free-generation line, a bisect on two free-generation list
items that KL-tools failed and the shipped file passed gave a new candidate: KL-tools with 26 projections (the
linear-attention output projections of 14 late layers and the 12 attention output projections) back at the shipped
widths. We call it KL-tools-2. It is 90.6 MiB smaller than the shipped file and has 2.4 % lower KL [-3.4, -1.3]; it
passed the recall gates at 117k and 245k (24/24 + 20/20), HumanEval (158, at its line), speed and memory [E6.14]. On
AppWorld test_challenge it scored 666 against 670 (+1 and -5; interval [-45, +37]) and gave fewer unasked answers
than the shipped file (41 against 58; BF16 121). On a second agent benchmark, BFCL v3 multi-turn (800 entries), it
scored 492 against 505 (scenario p = 0.31, interval [-35, +10]). Both results pass their break-even screens. The
author had fixed one rule before any AppWorld run of it: ship the candidate only if AppWorld is level or positive,
and hold it if AppWorld is negative too, even inside its screen, because two soft results that point the same way
are a pattern. Both benchmarks point the same way, so the candidate is held and the
product keeps the published allocation [E6.14]. This is a result of the method, not a failure of it: a map with lower KL that passed every other
gate stays out of the release.

**Two more hypotheses that failed.** *Context length:* the losses of the KL-only winner and of KL-table B do not grow
with the length of the task. The direction is the opposite: the share of discordant pairs that the shipped file wins
falls from the shortest third of the tasks to the longest (0.84, 0.74, 0.61 for KL-table B) [E5.10]. *Token class:*
on tool-call documents the KL rise of the band-only map is not on function names, parameter names or argument values
(each within [-0.0005, +0.0017] nats, intervals that touch 0). It is spread over the words, the control tokens and the
punctuation around the calls. The line "the damage concentrates on one class" was not met [E5.16].

![Average error against task outcome on the 27B model. (a) Recall-gate items lost at 117k for every map with lower
KL than the shipped allocation, and for the two baselines; open circles mark the one fragile code-copy item, counted
as weak evidence. KL changes marked * are on the build subset T, the others on the full build half p1 [E1.3,
E1.6-E1.9, E1.28, E1.29]. (b) AppWorld test_challenge (seeds 106 + 107): difference to the shipped file against the
KL change, with cluster-robust 95 % intervals. Filled markers: both seeds have the sign of the difference and the
seed-level p is below 0.05. The band-only map is 85 MiB smaller than the others; the map with the down band
restored has no KL value and is not plotted [E5.2, E5.12, E5.13].](figs/fig3-kl-vs-task.png){width=100%}

**Lower KL and the other gates.** Table 3 lists every map that we built with lower KL than the shipped allocation,
and the uniform baseline, with the results of the recall gates. None shipped.

*Table 3. Bit maps with lower KL than the shipped allocation, at the same size (within 0.1 %), and their gates.
Agent scores on the harder split are in Table 2; on the standard split, S = selection seeds 100 + 101 and fresh =
seeds 104 + 105, each of 336.*

| Map | How chosen | KL vs start | Recall at 117k | Standard split vs start (S; fresh) | Verdict | Ref. |
|---|---|---:|---|---|---|---|
| KL-group | measured KL per group | -9.3 % [-10.5, -8.3]; p2 -8.2 % | 23/24 + 19/20 (a needle lost). Held-out (B1): no loss past the refit floor | not run | fail on the gate, not confirmed on held-out documents | [E1.3, E1.34] |
| KL-table A | KL table | -8.5 % (subset) | 24/24 + 19/20 (fragile item only) | not run | fail as written (weak) | [E1.6] |
| KL-table B | KL table | -10.6 % (subset); -8.8 % (p1, vs shipped) | needle margin -1.94; held-out (B1): long-range margin loss | not run | fail; lowest score on the harder split | [E1.7, E1.34, E5.2] |
| KL-only winner | KL table | -7.0 % [-8.3, -5.6] | 24/24 + 20/20 (also at 245k) | 285 vs 302; 296 vs 292 | loses on the harder split | [E1.8, E1.31, E5.2] |
| KL-tools | KL table, tools-aimed | -7.9 % [-9.1, -6.9] | 24/24 + 20/20 | 301 vs 302; 299 vs 292 | level on agent work; long thinking on one deep item in free generation (an observation, Section 6); not shipped | [E1.9, E3.16, E5.11, E5.26] |
| Uniform (baseline) | uniform rule: 3 bits on the last 30 layers | -3.8 % [-6.0, -1.6]; p2 -1.2 % [-6.6, +4.4] | 24/24 + 19/20 (fragile item only); more margin on the other 43 | not run (rule) | fail as written (weak); chat +13.6 % | [E1.28, E1.33] |

"Fragile item" in Table 3 means one code-copy item (`_check_tzname`) on which the shipped map's own margin is 0.43
nats. Our rule calls an item fragile when the shipped map's own minimum margin is below 1 nat; a flip there can be
noise, so a gate miss on that item alone is weak evidence [E1.6, E1.33].

KL-group was better than the start on every KL line, including the untouched half, every document class, and every depth
band up to 256k tokens in both KL and the model's own NLL of the true next token [E1.4]. It still lost one needle and
one code copy at 117k. The KL-only winner passed every KL, recall and code gate. On the selection seeds of the
standard split it was 17 task-runs below the shipped file; that difference is not significant at the seed level
(p = 0.13), and on two fresh seeds the two were level (+4, interval [-19, +27]) [E1.8, E1.31, E5.28]. The standard
split does not separate these maps. The harder split does (Table 2).

**Held-out recall population (B1).** After an adversarial review of the draft by an LLM assistant, at the author's request (see AI use), the author fixed
three tests and their lines before
any of them ran, with a family alpha of 0.0167 (Bonferroni over the three); no other test was added until one of them
passed. The first asks whether a lower-KL map loses recall on material that no choice used. It has 20 haystack
documents built from standard-library code that no recall gate of the selection used, 240 needles and 240 distinct
functions at 64k, 117k and 245k tokens, and a shallow twin of every 117k and 245k item: the same question on a
haystack of about 3.9k tokens around the item, so that only the distance changes [E1.34]. Per map we compare the capped
item margin with the shipped file: L over the long items, S over their twins, and the depth contrast C = L - S, with
99.72 % bootstrap intervals over documents (Bonferroni over the six lower-KL maps inside the family alpha). The floors
come from a repeat run and from a second fit of the shipped map (0.212 nats for L, 0.295 for C).

One of the six lower-KL maps meets the line. KL-table B has L = -1.07 nats [-1.28,
-0.86] and C = -1.14 [-1.36, -0.94], with no loss at short distance (S = +0.07) (Figure 4) [E1.34]. The loss is on
needles at depth: -1.23 nats at 117k and -3.26 at 245k, against -0.16 at 64k; the copies are flat. It is a loss of
margin, not of answers: no long item went below zero, so greedy decoding still answers every item. The other five
lower-KL maps do not meet the line. This includes KL-group, whose needle loss on the gate's one document (Table 3) is
small on held-out documents (L = -0.13 [-0.24, -0.03], inside the refit floor). Two lower-KL maps have slightly more
long-range margin than the shipped file: the KL-only winner (L = +0.09 [+0.01, +0.18]) and the uniform map
(L = +0.13 [+0.05, +0.21]), both inside the refit floor [E1.34].

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
copies and twins are flat. Confirmed on one box with one GPU model (run floor 0.000) [E1.34, E1.35].](figs/fig4-b1-recall-population.png){width=100%}

**The held-out loss sits in one group, in that map.** KL-table B differs from KL-group, which kept its held-out
margin, in four groups of projections. We built four flip maps, each KL-table B with one group set back, and ran
each on the B1 population, with the line written first. One group carries the loss: with the MLP down-projections of
layers 16-31 back at 3 bits, L is -0.02 [-0.13, +0.08], which recovers 98 % of the loss (+1.09 nats [+0.92, +1.24]).
The other three flips recover 7-27 % each and still lose recall at depth [E5.15]. Two limits apply. First, the cut is
not sufficient on other maps: the KL-only winner, the uniform map and the HAWQ-style map also hold these
down-projections at 2 bits [E5.13, E5.19], and all three keep their held-out margin (L = +0.09, +0.13 and +0.06)
[E1.34]. Second, the recall loss and the agent loss are different effects: the same flip map is still 43 task-runs
below the shipped file on the harder split (Table 2), and the agent loss of KL-table B is larger on short tasks, not
on long ones [E5.10, E5.14].

**The recall probe does not predict the agent result.** KL-table B has the held-out recall loss and the largest agent
loss. The KL-only winner has an agent loss of 60 task-runs and slightly *more* held-out margin than the shipped
file. So the recall probe saw one of the two damaged maps. The standard agent split did not separate the one of the
two that ran on it (the KL-only winner).

A local proxy did worse than KL: a map chosen by the sum of per-projection Hessian-weighted fit errors was 36.7 %
*worse* in KL at the same size, because it took bits from the last MLP layers, whose own fit error is small but whose
effect on the output is large (+46.2 % KL for that family alone) [E1.1, E1.2]. A HAWQ-V2-style map (average Hessian
trace times the squared fit error, exact knapsack at the same bits) was 12.7 % worse in KL [+10.6, +14.7] and also
lost one code copy [E1.29]. A uniform baseline (3 bits on the last 30 layers) beat the shipped map on KL on the build
half (-3.8 % [-6.0, -1.6]; untouched half -1.2 % [-6.6, +4.4], chat documents +13.6 %) and lost the same code copy
[E1.28]. Both baseline misses are on the one fragile item (an exact function body with the comment
line above it copied too; the shipped map's own margin there is below 1 nat), so we count them as weak evidence;
the shipped file passed that item in the same run.

Graded margins on the same 44 items at 117k (the margin probe of this section, with lines written before the run)
show no general recall loss for either baseline; they point the other way [E1.33]. On the 43 items without the fragile
one, the mean difference of the capped item margin against the shipped file was +0.164 nats [+0.016, +0.379] for the
uniform map and +0.148 [+0.016, +0.356] for the HAWQ-style map: by the line written first, both have more recall
margin than the shipped map. The gain sits in few items. Most items are at the 5-nat cap in every arm; the difference
comes from the two hardest needles of the shipped file (vault 8 and vault 10, +1.7 to +3.7 nats) and from one copy
function asked three times (+0.34 each). Counted once per function, the lower end of both intervals is +0.000. On the
fragile item both baselines are below zero (-0.325 and -0.258, against +0.264 for the shipped file), so their
recall-gate misses are that item only [E1.33].

![Graded recall margins per item, sorted, capped at 5 nats. (a) Qwen, 117k, the 44 recall items: uniform and
HAWQ-style baselines against the shipped file; 6 of 44 items differ, the rest sit at the cap in both arms [E1.33].
(b) Inkling 2 Mini, 117k and 245k (88 item-depth units): blanket q4_0, blanket q8_0 and the 4-bit global cache
against the release K/V setting; wide spread, no clear shift of the mean [E2.23].](figs/fig5-item-margins.png){width=100%}

**A naive 2-bit file on fresh seeds.** On fresh seeds of the standard split, the shipped build beats the naive 2-bit
file (Unsloth UD-Q2_K_XL, a K-quant) in the same engine by 36 task-runs of 336: 292 against 256 on seeds 104 + 105
(+15 and +21; seed-level p = 0.0056; interval [+11, +61]) [E1.32, E5.28]. This is the second of the three tests
that the author fixed before their results existed, and it passes. It compares a
trellis build with a K-quant file: the codec, the fits and the bit allocation all differ, so it does not isolate the
allocation method. The naive file ran with its own chat template; the first prompt of every
task-run had the same token count as the shipped file's (336 of 336). On the selection seeds the gap was 35 (302
against 267) [E1.20].

**One gate item traces to a band of up-projections, but that band does not explain the held-out loss.** Margins on
the gate's own prompts turned the pass / fail needle into a continuous, exact measurement. The failed needle
("vault 8") had a margin
of +1.62 nats on the start map. It was wrong, or between -1.6 and -2.1, on every map but one that moved the MLP
up-projections of layers 21-47 to 2 bits. One of those maps has only that change (its 32 up-projection downgrades are
layers 21-47 and five earlier layers, as in every map of this bisection). Every map without that change kept
the needle [E1.5]. A one-group flip table on the shipped file (21 rows) agrees, and adds two small groups: the
attention output projections (15 MiB) and the attention key / value projections (20 MiB) each pushed the needle below
zero [E1.11]. KL rates the layer 21-47 up-projections among the cheapest sources (0.024-0.026 % KL per MiB) [E2.13].

Three results limit this. First, the effect does not add. The one exception has the same band change and more bits
in two other groups: the attention q / k / v projections of the 48 linear-attention layers, and the MLP gate of
layers 35-38 (+339 MiB in total). It answered vault 8 correctly. Neither of the two groups alone restored the needle
[E1.5]. The uniform baseline holds every projection of layers 0-33 at 2 bits, which includes the up-projections of
layers 21-33 that the shipped map holds at 3 bits. Its vault 8 margin was +4.14 nats, against +1.34 for the shipped
file (the start map with its 8-bit head) in the same run [E1.28, E1.33]. So a flip table from one start map does not
predict the item on a different map. Second, this band does not explain the held-out loss. KL-group and KL-table B
both have the band change. On the held-out documents, KL-table B lost 1.07 nats and KL-group lost 0.13, which is
inside the refit floor [E1.34]; the held-out loss of KL-table B sits in the down-projections of layers 16-31 (above).
Third, the same up-projection band is the second cut of KL-table B, and it is a candidate, not more, for the agent
loss that stays when the down band is restored [E5.14]. So the claim is narrow:
exact margins can trace the loss of one item to a group of layers on one map.

**The reverse also holds.** KL rates the up-projections of layers 56-63 as the *worst* source (0.148 % per MiB).
They broke no recall item. A candidate that took all of layers 48-63 kept every recall item above zero but was
8.1 % worse in KL on the untouched documents [+3.2, +13.2], and 8-17 % worse on five document classes [E1.12]. Both
signals are needed. Over the 16 groups with both numbers, the rank correlation of KL cost and task cost was
$\rho = 0.007$ [-0.56, 0.62] [E1.10]; this interval is wide because most single-group task changes are inside their
noise, so we do not read it as evidence that the rankings differ, only that KL does not predict the task ranking
here. The task-guided search itself found no map that beat the start: the best of five task-guided maps missed one line
(a 356-row non-English chat document) and the study is paused [E1.13].

### 4.5 Local repair beats blanket precision

**K/V cache, by cache class (Inkling 2 Mini).** The model has 48 K/V caches: 8 for global-attention layers, which
grow with context, and 40 for sliding-window layers, which are rings of about 1,024 cells. A blanket q4_0 cache was
far outside our line (KL 0.0092 at 2k against the same weights with f16 K/V) [E2.1]. Table 4 shows the scan. The
40 sliding-window caches cost 240-320 MiB at f16 at any context length, while the global caches need 32 KiB per token
at f16 and cannot hold 262k tokens on 12 GB [E2.3]. So the repair spends precision where it is cheap per byte: f16
for the sliding-window caches and the cheapest type that passes the line (q8_0) for the global ones.

*Table 4. K/V type per cache class, KL against the same weights with f16 K/V (Unsloth's file, so only the K/V type
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
setting that the tests compared against. On AppWorld it scored 595 of 672 against 592 for the official weights
(interval [-13, +19]) and 596 for the 8-bit cache (interval [-16, +14]) [E4.12, E5.28]; its KL against the official
weights is measured only on a 10-document subset. On
graded recall margins the result is inconclusive: -0.120 nats [-0.320, +0.090] against the 8-bit global cache. The
interval is not below 0, but its lower end is below the non-inferiority margin of -0.25 nats that we wrote before the
run. One copy item (`_decimal_lshift_exact`, asked twice at 117k) went from +1.76 and +1.14 nats to -0.10 and -0.41,
at the third-last answer token [E2.23]; the pass / fail gate had counted every copy at this depth as correct [E2.18].

**A repair that KL rewards and tasks do not show (27B).** The L0xRE-V2 package stores the output head and the
embedding at 3 bits, a memory trade of its author. Our build has an exact 8-bit head and embedding (an exact 8-bit
head came first from the Escha Labs export) and fits them in 12 GB with a two-stage head. On the same body, the
exact head lowers KL from 0.0889 to 0.0602, which is 32 % [E5.17]. Its task test had its line written first. On the
harder split, with the two-stage head off in both arms, the shipped file scored 687 and the same file with the 3-bit
head and embedding scored 670; without the 19 mixed task-runs of one doubled shard of seed 107 the difference is +15
(+16 and -1; seed-level p = 0.46; interval [-25, +55]) [E6.3]. The result is "not shown on
tasks" [E5.17]. This is the largest KL change of any single repair in this paper, and the agent benchmark does not
see it at this size of test.

**Shipped runtime settings cost nothing measurable on tasks.** On the standard split, the shipped 27B build (q4_0
K/V tiered to host memory, two-stage head, drafting) scored 294 of 336 on fresh seeds; the same weights with f16 K/V
all in GPU memory and no head or drafting changes scored 295 (difference -1, interval [-21, +19]) [E2.14, E5.28]. On
the harder split the same comparison gives 670 against 670 (interval [-40, +40]) [E5.5]. On Inkling 2 Mini, M2 with
the release K/V setting (q8_0 global + f16 sliding-window) against the official weights with f16 K/V everywhere scored
596 against 592 of 672 [E2.19]; that test covers the weights and the K/V setting together and does not separate them.

**Repairs that did not pay.** See Section 5.

### 4.6 Select, then confirm on fresh data

On AppWorld seeds 100 and 101 (standard split) the shipped 27B build was compared with three alternatives (two of our
KL-guided maps and the original author's own fits) and kept as the best of the four: 302 against 301, 288 and 285
[E3.8]. A comparison with BF16 on the same seeds then showed the quantized build *ahead* of BF16 by 21 task-runs
[E3.6]. Rather than report that, we re-ran
both on two seeds never used for any choice, and later all four candidates on two more (Table 5).

*Table 5. AppWorld test_normal, pooled task-runs of 336 per seed pair. S = selection seeds; the other two pairs were
never used for any choice. Gap intervals are cluster-robust 95 %; p is the seed-level p [E5.28].*

| Arm | S: 100 + 101 | 102 + 103 | 104 + 105 | Ref. |
|---|---:|---:|---:|---|
| Kintsugi-27B as shipped (12 GB GPU; the selected candidate) | 302 | 294 | 292 | [E3.1, E3.2, E3.15] |
| KL-tools (candidate) | 301 | not run | 299 | [E3.8, E3.16] |
| Author's fits of the shipped layout (candidate; its own chat template) | 288 | not run | 289 | [E3.8, E3.16] |
| KL-only winner (candidate) | 285 | not run | 296 | [E3.8, E1.31] |
| Kintsugi-27B weights, f16 K/V (80 GB GPU) | not run | 295 | not run | [E3.3] |
| BF16 (never selected) | 281 | 293 | 294 | [E3.4, E3.5, E3.16] |
| Inkling 2 Mini, Unsloth UD-Q2_K_XL (never selected) | 283 | 283 | 281-283 | [E3.9, E3.13] |
| **Gap shipped - BF16** | **+21 [-1, +43], p = 0.057** | **+1 [-22, +24], p = 0.93** | **-2 [-25, +21], p = 0.87** | [E3.6, E3.7, E3.16, E5.28] |

On seeds 102 + 103 the quantized build is not significantly different from BF16 (294 vs 293; weights-only 295 vs
293, interval [-19, +23]) [E3.7, E5.28]. This is a non-significant difference, not a proof of equivalence: the
intervals are about 23 task-runs of 336 wide on each side. On seeds 104 + 105 it was 292 vs 294 [E3.16]:
again not significantly different. The +21 on the selection seeds is not a result. Versions 1.0.x gave it a
task-level p of 0.0025; at the seed level it is 0.057 [E5.28].

![The selection-seed lead on the standard split. (a) AppWorld score per seed pair for the shipped (selected) file,
BF16 (never selected) and the three other candidates of the selection; candidates other than the shipped file have no
run on 102 + 103. (b) The shipped - BF16 gap with cluster-robust 95 % intervals and seed-level p: +21 on the selection
seeds, +1 and -2 on the two fresh pairs [E3.1-E3.8, E3.15, E3.16, E5.28].](figs/fig6-seed-pairs.png){width=100%}

How much of it was selection? We wrote the test before the run: on seeds 104 + 105, the selection effect
$\Delta_{\mathrm{sel}}$ is the winner's drop from S minus the mean drop of the other three candidates, and we claim
"the selection seeds inflated the winner" only if its bootstrap interval over tasks is above 0. The winner fell by 10;
the others moved by -2, +1 and +11 (the KL-only winner rose from 285 to 296). So $\Delta_{\mathrm{sel}} = +13.3$ task-runs,
95 % interval [+0.0, +27.0] [E3.15]. The interval is not above 0, and it is a task-level interval, which is too
narrow (Section 4.1). The drop is within seed-to-seed variation; we cannot
separate selection bias from noise with this data. The never-selected BF16 arm also moved: by +12 from the selection
seeds to 102 + 103 [E3.10] and by +13 to 104 + 105 [E3.15].

The same check overturned a public claim. Our published head-to-head against Inkling 2 Mini (Unsloth's 2-bit file on
its own build) said the 27B build was better on AppWorld (302 vs 283). On fresh seeds it was 294 vs 283
(interval [-10, +32]), and we withdrew the claim on the public page the same day [E3.9, E5.28]. Inkling's score was
the same on both seed pairs. At the seed level the first result was not significant either (p = 0.090) [E5.28].

**The check also holds a real result.** Our Inkling build against Unsloth's setup gave
298 vs 283 on seeds 102 + 103 (+6 and +9; seed-level p = 0.021) [E3.12, E5.28]. No choice for M2 used AppWorld, so
this was not a selected
winner, but because of the withdrawal above we re-ran both setups on seeds 104 + 105 before any public claim, with the
rule written first: keep "better" only if ours is at least equal there. It was 298 vs 281-283 (two Unsloth task-runs
did not finish at the output cap, so we give a bound; on the 334 finished pairs +4 and +11, seed-level p = 0.012)
[E3.13, E5.28]. No Inkling arm moved by more than 4 task-runs between seed pairs [E3.14], against +12 for the
never-selected BF16
arm on Qwen. This is a setup-against-setup result (engine, K/V type and reasoning budget differ). Section 4.7 gives
the comparison with the setup equal.

**Lesson (a procedure, not a measured effect).** A configuration chosen by its score on a benchmark needs a held-out
confirmation before any comparison with another model or with its reference weights is claimed, because of the
winner's curse [Thaler; Cawley & Talbot], whose size our data could not measure. In this study the check removed
one public claim and two claims of our own first draft on the standard split, and it kept one. The seed-level
re-analysis of Section 4.1 is the same lesson applied to our statistics.

### 4.7 A second model: Inkling 2 Mini

The same loop ran on Inkling 2 Mini with the same tools: fit library, KL gate with paired bootstrap, recall gate at
depth, pre-written lines. Both builds run on one 12 GB GPU with a 262,144-token window (K/V past the GPU limit is kept
in pinned host memory and streamed; the host path matched the all-GPU path to KL < 1e-6 [S.1]).

**The map.** Our engine supports 2- and 3-bit trellis codes. Two maps were named before any result: an expert-layer
map (both expert matrices at 3 bits in 11 layers) and M2 (all down projections at 3 bits, gate / up at 3 bits only in
the 4 global layers). Both passed the line (at least 10 % lower KL than Unsloth's UD-Q2_K_XL, interval below zero,
top-1 not lower, file not larger); M2 was better and smaller [E2.5]. One structural line
(per-document correlation of engine and simulation >= 0.95) failed as written (r = 0.919); the same statistic for
Unsloth's file was 0.912, and on documents with at least 500 scored rows it was 0.997, so the line sat below the
noise floor of short documents [E2.6]. We record the miss and the decision to proceed separately. The recall checks
were full for every arm, so the choice of M2 was a choice on KL.

**The task result, with the setup equal.** Versions 1.0.x compared our build with Unsloth's file as two whole setups
(Section 4.6), and credited "the same steps" for a 27 % lower KL. A reviewer asked for the baselines that isolate the
method. We ran three files through one engine build and one launcher, with the K/V types, the window and the
reasoning budget equal: M2; Unsloth's UD-Q2_K_XL file (0.78 % larger); and a uniform trellis map at the same bytes as
M2, whose rule we fixed before the build (3 bits for both expert matrices of the last layers, 2 bits elsewhere). The
lines were written before any build or run [E5.20-E5.23]. Table 6 gives the agent scores.

*Table 6. Inkling 2 Mini on AppWorld, task-runs per seed. Standard split: one engine, launcher and setting for the
three files. Harder split: the official weights ran with f16 K/V on 48 GB cards (a different setup) [E5.22, E5.23].*

| Arm | 108 | 109 | 110 | 111 | Standard split, of 672 | 112 | 113 | Harder split, of 834 |
|------------------------|----:|----:|----:|----:|----------:|----:|----:|----------:|
| M2 (ours) | 151 | 149 | 148 | 150 | 598 | 311 | 327 | 638 |
| Uniform map, same bytes | 149 | 147 | 148 | 152 | 596 | 326 | 323 | 649 (of 833) |
| Unsloth UD-Q2_K_XL, same engine | 144 | 141 | 146 | 148 | 579 | not run | not run | not run |
| Official weights (NVFP4 / MXFP8) | not run | not run | not run | not run | 592 (seeds 102-105; M2: 596) | 306 | 298 | 604 |

- *M2 against the public 2-bit file.* 598 against 579: +7, +8, +2 and +2 on the four seeds; seed-level p = 0.019;
  cluster-robust interval [+3, +35]; the t-test over the four seed differences gives p = 0.059 [E5.22]. With the setup
  equal, the M2 file scores higher than Unsloth's file on agent work.
- *M2 against the official weights.* On the standard split the two are level: 596 against 592 on seeds 102-105
  (interval [-10, +18]) [E4.8, E5.28]. On the harder split M2 scored 638 against 604 (+5 and +29). The line was a
  non-inferiority line (margin -25 of 834), and it is met: the cluster-robust interval is [+6, +62]. We claim "not
  worse", not "better": the two arms differ in the K/V precision and the card class, and the gain is in one of the
  two seeds [E5.23].
- *M2 against the uniform map.* Level on both splits: 598 against 596 (interval [-12, +16]), and 638 against 649 on
  the harder split (uniform +15 and -4; seed-level p = 0.42) [E5.22, E5.23].

![Inkling 2 Mini on AppWorld test_normal, per seed. (a) Seeds 102-105: M2 with the release K/V setting, M2 with the
4-bit global cache, the official weights (NVFP4 / MXFP8) in the same engine, and Unsloth's setup (setup against
setup; two of its task-runs on seed 105 did not finish) [E3.12, E3.13, E4.8, E4.12]. (b) Seeds 108-111, one engine
and one launcher setting for the three files: M2, the uniform map at the same bytes, and Unsloth's UD-Q2_K_XL file
[E5.22]. Totals in the legends.](figs/fig7-inkling-appworld.png){width=100%}

**Where the KL gap comes from.** Table 7 puts the KL numbers next to the task numbers.

*Table 7. Inkling 2 Mini: three files in one engine build. KL against the official weights on 298 documents
(451,585 rows). Graded recall: mean difference of the capped item margin against M2 over 88 item-depth units at 117k
and 245k, with 97.5 % intervals (two comparisons) [E5.20, E5.21].*

| | M2 (ours) | Uniform map, same bytes | Unsloth UD-Q2_K_XL |
|---|---:|---:|---:|
| File size | 9,067,638,560 B | 9,067,638,624 B | 9,138,715,328 B |
| Mean KL | 0.03103 | 0.03182 | 0.04248 |
| KL against M2 | - | +2.5 % (97.5 % interval holds 0) | +36.9 % (M2 is 27.0 % lower) |
| Top-1 agreement | 93.64 % | 93.50 % | 92.71 % |
| Recall counts, 24 + 20 at 32k / 117k / 245k | full | full | full at 117k and 245k; 19 / 20 strict copies at 32k (an exact body with one comment line added; not read as damage) |
| Graded recall against M2, all units | - | -0.97 [-1.66, -0.34] | +0.87 [+0.31, +1.44] |
| Graded recall against M2, copy items | - | **-2.73** [-3.39, -2.10] | **+2.34** [+1.81, +2.87] |
| Graded recall against M2, needle items | - | +0.49 [+0.22, +0.81] | -0.36 [-0.68, -0.04] |
| AppWorld, standard split (Table 6) | 598 | 596 | 579 |

Three readings follow.

1. *The KL gap to the public file belongs to the codec, the fits and the engine format.* M2 has 27.0 % lower KL than
   Unsloth's file on this engine build (27.3 % on the build of versions 1.0.x [E2.6]). The uniform map, with no
   per-group choice, is already 25.1 % below Unsloth's file. That is 93 % of the gap. The per-group map adds 2.5 %,
   and its 97.5 % interval holds 0 [E5.20]. So we no longer credit our method's steps for that KL number.
2. *The down projections carry long-range copy margin.* The uniform map moves the down projections of layers 0-13
   to 2 bits; M2 keeps every down projection at 3 bits. KL and the agent benchmark do not separate the two files. The
   graded recall margins do: the uniform map has 2.73 nats less margin on the copy items, and 16 of its 88 units are
   below zero, all of them copies. The pass / fail recall gate had counted every one of its copies as correct,
   because that gate checks the extracted code and the margin covers every forced token [E5.21].
3. *The same measure is against our build.* Unsloth's file keeps its down projections exact (the vendor's 4-bit
   format) in 11 layers and gives its gate / up projections fewer bits. It has 2.34 nats *more* copy margin than M2,
   0.36 nats less needle margin, and no unit below zero (M2 has 3) [E5.21]. KL ranks the two files the other way, and
   so does the agent benchmark. The higher KL of M2 on copy rows that versions 1.0.x reported (0.760 against 0.608 in
   the engine) points the same way as the margins [E1.16, E1.18]; that metric saturates (0.19 with no weight change),
   so we report it and do not rank by it [E1.17].

**An allocation rule from one lineage did not carry to another.** Inkling 2 Mini is not a Qwen derivative: it is
another base lineage and another architecture (a looped mixture of experts). On the 27B model, the repair in context
was in the MLP down-projections (Section 4.4). M2 keeps every down projection at 3 bits, and the uniform map does not. We wrote the
prediction before any score was read: the uniform map loses agent work on the harder split. It did not: 649 against
638 for M2 [E5.23]. A second line asked if the tool-call documents show a larger KL gap for the uniform map than
all rows do, with its interval above 0; the gap is larger (+14.5 % against +2.5 %) but its interval holds 0
[E5.20, E5.23]. So we make no claim about agent work for the down projections on a second lineage. The method expects
this: it finds the allocation again for each quantized model, on that model's own probes. What transferred from the
27B model is the procedure and two facts about instruments: KL alone does not decide, and the group that KL treats as
equal can carry a behaviour that only a graded probe shows. What carries within one lineage is knowledge of the base
model (Section 4.9).

**A third-party file.** Versions 1.0.x compared M2 with a LookAtMyQuant file. To be exact, the file that we measured
is the 8.9 GB pack at Hugging Face revision 1318b804e687 of 2026-11-21 (63 files, 8,874,213,027 B), which its author
now calls v1 [E5.24]. It is level with M2 on average KL (-3.3 %, interval [-0.0037, +0.0016]) and 62 % higher on copy
rows; by document group it is 25 % better on English prose and 29-83 % worse on code, tools, agent traces and
reasoning [E1.19]. Two files with the same average differ by class by up to a factor of almost two. A check that we
made for version 1.1 found that 3 of our golden web documents are word-for-word copies of public FineWeb-Edu
documents that are in the calibration text of that file, and that 2 agent-trace documents share 6-8 % of their text
windows with it. Without these 5 documents the numbers are: all rows -3.5 % [-0.0039, +0.0015], prose -24.9 %,
agent traces +90 % (2 documents left), copy rows +47 %. The reading is the same [E5.25]. A newer file of the same size
exists: revision 6a953d2d1787 of 2026-11-23 (code at the author's GitHub commit 826e5dde). Its author reports, on his
own 6-class suite and reference, 62.5 % lower mean KL than his v1 file and 70.1 % lower on tools and on agentic text
(model card at that revision). These are his numbers; we did not measure them. The author's README (commit 92bb52d) lists eight ideas taken from the Kintsugi repositories, among them per-class KL on chat-templated text and dense expert Hessians. We did not measure
what these ideas do in his file. The new file is not yet measured on our golden set [E5.24]. No sentence of this paper about LookAtMyQuant applies to the new file.

*Table 8. The two builds side by side.*

| | Qwen3.8-27B (dense hybrid) | Inkling 2 Mini (looped MoE) |
|---|---|---|
| File | 10.55 GB | 9.07 GB |
| KL against the reference | 0.0579 vs BF16 at shipped settings [E4.5] | 0.0310 vs the official weights [E5.20] |
| Recall 24 + 20 | full at 117k and 245k [E4.3] | full at 64k, 117k and 245k [E4.4] |
| Agent, standard split, against the reference weights | 294 vs 293 and 292 vs 294 of 336: level [E3.7, E3.16] | 596 vs 592 of 672: level [E4.8]; with the 4-bit global cache 595 [E4.12] |
| Agent, harder split, against the reference weights | at temperature 1.0: +117 of 1,668 over four seeds; in BF16's setup 670 vs 614; at temperature 0: 330 vs 319, not significant [E5.4, E5.5, E5.18] | 638 vs 604 of 834: not worse; the setups differ [E5.23] |
| Agent against a public 2-bit file, one engine | 292 vs 256 of 336 (a K-quant file) [E1.32] | 598 vs 579 of 672 [E5.22] |
| Code (health gate) | HumanEval 159-161 / 164 [E4.9] | HumanEval 158 / 164 [E4.11] |

**A decision that these totals hide.** The AppWorld totals of M2 and of the official weights are level. Section 4.10
shows one decision at which the two files differ by a wide margin, the first call of a task, and shows that the
difference has no measured task cost.

### 4.8 Faults outside the weights (Inkling 2 Mini)

The release candidate engine passed its
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

### 4.9 Readings across compressions of one base

The 27B files of this paper are compressions of one base model: the shipped file and the seven other quantized arms
of Table 1c, beside BF16 itself. We read such a set with a fixed rule. A behaviour that is in every compression and in
the reference is a trait of the base. A behaviour that the compressions share and the reference does not is shared
damage of what the files share (the codec and most of the fits). A behaviour in one map only is a candidate for damage
in that map. Four readings follow.

1. *The unasked answer* (Section 4.3) is in BF16 and in all eight quantized arms: a trait of the base. The maps
   change its rate, from 4.7 % to 17.3 % of the task-runs [E6.4].
2. *The vault 9 departure* (Section 6): the shipped file and KL-tools leave BF16's path at the same two early tokens,
   where BF16's own gaps are 1.41 and 2.95 nats. That is shared damage of the two files, not damage in the new map
   [E5.31].
3. *Blind instruments.* KL, the recall margins and the held-out recall test did not see the free-generation change
   (Section 6), and the teacher-forced agent-trace probe did not see a known loser (Section 3). Each is blind in the
   same way on every map.
4. *The returned tool* (Section 4.10): BF16 and every quantized file that we ran miss the return turn of BFCL
   miss_func. It is a trait of the base. The files change its rate: in one setup, from 36 of 200 (shipped) to 61 of
   200 (BF16) [E7.4].

For the next compression of the same base, these are known before any run: the habits to count, the tokens where the
quantized files tend to leave the reference, and the instruments that cannot see a given failure. They are a product
of the method. They did not carry to another lineage: on Inkling 2 Mini the down-projection rule of the 27B model did
not hold (Section 4.7). The reading itself did carry: on Inkling 2 Mini the first-call decision is in the official
weights and in a standard quant at one rate, and M2 changes that rate (Section 4.10).

### 4.10 Decision-point habits: detect, replay, locate

Section 4.3 found that one decision carries a benchmark gap. This section makes that reading a procedure. It applies
the procedure to a second decision of the 27B model and to a second model family.

A *decision-point habit* is a choice that a model makes at one kind of turn, at a rate that a count over many entries
can measure. The procedure has four steps.

1. **Detect.** Split a benchmark by turn type. Look for a turn type where two files differ.
2. **Freeze and replay.** Record one history. Send the one call that decides the turn to each file again.
3. **Locate.** Copy tensors between two files, and count again.
4. **Repair.** Build a map that removes the habit. Confirm it on turns that no step used.

This paper does steps 1 to 3. Step 4 is open: we state this at the end of the section. "Habit" means what these
benchmarks punish, or what they count. It is not a general statement of quality.

**Detect: the totals are level, and one turn type is not.** BFCL v3 multi-turn [BFCL] has 800 entries in four
categories. In the category miss_func, one function is held back at first. A later turn (the *return turn*) puts it in
the tool list, with the fixed user line "I have updated some more functions you can choose from. What about now?". The
ground truth has calls at every return turn. We ran a third candidate map, KL-tools-3 (KL-tools with 29 projections
set back to the shipped widths; it passed our long-copy gates). Its totals are level with the shipped file: 493
against 505 of 800 on BFCL (scenario p = 0.34), and 1,354 against 1,335 of 1,668 on AppWorld test_challenge (four
fresh seeds, seed-level p = 0.53) [E7.1]. A split of the BFCL entries by turn type showed one cell. At the return turn
KL-tools-3 gave no tool call in 48 of 198 entries, and the shipped file in 28 of 199 (paired 29 / 9, exact sign
p = 0.0017). This cell is 1 of about 10 turn classes, and we found it after the run [E7.1]. A repeat of that turn on
our workstation gave 57 of 198 against 32 of 200 (paired 35 / 10, p = 2.5e-4) [E7.2].

**Freeze and replay.** The benchmark records could not be replayed. The harness keeps the calls, but 63 % of the
tool-call replies also hold a short text before the call, and the record does not keep it [E7.2]. So we ran each of
the two files again on our workstation with every request kept: the official multi-turn loop, cut after the return
turn. HS\* is the history that the shipped file made (200 entries). HT\* is the history that KL-tools-3 made (198
entries; two entries are out). A *unit* is the first model call of the return turn of one entry on one history. The
replay sends that exact request to a file, with the prompt cache off, at the benchmark's temperature (0.001) and with
the same seed. One cell (one file on one history) takes about 35 minutes on one 12 GB card. The benchmark run takes
about 9 hours of model time.

Checks of the instrument [E7.2]: every replay prompt has the token count of the live call (200 of 200 and 198 of
198). Two repeats of one cell, one with the same seeds and one with other seeds, gave the same outcome on 200 of 200
units. At this temperature the replay is deterministic. The replay of a file on its own history equals its live call
on 95.0 % (shipped) and 89.4 % (KL-tools-3) of the units; the live calls ran on the workstation and the replays on a
rented box with the same card type.

**Two miss rules.** Rule 1: the reply has no tool call. This is the rule of the study's plan, and every line was
written for it. Rule 2: no call names the returned function. We added rule 2 after the first results, so it is
exploratory. It is stricter: a reply that calls other functions is a miss. Table H1 gives both rules for every
comparison. We say which results hold under both.

*Table H1. The return turn. Misses of file X against file Y on the same requests, under both rules: counts, then the
units where only X misses / only Y misses, then the exact sign p. "Both, by entry": the entry is the unit and the two
histories are summed (the plan's pooled test); the first number is the difference in misses. All cells are
near-greedy (temperature 0.001) except the sampling row [E7.2, E7.4-E7.7, E7.9].*

| X against Y | History | Rule 1: no call | Rule 2: returned function not called |
|--------------------------------|-------------|-----------------------------|-----------------------------|
| KL-tools-3 against shipped | HS\* (200) | 48 / 40; 17 / 9; p = 0.17 | 80 / 61; 31 / 12; p = 0.0054 |
| | HT\* (198) | 58 / 31; 31 / 4; p = 3.5e-6 | 84 / 59; 32 / 7; p = 7.0e-5 |
| | both, by entry | +35; 40 / 10; p = 2.4e-5 | +44; 51 / 14; p = 4.5e-6 |
| BF16 against shipped, one 80 GB card, one setup | HS\* | 61 / 36; 33 / 8; p = 1.1e-4 | 77 / 59; 33 / 15; p = 0.013 |
| KL-tools-3 against shipped, the same card and setup | HS\* | 53 / 36; 24 / 7; p = 0.0033 | 81 / 59; 33 / 11; p = 0.0013 |
| BF16 against KL-tools-3, the same card and setup | HS\* | 61 / 53; 23 / 15; p = 0.26 | 77 / 81; 22 / 26; p = 0.67 |
| Shipped with earlier thinking stripped, against shipped | HS\* | 14 / 40; 6 / 32; p = 2.4e-5 | 31 / 61; 5 / 35; p = 1.4e-6 |
| | HT\* | 13 / 31; 6 / 24; p = 0.0014 | 41 / 59; 7 / 25; p = 0.0021 |
| KL-tools-3 against shipped, both with thinking stripped | both, by entry | +12; 16 / 4; p = 0.012 | +6; 12 / 5; p = 0.14 |
| Wording (a), KL-tools-3 against shipped | both, by entry | 13 / 5 misses; +8; 10 / 3; p = 0.092 | 66 / 60 misses; +6; 13 / 7; p = 0.26 |
| Wording (b), KL-tools-3 against shipped | both, by entry | 91 / 78 misses; +13; 29 / 15; p = 0.049 | 268 / 255 misses; +13; 34 / 21; p = 0.10 |
| Wording (c), KL-tools-3 against shipped | both, by entry | 97 / 28 misses; +69; 58 / 5; p = 1.7e-12 | 127 / 63 misses; +64; 56 / 8; p = 5.6e-10 |
| Sampling, three seeds, KL-tools-3 against shipped | both, by entry | 350 / 210 of 1,194; +140; 91 / 22; p = 3.7e-11 | 511 / 370 of 1,194; +141; 102 / 28; p = 4.4e-11 |
| Second-fit file against shipped | HT\* | 46 / 31; 29 / 14; p = 0.032 | 70 / 59; 30 / 19; p = 0.15 |
| | both, by entry | +26; 47 / 25; p = 0.013 | +23; 51 / 29; p = 0.018 |

**The replay is valid, and the gap depends on the history.** The plan's line asked for the gap on one history or
more (10 misses or more, p < 0.05), or on both together. Under rule 1 the gap is on HT\* (+27) and on both together
(+35), and not on HS\* (+8, p = 0.17). By the plan's wording the reading is "mixed": the weights effect is larger on
the history that the candidate made, and neither history effect is shown alone [E7.2]. Under rule 2 the gap is on
both histories (+19 and +25). The plan's set for the locate step is HT\*, and its gap there is 27 misses.

**The mechanism: the model says that the tool is not there.** The updated tool list is in the prompt of every unit.
In most misses the reply, or the thinking before it, says that the tool is absent ("I still don't see a ... function
in my available toolset", "these are the same tools"). Often the reply recites the old list. Counts of such misses: 46
of 58 for KL-tools-3 on HT\*, 24 of 31 for the shipped file on HT\*, and 53 of 61 for BF16 on HS\* [E7.3]. So most
misses are not a choice of text over a call. The model keeps its earlier belief about its tools over the tool block
in front of it. Under wording (c), where the user names the new function, 50 of 50 misses of KL-tools-3 and 13 of 14
of the shipped file still say that it is missing. These are counts of fixed phrases, not a reading of each text, and
the blind recount (below) did not repeat them. A second phrase list gave 52, 24, 53, 50 and 14: the reading is the
same, and the counts move by up to 6 [E7.3]. In the other misses the model sees the new tool and asks what to do with
it.

The misses are not explained by the length of the prompt. Split at the median prompt length, the shorter and the
longer half have 31 and 27 misses (KL-tools-3, HT\*), 10 and 21 (shipped, HT\*), and 29 and 32 (BF16, HS\*): no
consistent direction. Every reply ends normally or with calls. No reply is cut at a limit, and no call fails to
decode [E7.3].

**The trigger: three wordings of the return turn.** We changed only the last user message [E7.6].

- *(a) A paraphrase:* "The list of tools is longer than it was. Please try my last request again." Under rule 1 the
  misses almost go away for both files: 4 and 1 for the shipped file (40 and 31 with the benchmark line), 8 and 5 for
  KL-tools-3. The two files are not separated (p = 0.092).
- *(b) No announcement:* the user's request of the turn before, sent again word for word. Under rule 1 the rates
  are those of the benchmark line (shipped 40 and 38; KL-tools-3 42 and 49). Under rule 2 they are high for both
  files (125 to 136 of about 200), because many replies call other functions.
- *(c) The function is named:* "I have added the function \<name\> to your tools. Use it now for my last request." The
  shipped file misses 14 and 14. KL-tools-3 misses 47 and 50. The gap grows, under both rules.

The benchmark line does not repeat the request and does not name the tool. Wording (a) asks for the last request
again, and most misses go away. So the line is a trigger, and the weights decide how often a file falls for it. Of the
three lines written first, one is met at the edge and two are not. With wording (b) the gap holds under rule 1
(p = 0.049) and not under rule 2 (p = 0.10). With wording (a) the gap is not shown. With wording (c) the miss rate is
not under 5 % for both files (7.0 % and 24.4 %), so the reading "the model can call the tool but does not notice it"
is not established as written (Table 10).

**Serving levers.** These are measurements, not product changes [E7.5].

- *Earlier thinking stripped.* A template option drops the thinking text of the earlier turns from the prompt. The
  shipped file then misses 14 and 13 in place of 40 and 31 under rule 1: 62 % fewer. Under rule 2 it misses 31 and 41
  in place of 61 and 59: 40 % fewer. For KL-tools-3 the misses fall from 106 to 39 under rule 1. Both results hold
  under both rules. With thinking stripped in both files, a small gap between them stays under rule 1 (+12,
  p = 0.012) and is not shown under rule 2 (+6, p = 0.14). We did not measure what this option costs on other turns.
- *Reasoning effort xhigh* on KL-tools-3: 67 misses against 106 (by entry -39; 17 / 45; p = 5.0e-4), which is level
  with the shipped file at the default effort (-4, p = 0.78). The result is the same under rule 2.
- *Sampling.* The benchmark is near-greedy. With the launcher's default sampler (temperature 1.0, top-k 20,
  top-p 0.95, min-p 0.05) and three seeds, the gap between the two files holds on each seed: +49, +49 and +42 misses
  under rule 1 (mean +46.7, 95 % t interval [+36.6, +56.7] over the three seeds; near-greedy +35), and +58, +51 and
  +32 under rule 2 [E7.7]. The line written first ("the gap holds when the model samples") is met.

**The parent check.** The benchmark's BF16 arm ran in another setup, so we ran BF16 and the shipped file on the same
requests on one 80 GB card, with one engine build, one template, a 16-bit K/V cache, no drafting and temperature
0.001 [E7.4]. On HS\*, BF16 misses 61 of 200 return turns and the shipped file 36 (33 / 8, p = 1.1e-4). Under rule 2
it is 77 against 59 (33 / 15, p = 0.013). The line written first ("on identical prompts the shipped map misses the
returned tool less often than the parent") is met, and the result holds under rule 2. KL-tools-3 on the same card is not separated
from BF16 (53 against 61, p = 0.26; rule 2: 81 against 77). The setup has a small effect: the shipped file misses 36
on this card and 40 in the 12 GB setup on the same requests; 10 entries differ (p = 0.34). The BF16 cell on HT\* did
not finish in its time limit (190 of 198 units) and is in no claim.

The statement is narrow. At the return turn of this benchmark, at temperature 0.001, on one history, in one setup,
BF16 misses the returned tool more often than the 2.47-bit file. We do not claim that the file is better than BF16
in general (Section 4.2 states the same limit for AppWorld).

**Locate, part 1: what the candidate maps share (a negative result).** Three candidate maps (KL-tools, KL-tools-2,
KL-tools-3) miss the return turn at about BF16's rate. Against the shipped map they share 73 changed projections: 40
raises and 33 cuts. The plan fixed a rule before any cell ran. An arm "carries" the effect if it has 60 % or more of
the gap on HT\* with p < 0.05. It is "clear" if it has 30 % or less with p >= 0.05. Otherwise it is "partial". The
shipped map with the 40 raises misses 44 against 31 (+13, p = 0.029, share 0.48: partial). The shipped map with the
33 cuts misses 36 (+5, p = 0.36, share 0.19: clear). No side carries. The plan's stop line for this case is "diffuse
at the side level", and the hunt at the level of the map stopped there [E7.8]. Five arms with one kind of projection
each had already run, and none carries: +8, +10, -3, +4 and +2 misses on HT\* (linear-attention q / k / v, MLP
gate, linear-attention output, MLP down, MLP up). The planned candidate map without the carrying kind was not built. Under
rule 2 no arm of this part separates from the shipped file.

**An observation: closer to the parent, more of this habit.** The habit is the parent's: BF16 has the highest rate of
the files that we ran in one setup (61 of 200, against 36 for the shipped file) [E7.4]. Three results point the same
way. (1) More bits move the count toward the parent. The shipped file with the 40 common raises, each from 2 to 3
bits, misses 18 turns more than the shipped file over both histories (26 / 11 by entry, p = 0.020; rule 2: +16,
p = 0.059), while the 33 cuts do not move it (+2) [E7.8]. (2) The two candidate maps with a measured KL below the
shipped file's (KL-tools, -7.9 %; KL-tools-2, -2.4 %) miss this turn in the benchmark runs in 59 of 200 and 52 of 195
entries, against 28 of 199 for the shipped file and 54 of 200 for BF16 in its own setup [E1.9, E6.14, E7.1]. (3)
KL-tools-3 is not separated from BF16 in one setup (53 against 61, p = 0.26) [E7.4]. So at this turn a file that is
closer to the parent, in bits or in KL, has more of the parent's habit. This is an observation with its source rows,
not a law. The raise arm is "partial" by the plan's rule, and the first-call decision of Inkling 2 Mini goes the
other way: there the file with the lower KL is the one that departs from its parent (below).

**Locate, part 2: two files with one bit map (suggestive only; mixed on a private set).** The second-fit file has the same bit map as the
shipped file and a second set of fits (Table 5). We use the pair because a transplant
between two files with one bit map is a byte-exact copy of tensors. It does not rank the two sets of fits: this is
one turn of one benchmark. The second-fit file misses 51 and 46 return turns against 40 and 31 (both histories by entry: +26; 47 / 25; p = 0.013). By the plan's rule this arm is
"partial" (share 0.56 on HT\*). So the low rate of the shipped file at this turn belongs to its fits as much as to
its bit map.

The two files differ in 1,653 tensors: 1,600 fit tensors (400 coded projections, each with a code, two scale vectors
and a bias that the engine does not apply) and 53 others (the embedding, the output head, 48 convolution weights of
the linear-attention layers and 3 norm weights). An amendment, written before any of its cells ran, set a *half-gap
rule*: an arm meets it if its difference to the shipped file, by entry over both histories, is 13 misses or more with
p < 0.05. An arm "carries" if it meets the rule and its complement arm does not. Table H2 gives every arm.

*Table H2. Tensor copies between the shipped file and the second-fit file, against the shipped file, by entry over
both histories (difference in misses; entries up / down; exact sign p). "Half-gap" is the label by the amendment's
rule. "Plan" is the label by the plan's rule on HT\* against the gap of KL-tools-3 (27 misses), with the share
[E7.9].*

| Arm (base file + tensors copied in) | Rule 1 | Rule 2 | Half-gap | Plan |
|------------------------------------|--------------------|--------------------|-----------|-------------|
| Shipped + the 53 other tensors | -3; 15 / 17; p = 0.86 | 0; 20 / 19; p = 1 | not met | clear |
| Second-fit + the shipped file's 53 other tensors (so: the 400 second fits) | +26; 47 / 23; p = 0.0056 | +24; 51 / 27; p = 0.0088 | met; carries | partial (0.59) |
| Shipped + the output head alone | -7; 1 / 6; p = 0.12 | -5; 1 / 5; p = 0.22 | not met | clear |
| Shipped + second fits, attention and linear-attention kinds (208 projections) | +31; 44 / 19; p = 0.0022 | +20; 40 / 26; p = 0.11 | met; carries | carries (0.67) |
| Shipped + second fits, MLP kinds (192) | +18; 36 / 21; p = 0.063 | +21; 41 / 25; p = 0.064 | not met | partial (0.33) |
| Shipped + second fits, layers 0-31 (200) | +31; 43 / 21; p = 0.0081 | +23; 43 / 28; p = 0.096 | met; carries | carries (0.85) |
| Shipped + second fits, layers 32-63 (200) | +3; 28 / 25; p = 0.78 | +5; 30 / 28; p = 0.90 | not met | clear |
| Shipped + second fits, attention q / k / v of the linear-attention layers (48) | +8; 23 / 15; p = 0.26 | +5; 28 / 20; p = 0.31 | not met | partial (0.33) |
| Shipped + second fits, linear-attention gate (48) | -2; 21 / 23; p = 0.88 | -10; 21 / 30; p = 0.26 | not met | clear |
| Shipped + second fits, linear-attention output (48) | +21; 32 / 17; p = 0.044 | +25; 40 / 20; p = 0.013 | met; carries | partial (0.44) |
| Shipped + second fits, attention q, k, v, output of the full-attention layers (64) | +4; 18 / 15; p = 0.73 | +6; 28 / 21; p = 0.39 | not met | clear |

What the table supports under rule 1: the fits carry the gap between the two files, and the head and the other
tensors do not. The arm with the 400 second fits has the whole gap (+26), and the arm with the 53 other tensors has
none (-3). Two arms with the shipped file's metadata reach the whole gap (+31 and +31), so the metadata keys of the
second-fit file are not needed. The fits of layers 0-31 carry it and those of layers 32-63 do not (direct difference
+28, p = 0.0054). The split by kind is not sharp (direct difference +13, p = 0.28), and the parts do not add: +31 and
+18 against +26 for all 400. Inside the attention kinds, the output projections of the linear-attention layers meet
the half-gap rule at the edge (p = 0.044); the three other group arms do not carry.

We call this result **suggestive only**, for three reasons.

1. *It does not hold under every rule.* Under rule 2 the gap between the two files is not shown on the plan's
   history (HT\*: 70 against 59, p = 0.15; the arm with the 400 second fits: 71 against 59, p = 0.12). Over both
   histories by entry it is there (+23, p = 0.018; +24, p = 0.0088), but the two half splits are not: the two kind
   halves are level (+20 and +21), and the layer half is at p = 0.096.
2. *The labels depend on the rule.* By the amendment's half-gap rule the arm with the 400 second fits and the
   output-projection arm are "met". By the plan's own rule both are "partial" (shares 0.59 and 0.44). We give both
   labels. The plan's rule was written for the transplants of part 1; its use here is an extension by the recount.
3. *The test set is the set that gave the lead.* Every arm ran on the same public entries on which we found the
   gap. The reference gap is small (47 entries up, 25 down), so an arm with half of it often cannot reach p < 0.05.
   Six primary and six secondary tests ran, with no correction for their number; by chance alone about 0.3 of six
   tests show p < 0.05.

**A fresh private set.** To test the result on entries that no step had used, we generated a private set of return
turns. It has the structure of BFCL miss_func and no overlap with the public entries, and it is split into a
development half and a sealed half. The plan and its lines were committed before any model row. On the development
half we made two histories in the same way (one by the shipped file, one by KL-tools-3) and replayed the return turn.
The primary set is the 115 entries with a vague user line, on both histories: 230 units. Every count below is against
the shipped file on those units, by entry, under rule 1 and then under rule 2 [E7.15].

- *The control holds.* KL-tools-3 misses 83 units against 62 (+21; 27 / 10; p = 0.0076). Under rule 2 it is 89
  against 71 (+18; 25 / 10; p = 0.017). So the instrument sees the gap of the candidate map on fresh entries.
- *The gap between the two same-map files is not shown.* The second-fit file misses 74 against 62 (+12; 16 / 8;
  p = 0.15). Under rule 2 it is 83 against 71 (+12; 17 / 8; p = 0.11). The line written first asked for a gap above 4
  misses with p < 0.05 under both rules. It is not met. The direction is that of the public entries, and the rate is
  a little lower (5.2 against 6.5 misses for each 100 units). The plan had stated the limit before the run: a gap of
  the public size is about 15 misses on this set, with p near 0.05 at best. So this reads "not shown at this size",
  not "absent". By the plan's rule, the lines on the fit copies and on the repair arms were then not read. The rest
  of this list is a report.
- *One of the public leads shows again, and the others do not.* The shipped file with the second fits of the
  attention and linear-attention kinds (the kind half that led on the public entries, Table H2) misses 84 against 62
  (+22; 24 / 8; p = 0.0070). Under rule 2 it is 90 against 71 (+19; 22 / 9; p = 0.029). The other leads of Table H2 do
  not show again: the arm with all 400 second fits, the arm with the fits of layers 0-31 and the arm with the
  linear-attention output fits are at +10, -1 and +1 misses (their counts are in the evidence row).
- *A lead in the other direction, chosen after the look.* Five graft arms put the shipped file's fits into the
  second-fit file. One of them has the shipped fits of the attention and linear-attention kinds: it is the reverse
  of the copy above. It misses 51 units against 62 for the shipped file (-11; 7 / 18; p = 0.043). Under rule 2 it is
  59 against 71 (-12; 6 / 19; p = 0.015). Against its own base, the second-fit file, it misses 23 units fewer (4 /
  23; p = 3.1e-4), and 24 fewer under rule 2. We did not predict a count below the shipped file. We chose this arm
  after we looked at the development half: it is 1 of 5 graft arms, and their lines were not read. A guard read of
  the first turn of every entry found a valid tool call for every arm (150 of 150).
- *Where these two leads sit.* The primary set has two vague user lines: the benchmark's line and a second wording.
  Both leads sit mostly under the second wording (+19 of the +22, and -10 of the -11). There the control gap is small
  (+2). Under the benchmark's line the control gap is +19, and the two arms are near the shipped file (+3 and -1).
- *With the function named* (35 entries, 70 units), the shipped file, the second-fit file and the two arms almost
  stop missing (2, 1, 1 and 3 misses). KL-tools-3 misses 9.

**One read on the sealed half.** The graft arm of the list above then got one read on the sealed half. New lines
were written before the read [E7.16]. *C, the control:* KL-tools-3 misses more than the shipped file, with p < 0.05
under rule 1; if C fails, the other lines are not read. *S-1:* the graft misses less than its base, the second-fit
file, with p < 0.05 under both rules, and with 75 % of the base's misses or fewer under both rules. *S-2:* the graft
misses less than the shipped file, with p < 0.05 under both rules. *S-3:* the reverse copy misses more than the
shipped file, with p < 0.05 under both rules. The histories of the sealed half were made in the same way. Every count
is by entry on the 230 units of its primary set, under rule 1 and then under rule 2.

- *C, the control, fails.* KL-tools-3 misses 78 units against 62 (+16; 24 / 12; p = 0.065). Under rule 2 it is 89
  against 75 (+14; 22 / 13; p = 0.18). So by the rule written before the read, no repair line is read. The counts
  that follow are a report.
- *S-1, the graft against its base.* 55 misses against 79 (-24; 7 / 29; p = 3.1e-4; ratio 0.70). Under rule 2 it is
  65 against 81 (-16; 9 / 24; p = 0.014; ratio 0.80, which is above the 0.75 of the line).
- *S-2, the graft against the shipped file.* 55 against 62 (-7; 10 / 15; p = 0.42). Under rule 2 it is 65 against 75
  (-10; 9 / 16; p = 0.23).
- *S-3, the reverse copy against the shipped file.* 75 against 62 (+13; 19 / 9; p = 0.087). Under rule 2 it is 81
  against 75 (+6; 15 / 12; p = 0.70).

The lead of the development half did not hold. The control did not reach its line, so the lines are not read. With
the control, the reading rule gives the same words: the graft is not below the shipped file (S-2 fails under both
rules), and its cut against its own base misses the 75 % part of S-1 under rule 2. One thing repeats on unread data,
as a report: the graft misses less often than its own base (55 against 79 under rule 1), mostly under the second
vague wording (-15 of the -24). It brings the second-fit file to about the level of the shipped file, and not below
it. We chose this arm after we looked at the development half, as 1 of 5 arms. The sealed read is the test that such
a choice needs, and it does not confirm a repair. On the sealed half the gap between the two same-map files shows
under rule 1 (+17; 26 / 12; p = 0.034) and not under rule 2 (+6; p = 0.49). A guard read of the first turn found a
valid tool call for every arm (150 of 150). No repair is confirmed on held-out turns.

One fact from the public entries agrees with the last point: with wording (c) the second-fit file is close to the
shipped file (18 and 21 misses against
14 and 14; p = 0.14), and KL-tools-3 is not (47 and 50). So the gap between the two same-map files and the gap of the
candidate maps are not the same thing.

**A blind recount.** A separate agent recounted both studies of this section from the raw rows. Before its first
commit it read only the plans (for the rules) and the tools that wrote the rows (for the format). It did not read the
entries, the reports or any result file. Every count of a complete cell matches: the misses, the paired counts, the
differences and the p values [E7.14]. It also applied the plan's rule to the arms of Table H2, which gave the second
set of labels. The scripts of the evidence repository repeat the counts of Tables H1 to H3 from per-unit outcome
files.

![The return turn of BFCL miss_func on the 27B model: share of units with a miss, for pairs of cells on the same
requests. (a) Rule 1: no tool call. (b) Rule 2: the returned function is not called. Pairs: KL-tools-3 against the
shipped file on each history; BF16 against the shipped file on one 80 GB card in one setup; the shipped file with and
without the thinking of earlier turns (both histories); the two files under wording (c) (both histories). Numbers on
the bars are misses [E7.2, E7.4-E7.6].](figs/fig8-return-turn.png){width=100%}

**A second family: Inkling 2 Mini.** We asked two questions. Does a turn-type split find a class that holds a score
gap, as on the 27B model? If not, does it find any decision where the files differ?

*The turn split found no class that holds the gap (a negative result).* We classed every failed run of the recorded
AppWorld runs of the standard split (168 tasks, seeds 102-105; Figure 7a): the official weights (run as Unsloth's
UD-Q8_K_XL file, an exact copy of the release [E4.10]), the public 2-bit file (UD-Q2_K_XL) in its own build, and M2. The
line was written first: "one class" if the top class has half of the gap or more and the sign test by task has
p < 0.05; "no gap to split" under 10 task-runs. The official weights and M2 have no gap (592 and 596 of 672). For the two
pairs with a gap (30 and 26 task-runs), the top class holds 0.30 and 0.31 of it (p = 0.12 and 0.096): "diffuse"
[E7.10]. The unasked answer of Section 4.3 is rare here (18 to 26 failed runs for each arm) and does not differ by
arm. So the pattern of the 27B model, one class that holds the gap, did not repeat.

*A decision found after the fact.* The same read showed a decision that the totals hide. At the first call of a
task, the reply that the client receives is often one sentence ("I'll start by exploring the available apps.") with
no code block. The client answers "No code available to execute." and the run goes on. (This is what the client
receives. What the model samples there is more than the sentence: see "What the client does not see", below.) In the record, the first reply
has no code block in 227 of 672 runs for the official weights (33.8 %), in 254 of 670 for the public 2-bit file in its own
build (37.9 %), and in 50 of 672 for M2 (7.4 %) [E7.10]. This was not a line written first. So we ran a confirm.

*A pre-registered confirm on fresh tasks.* The first call has no model history: the request is the fixed agent prompt
and the task. So a replay of the built request is the whole state of that turn. The requests that our driver builds
equal the recorded first requests on 2,014 of 2,014 runs [E7.11]. The fresh set is the harder split (417 tasks), on
which no file had been read for this decision. The plan, with three lines, was committed before any call. M2 and the
public 2-bit file ran in one engine build and one setup on one 12 GB card, on the same 834 requests (seeds 102 and
103), with an output cap of 1,024 tokens. A reply that the cap cut counts against the claim in L1 and L2.

- **L1, the decision shows on fresh tasks:** the public 2-bit file's rate is 20 % or more. Result: 350 of 834
  (42.0 %). Pass.
- **L2, M2 is below it:** a sign test by task with p < 0.001, and a rate of half or less. Result: 95 of 834 (11.4 %)
  against 42.0 %, ratio 0.27; by task 223 / 22 (p = 4.8e-43); by scenario 105 / 5 of 139. Pass.
- **L3, the replay repeats the record:** M2 on the standard split, seed 102, inside the 95 % interval of the record's
  rate ([6, 20] of 168). Result: 18 of 168. Pass. On our workstation the replay gives the record's reply text on 163
  of 168 tasks.

After the lines were scored, we added the official weights as a report row (an amendment written before its first
call): the same engine commit, the same 834 requests, a 16-bit K/V cache, on a rented 48 GB card, because the file
does not fit 12 GB [E7.11]. Table H3 gives the three rates for each seed.

*Table H3. Inkling 2 Mini, the first call of an AppWorld task on the harder split: first replies with no code block,
of 417 for each seed (a cut reply is scored by its content). The official weights ran as Unsloth's UD-Q8_K_XL
file, an exact copy of the release; this model has no BF16 release [E4.10, E7.11].*

| Arm | Seed 102 | Seed 103 | All seeds |
|------------------------------|-----------:|-----------:|------------------:|
| Official weights (run as UD-Q8_K_XL), 48 GB card | 261 (62.6 %) | 125 (30.0 %) | 386 of 834 (46.3 %) |
| Public 2-bit file (UD-Q2_K_XL), our engine, 12 GB card | 192 (46.0 %) | 184 (44.1 %) | 376 of 834 (45.1 %) |
| M2 (ours), 12 GB card | 64 (15.3 %) | 31 (7.4 %) | 95 of 834 (11.4 %) |

The public 2-bit file is at the rate of the official weights (by task 108 / 112, p = 0.84): a standard quant keeps this decision
of its parent. M2 departs from its parent at this decision (by task 246 / 13). Average KL does not show the
departure, and it points the other way: M2 has the lower KL of the two 12 GB files (Table 7).

![Inkling 2 Mini, the first call of an AppWorld task on the harder split: share of first replies with no code block,
for each seed and for all seeds, for the official weights (run as UD-Q8_K_XL, an exact copy of the release), the public 2-bit file and M2. The
calls of one seed share one random stream (Section 4.1) [E7.11].](figs/fig9-inkling-first-call.png){width=85%}

*Two controls.* The official weights ran with a 16-bit K/V cache, and M2 with its release K/V setting. M2 with a 16-bit K/V
cache gave 11 of 168 against 18 of 168 on the same requests (by task 6 / 13, p = 0.17). M2 with no reasoning budget
gave 18 of 168, with the same outcome on every task. So the K/V precision and the reasoning budget do not explain the
difference between the official weights and M2 [E7.11].

*A decision probe that did not locate the decision (a negative result).* We pre-filled the reply with the common
first sentence and read the next-token probabilities, with no sampling, for M2 and for the public 2-bit file (417
tasks each; the form was written before the probe ran). The template drops an empty thinking block, so the probe read
a state with no thinking, which a sampled reply at this effort never has. In that state both files go on to code: the
most probable next token is a line break on 417 of 417 tasks in both. M2 gives the end-of-turn token *more*
probability than the public file (median 0.120 against 0.046; higher on 417 of 417 tasks). That is the opposite of
the sampled rates [E7.12]. So this decision is not made at the token after the sentence in a state with no thinking.
It is not the choice of the sentence either, as the next probe shows.

*A cross-over of the thinking: the step after the thinking decides.* Every sampled reply has a thinking block first.
Does the thinking carry the difference between the two files, or the step after it? We took each file's own sampled
thinking from the confirm (1,668 states: the 834 requests, for two files). Then each file's weights wrote the reply
text once after each thinking, with the confirm's sampler. The plan and its lines were committed before any row
[E7.18].

*Table H3b. The cross-over: continuations with no code fence, of the scored states. A continuation that reached the
limit of 200 tokens is left out (7, 13, 78 and 91 states) [E7.18].*

| Weights that write the reply | After M2's thinking | After the public 2-bit file's thinking |
|------------------------------|--------------------:|--------------------:|
| M2 | 111 of 827 (13.4 %) | 77 of 821 (9.4 %) |
| Public 2-bit file | 395 of 756 (52.2 %) | 331 of 743 (44.5 %) |

The validity line holds: each file on its own thinking is inside the 95 % interval of its rate in the confirm (11.4 %
and 45.1 %). With the thinking fixed, the public file's weights give more replies with no code: +38.8 points after
M2's thinking (pairs 312 / 29) and +35.2 points after its own thinking (272 / 18). With the weights fixed, the
public file's thinking gives fewer such replies, not more: -4.0 points under M2's weights (49 / 85; p = 0.0024) and
-7.7 points under its own weights (104 / 165; p = 2.4e-4). By the lines written first, the step after the thinking
decides, and the thinking does not carry the difference.

*What the client does not see (found after the fact).* A token count in the cross-over showed that "one sentence and
no code" is what the client receives. It is not all that the model samples. In most replies with no code, the model
ends its text block after the sentence and then writes the action in a second block: bare code with no fence, or a
tool-call form. In the cross-over, 690 of the 726 continuations with no fence under the public file's weights hold
such code after the sentence, and 158 of the 188 under M2's weights. In the confirm, 373 of the public file's 376
replies with no code, and 87 of M2's 95, have more sampled tokens than the returned thinking and text hold (a median
of about 26 to 28 tokens) [E7.19]. The server's chat parser returns only the first text block, because the request
declares no tools. So the decision is a choice of channel: fenced code in the text block, as the agent prompt shows
it, against a second block that this server drops. Three limits apply. The reading is post hoc. The counts of
"holds code" are counts of patterns in the text. And the exact type of the second block needs the token ids: a check
is queued, and it is not in this version. The consequence is plain. A part of this habit may be removable on the
serving side, by a return of the dropped block, with no change of the weights. That is a separate test, and it has
not been run. It is the lesson of Section 4.8 again: a choice outside the weights can look like a property of the
weights. The lines of the confirm stand as scored, because they count what the client receives.

*Does the decision cost the task? Not in this client.* A habit read has two parts. Part 1: is the decision
different? Part 2: does it cost the task? We tested part 2 on the recorded full runs of the standard split, with the
reads and a rule written before the count (a post hoc check on existing runs; no line of the confirm depends on it)
[E7.13].

- *The raw split is a task effect.* Runs whose first reply has no code block pass less often than the other runs of
  the same arm: 83.7 % against 90.3 % (official weights), 79.1 % against 87.3 % (public 2-bit file in its own build), 86.0 %
  against 88.9 % (M2). But the tasks differ between the two groups: the decision shows more on harder tasks. This
  split is not a cost.
- *With the task held fixed, no cost is shown.* For each task that has both kinds of run on the seeds of one arm, we
  compared the pass rates of the two kinds. Over 273 arm-task units the runs with no code first pass 1.5 points
  *more* often (95 % bootstrap interval [-1.8, +4.7]; 33 units better, 18 worse, 222 equal; sign p = 0.049). The
  rule written before the count gives this case the label "a small cost", because its pooled p is under 0.05. The
  rule did not foresee the direction that came, so the label does not fit. The plain reading is: no task cost shown.
  A cost under about 2 points is not excluded.
- *The cost is about one call.* The run gives code at the next call in 89.0 %, 75.2 % and 96.0 % of the cases
  (official weights, public 2-bit file, M2), and uses about one call more (mean +0.68, +1.61 and +1.62).
- *The gap between the official weights and M2 is not at this decision.* M2 passes 4 runs more than the official
  weights (596 against 592). All 4 are in the 426 task-runs where both files give code first. In the 246 task-runs where one file
  or both give no code first, the two are level (205 and 205).

So on Inkling 2 Mini this habit is a *decision marker*. The method detects it, replays it and confirms it across
builds. It is not damage, and the lower rate of M2 is not a quality gain. What we state is this: M2 departs from its
parent at this decision, a standard 2-bit quant stays at the parent's rate, and the departure has no measured task
effect in this client. A client that ends the turn on a reply with no tool call would lose the turn; that case is
not measured.

The contrast with the 27B model is the point. There, the miss at the return turn *is* the scored failure of that
turn, and the unasked answer *is* the failed test of the run. Here, part 2 is null. It turns the older point of this
paper around: a change in behaviour can be large (46 % against 11 % of the first replies) and still be invisible in
the task total.

*Limits of the confirm.* (1) Two seeds are two clusters (Section 4.1). Seed 102 has about twice the events of seed
103 for the official weights and for M2 (261 against 125; 64 against 31); for the public 2-bit file the two seeds are close
(192 against 184). (2) M2's rate is higher on the harder split (11.4 %) than in the record of the standard split
(7.4 %); on the one seed that both have, it is 15.3 % against 10.7 %. (3) The decision was found after the fact on
old transcripts, and then confirmed on fresh tasks with the lines written first. (4) The official weights ran on
another card class. Their replies are not token-identical to the record there (the same text on 21 of 168 tasks of the
standard split; the rate holds: 74 against 80 of 168), while the replay on our workstation is (163 of 168). The
parent's rate on the harder split is therefore from another card than the two 12 GB rows. (5) In the record the
public 2-bit file ran in its own build; in the confirm it ran in our engine and setup, so the confirm compares
weights in one setup. (6) The cap of 1,024 tokens cut 5, 26 and 9 replies (M2, public file, official weights); the lines
count them against the claim.

**What this section shows, and what it does not.** Table H4 puts the three habits side by side.

*Table H4. Three decision-point habits. Rates for the full-precision weights (Section 2), a standard quant and our build
[E6.4, E7.4, E7.11, E7.13].*

| Habit | Model, turn | Full-precision weights | Standard quant | Our build | Does it cost the task? |
|--------------|------------------------|----------------|--------------|--------------|------------------|
| Unasked answer (Section 4.3) | 27B, the last call of an AppWorld action task, temperature 1.0 | BF16: 258 of 1,668 runs | not run | shipped: 112 of 1,668 | yes: it is the failed test of the run |
| Returned tool not called | 27B, the BFCL return turn, temperature 0.001, one setup | BF16: 61 of 200 | not run | shipped: 36 of 200 | yes: it is the scored failure of the turn |
| First reply with no code, as the client receives it | Inkling 2 Mini, the first call of an AppWorld task, temperature 1.0 | official weights: 386 of 834 | public 2-bit file: 376 of 834 | M2: 95 of 834 | none shown in this client; about one call |

On two model families, a build of ours differs from its parent at one decision: it shows the parent's habit less
often. **Neither build was made to do that.** The shipped 27B map is the published L0xRE-V2 allocation with our fits, and our own candidate
maps are at BF16's rate at the return turn. M2 was chosen on KL. So these are observations of what our builds do, not
results of a repair. The steps that this paper supports are: detect a habit that a total hides; freeze it and replay
it for the cost of minutes; test levers and controls on the frozen turn; and, on the public entries only and with the limits
above, point at the tensors that carry a difference between two files. **The missing step is a map built to remove a habit, and then
confirmed on held-out turns.** We have not done it. The hunt for such a map on the 27B model ended at its own stop
line (part 1 of the locate step). The result of part 2 is suggestive only: on the development half of a private set
the gap between the two files is not shown, and one of its four leads shows again. A graft that we chose on that
half was not confirmed on the sealed half.

## 5. Negative results (repairs measured and rejected)

All lines below were written before the run. Rejected items stay in the record. The hypotheses of Sections 4.2 and
4.4 that failed are in Table 10.

*Table 9. Repairs that we measured and rejected.*

| Repair | Idea | Measured | Decision | Ref. |
|---|---|---|---|---|
| M2 with a 4-bit head | Q4_0 output head, saves 232 MB | +6.9 % KL vs M2 (diff +0.0021 [+0.0019, +0.0024]) | option note only | [E2.8] |
| Copy-aware map | bits to layers that help copy rows | +10.3 % KL on scored rows; copy rows only -2.4 % | not shipped | [E2.9] |
| Borrowed allocation | the byte allocation of the old LookAtMyQuant file on our fits | +43 % KL (multilingual +153 %) | not shipped | [E2.10] |
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

- **One benchmark, one sampler chain, and the temperature.** The lead over BF16 and the map ordering of Section 4.4
  are from AppWorld test_challenge at temperature 1.0, with top-k 20, top-p 0.95 and min-p 0.05. On the standard
  split the same files are level. At temperature 0 the lead over BF16 is +11 [-5, +28] of 417, and one greedy run
  cannot show whether it is smaller than the lead at 1.0 [E5.18, E6.1]. Three maps of Section 4.4 have one
  temperature-0 run each [E6.7]. The temperature-0 control is one run of 417 tasks for each arm. A second agent
  benchmark (BFCL v3 multi-turn) is used for the screen of the held candidate [E6.14] and for the return-turn study
  of Section 4.10.
- **Few seeds.** Most comparisons on the harder split have two seeds. The seed-level p uses a design effect that is a
  mean over arm pairs with a wide range, so it is approximate; a strict test across two seeds has almost no power.
  Three effects sit in one seed and are marked in the text: the losses of the uniform and the HAWQ-style maps (seed
  107), and the gain of M2 over the official Inkling weights on the harder split (seed 113) [E5.1, E5.13, E5.23].
- **Teacher-forced instruments cannot see free-generation behaviour.** KL, the recall margins and the held-out recall
  test (B1) score tokens that we force. They cannot see what a model does when it writes freely. Two observations
  show it. (a) On the 24 needle questions at 245k, in free generation, the shipped file thinks for 197-209
  characters on every item. KL-tools does the same on 23 items (207-211). On one item it thinks for 1,184
  characters: it lists all 24 codes and then gives the correct answer. The text is the same on two operating
  systems, with or without drafting, and in any question order, so it is in the weights [E5.26]. In free
  generation BF16 is short too: 199-232 characters on every item, 221 on that item, no other code named, all 24
  answers correct [E5.30]. KL, the margins and B1 did not see the difference: KL-tools has lower KL than the shipped
  file, full recall counts and no held-out margin loss. (b) An independent port of Inkling 2 Mini
  to another runtime reported a 2-bit map that failed a 245k needle by a copy loop in free generation, while the
  teacher-forced margin of that item was not low [E5.27]. That report is not our run.
- **A line on a single item is fragile.** We wrote a line before the free-generation run: no item with more than
  1,000 characters of thinking. KL-tools missed it by one item. Three results make this line a poor instrument, and
  we report them as findings about the instrument. First, when one group of KL-tools went back to the shipped bits
  (the attention projections, or the output projections, of the linear-attention layers), the long thinking on that
  item went away, and 3 *other* items of the 24 got it; every answer stayed correct [E5.26]. So a map tuned until the
  one item passes moves the event; we tuned no map. Second, the limit is not a validated threshold: the independent
  port reported that the official Inkling weights exceed 1,000 characters on 3 of 20 held-out items [E5.27]. Third,
  and most important, the line misread the event. We forced the three replies to that item (the shipped file's, the
  KL-tools reply and BF16's own) through BF16 and read BF16's probability of each token [E5.31]. Both quantized files
  leave BF16's path at the same two early tokens, before any code is written: token 22 ("it" where BF16 prefers
  "this"; BF16 gap 1.41 nats) and token 34, where both start a list on a new line and BF16 prefers to quote the line
  inline (gap 2.95). After that shared list prefix, BF16 itself prefers to list the codes: at token 43 it puts 0.94
  of its probability on the first code of the list (the KL-tools token) and gives the vault 9 code, which the shipped
  file writes, a log probability of -2.75 (gap 2.68 nats). On the whole 1,184-character KL-tools reply, BF16's top
  token equals the forced token on 478 of 480 tokens. BF16 is short in free generation only because it never takes
  the list form. So the long reply is not a further departure of KL-tools from the reference; on this prefix the
  shipped file's short answer is itself a departure (at token 43). The two early departures are damage that both
  files share; they are not damage in the new map. The length fact stays true; the reading that the length showed
  damage in the new map does not. This is the thesis of the paper in a small case: a line on an output
  statistic (here, length) misread a departure that both files share, and only a reading against the reference at
  each token separated the two. The decision stays: the shipped build keeps the published allocation, because
  KL-tools failed a line written before the run and we do not ship a miss, not because damage was shown. Limits: BF16
  ran with a 16-bit K/V cache, an exact head and no drafting (KL-tools gives the same text with and without
  drafting); the forced reading is one item and one run.
- **Two meanings of "fragile".** For teacher-forced items we call an item fragile when the *shipped map's* own
  minimum margin on it is below 1 nat (the code-copy item of Table 3: 0.43, and 0.26 in the graded probe). A second
  definition asks if the *reference weights* have a small gap between their two best tokens at the place where two
  maps go different ways. For the free-generation item above it is measured: the place where the two files go
  different ways is clear in BF16 (gap 2.68 nats), and the earlier shared departures are clear too (1.41 and 2.95)
  [E5.31]. Verdicts on fragile teacher-forced items in this paper use the first definition.
- **The limits of the mechanism.** Section 4.3 puts the lead over BF16 and the map order at one decision, the
  unasked answer at the end of an action task. Its classes were added after the first reading (post hoc); the maps
  have two seeds; the counts have no p value, because the task-runs of a seed are one cluster; and "pass + unasked
  only" is not a score. We do not know why a map changes the rate of the habit: no single group of layers explains
  why the lower-KL maps give more unasked answers. The xhigh result is one seed on a selected set in a later
  launcher. The sharpness and token-class tests use golden-set text, not AppWorld traffic, and drafting exactness is
  from code reading [E5.7, E5.8, E5.16, E6.4, E6.9]. One benchmark and one prompt show the habit; on another
  prompt its rate can differ.
- **Decision-point habits: what the word covers.** "Habit" means what these benchmarks punish or count. It is not
  a general statement of quality. A habit read has two parts: is the decision different, and does it cost the task.
  On the 27B model the miss is the scored failure of the turn. On Inkling 2 Mini the second part is null: the
  first-call decision is a marker of a departure from the parent, not damage, and its lower rate in our build is not
  a quality gain (Section 4.10) [E7.13].
- **No build was made to remove a habit.** The shipped 27B map is the published allocation, and M2 was chosen on KL.
  That our builds show a parent habit less often is an observation. The repair step of Section 4.10 is open: a first
  graft, chosen after a look, was not confirmed on held-out turns.
- **The return turn (27B).** The turn was found after the fact, as 1 of about 10 turn classes of one benchmark; the
  replay is its confirmation [E7.1, E7.2]. Under the plan's miss rule the gap of the candidate map is on one of the
  two histories and on both together, not on each ("mixed"). Rule 2 was added after the first results and is
  exploratory. The counts of what the misses say are counts of fixed phrases; a second phrase list moves them by up
  to 6 [E7.3]. The levers (wording, stripped thinking, effort, sampler) were measured at this turn only; none is a
  product change, and what stripped thinking costs elsewhere is not measured.
- **The parent check is narrow.** BF16 misses the returned tool more often than the shipped file on one history, on
  one card, in one setup, at temperature 0.001 [E7.4]. The benchmark's own BF16 arm ran in another setup (engine
  build, K/V type, card). The second history has no complete BF16 cell. The AppWorld lead over BF16 is at temperature
  1.0; at temperature 0 it is not shown (Section 4.2). We make no statement about BF16 without the temperature and
  the benchmark turn.
- **The locate step is suggestive only, and a private set gives a mixed answer.** On the development half of a
  fresh private set of our own, the control gap of the candidate map shows, and the gap between the two same-map
  files does not: +12 misses of 230 units (p = 0.15; rule 2: p = 0.11). The set could show a gap of the public size
  only at the edge, so this reads "not shown", not "absent". By the plan's rule the lines on the fit copies and on
  the graft arms were then not read. As a report, one of the four public leads shows again (the copy of the
  attention and linear-attention fits) and three do not; and one graft arm, chosen after the look, is below both
  files. These two leads sit mostly under one of the set's two vague user lines [E7.15]. On the sealed half, in one
  read, the control did not reach its line, so the graft's lines were not read. As a report, the graft misses less
  often than its own base there and is not below the shipped file: no repair is confirmed [E7.16]. On the
  public entries, the fit result of Section 4.10 holds under rule 1 and, over both histories,
  under rule 2, but not under rule 2 on the plan's history, and its half splits do not hold under rule 2. Its labels
  differ between the amendment's rule and the plan's rule. It was tested on the public entries that gave the lead,
  with a reference gap of 47 entries up and 25 down, and with six primary and six secondary tests without a
  correction [E7.9]. A habit is located in one model family only, and only on the public entries. On Inkling 2 Mini the decision is detected and
  confirmed. A cross-over places it at the step after the thinking, and no group of weights is located [E7.12,
  E7.18].
- **What the first-call decision is.** The confirm counts what the client receives. A post hoc token check shows
  that in most replies with no code the model writes the action in a second block, which this server drops because
  the request declares no tools. So the decision is a choice of channel, and a part of it may be removable on the
  serving side with no change of the weights. That test has not been run, and the type of the second block is not
  read yet [E7.19].
- **The Inkling confirm.** Two seeds are two clusters, and one seed has about twice the events of the other for the
  official weights and for M2. M2's rate is higher on the harder split than in the record of the standard split. The
  decision was found after the fact on old transcripts and then confirmed on fresh tasks with the lines written
  first. The parent is the official weights (this model has no BF16 release); they ran on another card class, where their
  replies are not token-identical to the record (21 of 168), while the replay on our workstation is (163 of 168)
  [E7.11]. The task-cost check is post hoc and observational, on the standard split, in one client that asks again
  after a reply with no code; a cost under about 2 points is not excluded, and a client that ends the turn is not
  measured [E7.13].
- **Context length.** Every habit replay of Section 4.10 is at short context. The return-turn units of the public
  entries have 3,111 to 48,571 prompt tokens, with a median of 5,742; 3 of the 398 units are above 16,384 tokens. The
  units of the private set's development half have 1,718 to 13,123 prompt tokens, with a median of 4,119. The first-call requests of
  Inkling 2 Mini have 3,969 to 4,153 prompt tokens, with a median of about 4,000 [E7.17]. So the habit rates at long
  context are not measured in this version. At tens to hundreds of thousands of tokens the tool block is far from
  the turn, and the rates can differ. A depth test of the return turn is running, and it has no result here. The
  long-context evidence of this paper (recall at 117k and 245k tokens [E4.3], and the long-copy gates) is about
  retrieval and copying. It is not about these decisions.
- **Another lineage.** The allocation rule of the 27B model did not carry to Inkling 2 Mini, a model of another
  lineage and architecture (Section 4.7). What carries within one lineage is knowledge of the base (Section 4.9),
  read from one base model here.
- **Coding agents (a description of one run each, not a test).** A public third-party board has 20 repository-repair
  tasks [PatchEval]; each setup has one run with the Pi agent. On release v1.1.1 the shipped file solved 13 of 20. We
  read each of the 7 misses of that run against the hidden test and the upstream fix: 6 were the model's own patches,
  1 was a valid fix that an exact token snapshot rejected, and 0 came from the setup [E6.16]. On release v1.2, with
  the same model file, the board's run solved 18 of 20, and 17 at the first try [E6.16b]. Between the two runs the
  serving stack changed and the weights did not. One change is in release v1.1.6, and the board's note names it:
  with thinking on, the server now keeps a token cap of 4,096 or more that the agent sends, and it ends the thinking
  early enough to leave room for the answer (the reasoning budget is the cap minus 1,024). Before that release it
  raised every cap to 24,576. The other releases between the two runs changed the stack too, and we did not measure
  which change caused the rise. It is one more case where a task run shows an effect from outside the weights
  (Section 4.8). The official API run of the base model
  solved 16 of 20 on the same tasks. Paired by task, the v1.2 run solves 3 tasks that the API run does not, and the
  API run solves 1 that the v1.2 run does not (exact sign test p = 0.63) [E6.16b]. This is not a test of the weights:
  each setup has one run, and the two runs use different serving stacks.
- **An error in our record.** Our record said for some hours that KL-table B had one cut against the shipped map. The
  model files show two (Section 4.4). We now read bit maps from the model files, not from map tables [E5.19].
- **One fragile item.** Three recall-gate failures (KL-table A and the uniform and HAWQ-style baselines) are on that
  one code copy; in the baselines the body was exact
  and the comment line above the function was copied too. The gate counted all three, as written, and the text treats
  all three the same: a failed gate, weak evidence. Graded margins support that reading for the baselines: on the
  other 43 items they have more margin than the shipped file [E1.33]; KL-table A has no graded run.
- **Selection inside the study.** The fit library, the probes and the named documents were built and chosen on the
  same golden-set half (p1). The untouched half and fresh seeds limit, but do not remove, this.
- **Confounds in the reference rows.** On the standard split BF16 ran on 80 GB GPUs and the shipped build on 12 GB
  GPUs. On the harder split the run of the shipped file in BF16's setup removes that difference for two seeds
  [E5.5]. The Inkling official-weights arms differ from ours in the weights, the K/V precision and the card class;
  they have no weights-only arm [E4.8, E5.23].
- **Setup against setup.** The Inkling rows of Section 4.6 compare two shipped products. The comparison with the
  setup equal is Section 4.7. Two Unsloth task-runs on seed 105 did not finish (bound given).
- **The golden set and a third party's calibration text.** Five of our 298 Inkling golden documents overlap the
  calibration text of the LookAtMyQuant files; we give the comparison with and without them [E5.25]. We measured only
  the old file of that project; its newer file is not yet measured on our golden set [E5.24].
- **Copy-row KL** saturates and its meaning is unclear (Section 4.7).
- **Runtime repair on tasks.** The per-class K/V choice changed KL and memory but not pass / fail recall (blanket
  arms were full at every depth) and not graded recall margins at a resolution of about 0.2 nats. On AppWorld only
  the global cache type was varied on fixed weights (4-bit 595 vs 8-bit 596 of 672 [E4.12]); the blanket settings
  have no agent score. The head repair has a KL gain and no shown task effect [E5.17].
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
  near-tie token is set aside, and 24/24 needles at 32k [E4.13]. An RTX 4080 gave other tokens than an RTX 4070 on 19
  of 50 greedy identity prompts with the same file, so the 12 GB arms of the 27B model in the second batch ran on RTX
  4070 cards only [E5.29].
- **Every missed line.** Table 10 lists every pre-written line that a run missed and what we did.

*Table 10. Every pre-written line that a run missed, and what we did. Lines that rejected a candidate map or repair
are in Table 3 and Section 5. The first block is new in version 1.3, the second in version 1.2, the third in
version 1.1.*

| Line (written before the run) | Measured | What we did | Ref. |
|---|---|---|---|
| Private set, development half: the gap between the two same-map files shows on fresh entries (above 4 misses, p < 0.05, under both miss rules) | +12 of 230 units (p 0.15); rule 2: +12 (p 0.11) | not shown at this size; by the plan the lines on the fit copies and on the graft arms were not read (their counts are a report); the control line of the set holds (+21, p 0.0076) | [E7.15] |
| Private set, sealed half, one read: the control (KL-tools-3 above shipped, p < 0.05 under rule 1); if it holds, the graft is below its base (S-1, with 75 % of the base's misses or fewer) and below shipped (S-2), and the reverse copy is above shipped (S-3), each under both miss rules | control +16 of 230 units, p 0.065. Report: S-1 -24 (p 3.1e-4, ratio 0.70), rule 2 -16 (p 0.014, ratio 0.80); S-2 -7 (p 0.42), rule 2 -10 (p 0.23); S-3 +13 (p 0.087), rule 2 +6 (p 0.70) | the control fails, so the lines are not read; by the counts S-2 fails under both rules and S-1 fails its 75 % part under rule 2: the development lead did not hold; no repair is confirmed | [E7.16] |
| Return-turn replay: the gap of the candidate map shows on each history (10 misses or more, p < 0.05) | +8 (p 0.17) on one history, +27 on the other; +35 on both together | valid by the line for both together; the reading is "mixed"; the locate step uses the one history | [E7.2] |
| A replay of a file on its own history equals the live call on 90 % or more of the units (an expectation) | 95.0 % and 89.4 % (the replays ran on another machine) | reported | [E7.2] |
| Return-turn hunt: the 40 common raises or the 33 common cuts of the candidate maps carry 60 % or more of the gap (p < 0.05) | raises 0.48 (partial); cuts 0.19 (clear) | stop line "diffuse at the side level"; the hunt at the level of the map stopped; no candidate map built | [E7.8] |
| Wording (a): the gap between the two files holds (above 0 on each history, p < 0.05) | +4 and +4; p 0.092 | not shown; the misses almost go away for both files | [E7.6] |
| Wording (c), "ceiling": no call in 5 % or less of the units for both files | 7.0 % and 24.4 % | not met | [E7.6] |
| Wording (c) is the stronger probe for the two same-map files (gap above 26, p < 0.013) | +11; p 0.14 | not met; the later arms ran on the benchmark line | [E7.9] |
| Fit transplants, by kind: a group arm of the attention half meets the half-gap rule (13 misses or more, p < 0.05) | 1 of 4, at the edge (+21, p 0.044); +8, -2 and +4 for the other three | three group arms did not carry; the result is suggestive only | [E7.9] |
| Inkling turn split: one class holds half of the score gap or more (p < 0.05) | top class 0.30 and 0.31 of the gap (p 0.12, 0.096) | "diffuse"; no class located | [E7.10] |
| Inkling task-cost check: the rule's label for a pooled p under 0.05 that is not a clear cost | "a small cost"; the sign is the other way (+1.5 points) | the label does not fit; the plain reading is given beside it | [E7.13] |
| The KL-only winner and KL-table B at temperature 0 are each within 17 of the shipped file (their loss is slip damage) | 322 and 312 (shipped 330) | not met: KL-table B is one task below; reported per map | [E6.7] |
| One targeted sentence at medium effort lowers the unasked-answer habit (exact McNemar p < 0.05) | 10 vs 13, p 0.65 | not shown | [E6.10] |
| The sentence does not suppress asked answers (missing answers rise by 2 at most, passes drop by 3 at most) | missing answers 0 vs 0; passes 43 vs 47 | failed as written by one task; no asked answer was stopped | [E6.10] |
| The candidate map (KL-tools-2) ships only if AppWorld is level or positive | 666 vs 670 (and BFCL 492 vs 505) | held; the file is kept | [E6.14] |
| The coded bias, switched on, lowers the habit (forced read) | it raises it: -0.951 against a floor of 0.079 | the bias stays off | [E6.12] |
| Gradient map: a PyTorch forward equal to the engine's margins within 0.005 nats | max 0.041 | stopped before any gradient | [E6.13] |
| The band alone (down-projections of layers 20-27 at 2 bits in the shipped map) is below the shipped file, p < 0.05 | 671 vs 670 | not met; no band rule | [E5.12] |
| KL-table B with the down band restored is not below the shipped file and is above KL-table B | 627: below the shipped file (seed-level p 0.045); above KL-table B only at the task level (seed-level p 0.089) | not met; the cause of the remaining gap is open | [E5.14] |
| The uniform and HAWQ-style maps are below the shipped file (p < 0.05) and under 640 | 630 and 637; task-level p 0.0045 and 0.022; seed-level p 0.094 (without the doubled shard) and 0.15; seed 107 only | met as written at the task level; not kept at the seed level; no loss claimed | [E5.13] |
| The agent loss of the lower-KL maps grows with context length | the direction is the opposite | not shown | [E5.10] |
| The band damage concentrates on one token class | spread over four classes | not met | [E5.16] |
| The two-stage head lifts the score | 670 with it, 687 without it (+14 without the doubled shard, seed-level p 0.44) | not shown | [E5.6, E6.3] |
| The head repair improves agent work | 687 vs 670 (+15 without the doubled shard, seed-level p 0.46) | not shown on tasks | [E5.17, E6.3] |
| Sharpness explains the agent ordering | every map is flatter than BF16 | not met | [E5.8] |
| BF16 fails by running out of steps more than the shipped file | 3.8 % vs 6.1 % of the failed runs | not shown | [E5.9] |
| KL-tools is worse than the shipped file on chat documents (mean and median) | mean above with an interval that holds 0; median lower | "two documents, not the class" | [E5.11] |
| Free generation: no item over 1,000 characters of thinking (KL-tools) | one item of 24 with 1,184 characters | reported as an observation; the limit is not a validated threshold; map not shipped | [E5.26, E5.27] |
| Inkling: at equal bytes the per-group map lowers KL against the uniform rule | -2.5 %, 97.5 % interval holds 0 | "level"; the KL gap is credited to the codec, the fits and the format | [E5.20] |
| Inkling: the uniform map loses agent work on the harder split | 649 vs 638 for M2 | not met; no transfer claimed | [E5.23] |
| Inkling: the uniform map's tool-call KL gap has its interval above 0 | +14.5 % [-0.00300, +0.01282] | not met | [E5.23] |
| Inkling K/V setting: top-1 >= 98 % at 32k | 97.97 % +- 0.06 % | failed as written; setting kept on the KL line, which passed | [E2.2] |
| Inkling engine vs simulation: per-document r >= 0.95 | 0.919 (0.997 on documents with >= 500 rows) | failed as written; proceeded, decision recorded | [E2.6] |
| Harness gate hard lines on the final Inkling build: 0 loops | 1 OpenCode loop | failed; model-behaviour checks with rules written first; stated as a known limit | [E1.24, E1.25] |
| Inkling loop guard: >= 8 / 10 runs unstuck at each stuck point | 6, 6 and 7 / 10 (0-4 / 10 without it) | failed; guard on by default, release text gives the measured effect, not a pass | [E2.24] |
| Inkling loop guard: prompt-cache reuse >= 0.9 after a firing | 0.656 and 0.735 | failed; recorded | [E2.24] |
| Inkling release engine on Ampere (RTX 3080 Ti): KL within 2 % of the RTX 4070 value | 0.003092 vs 0.003341 (7.5 % lower); without one near-tie token on the 4070, -0.6 % | failed as written; the gap was traced to summation order (one token); top-1, recall and memory lines passed | [E4.13] |
| Inkling two-slot memory: shared (host) memory <= start + 64 MiB | 394 MiB (the tiered cache's pinned tail) | failed; two slots opt-in, the default stays one slot | [E2.25] |
| KL-only selection costs agent work (standard split, fresh seeds, p < 0.05) | 296 vs 292 | not met on that split | [E1.31] |
| Selection effect: interval of the selection effect above 0 | +13.3 [+0.0, +27.0] | not met; stated as a procedure | [E3.15] |
| Per-class K/V matters for recall (pass / fail) | every arm full | not met; repair shown on KL and memory only | [E2.22] |
| Per-class K/V matters for recall (graded) | intervals include 0 | not met; 4-bit global cache inconclusive | [E2.23] |
| Agent-trace probe sees a known loser | +0.0021 [-0.0030, +0.0072] | probe dropped | [E1.14] |
| Task-guided map search: best candidate on doc 529 | 0.728 vs line 0.422 | failed; study paused | [E1.13] |

## 7. What we did not do (honest scope)

We did not invent the quantizer, the trellis codec, or the 27B bit map, which is L0xRE-V2 by YourHighnessLA. We did
not find a better 27B map that we could ship. The method found and repaired damage in context: KL-tools scores 70
task-runs above the KL-only winner (seed-level p = 0.0017), its 8 raised projections carry the gain, and it is level
with the published allocation on agent work at 7.9 % lower KL. Class-level KL found that repair, not a task probe.
KL-tools thinks much longer than the published allocation on one deep item in free generation and so failed a line
written before the run; a forced reading under BF16 shows that this is not damage in the new map (both files share
the departure from BF16, and after it BF16 prefers the long reply). A second candidate, with lower KL than the shipped
file and every other gate passed, was held because both agent benchmarks came out a little negative. The shipped
build keeps the published allocation because of these lines, not because of shown damage. We found where the lead
of the 2.47-bit file over BF16 at temperature 1.0 sits (one end-of-task habit of the base model, at a lower rate), but
not why a bit map changes that rate. At temperature 0 the lead is not significant. On Inkling 2 Mini our rule chose
the map and the K/V setting; the measured value of the per-group map over a uniform one is in long-range copy
margin, not in KL and not in agent work, and a public file with exact down projections has more copy margin than
ours. The method kept six other changes out of that release (Section 5), held one public claim on fresh seeds, and
its harness gate caught an engine fault that the KL gate passed. The engine fault came from our own porting, and the
fix came from upstream work by others. The statistics of versions 1.0.x were too optimistic; version 1.1 corrected
them (Appendix B).

For the decision-point habits of Section 4.10 the scope is the same. We did not build a map to remove a habit. The
hunt for the part of our candidate maps that brings the return-turn misses to BF16's rate ended at its stop line
("diffuse at the side level"). The fit transplants point at the fits of the first half of the layers, and we call
that suggestive only; on the development half of a fresh private set the gap between the two files was not shown,
one of the four leads showed again, and a graft chosen there was not confirmed on the sealed half. On Inkling 2 Mini the turn split found no class that holds a score gap. A cross-over places the first-call
decision at the step after the thinking, no group of weights is located, a part of the decision may sit in the
serving stack, and the decision has no measured task cost. What we have is a way to detect a
habit that a total hides, to replay it in minutes, and to test levers and controls on the frozen turn.

## 8. Future work

The step that this paper does not have is the repair of a habit (Section 4.10). It has three parts, and we claim
no result for any of them. (a) A map that is built to remove a habit: for the return turn, a file built from the
prediction of the locate step. (b) A confirmation on held-out turns. A private set now exists. On its
development half one graft arm, chosen after the look, missed less often than both of its parent files. On the
sealed half it was not confirmed: the control of the read did not reach its line, and the graft was not below the
shipped file (Section 4.10). The next test needs a group that the locate step names before any look, and more
entries. (c) A check that the map
loses nothing else that we measure. Until these exist, the claim of the paper stops at detect, replay and locate.

*Donors that a lab already has (a direction, not a result).* The tensor copies of Table H2 use a donor file with the
same parent weights and the same bit map. A lab that quantizes or trains a model has more such files: its own other
attempts from the same parent weights, with other calibration sets, seeds or Hessians, and its rejected candidates.
A habit replay can say which of those files has less of a given habit in the located groups, also when that file lost
on the total score. Then the rejected candidates are a pool of donors for a graft, and a habit profile for each file
selects from the pool. Two conditions apply. The files must come from the same parent weights, so that the tensors
line up. They must have the same bit layout in the grafted groups, so that the copy is exact. Two questions are open.
First, do grafts for two habits compose? Our records show that habits can trade between files: KL-tools gives fewer
unasked answers than the shipped file (39 against 58 runs) and more misses at the return turn (59 against 28
entries), and on BFCL the candidate map KL-tools-3 has more turns with no call (93 against 73) and fewer failures
with a wrong state (150 against 167) [E6.4, E7.1]. Second, does a graft hold on held-out turns? The one that we
read was not confirmed (Section 4.10). We claim no result for any of this beyond the rows of Table H2. In those rows every copy
moved the count toward the file with more misses. One graft in the other direction, chosen on the development half
of the private set, was not confirmed on the sealed half; a graft that lowers a habit on held-out turns has not been
built.

Open steps of the studies of Section 4.10. For the 27B model: a depth test of the return turn at long context (it is
running, and it has no result here; Section 6); the transplant arms that did not run (the
output-projection fits of the linear-attention layers in layers 0-31 alone, and their complements; the split of the
lower layer half; the split of the MLP kinds; the reverse copies into the second-fit file; the metadata keys alone),
and a repeat of the group result on other histories and under sampling. For Inkling 2 Mini: the type of the second block that the
server drops at the first call (a check is queued, and it is not in this version); a serving-side test that returns
that block, to see how much of the first-call decision goes away with no change of the weights (not run); two more
seeds for the confirm (they are running, and they are not in this version); and the same decision in a client that ends the turn on a reply with no tool
call. For both: a *habit profile* of a map, one replay count for each known habit, set against the parent and the
shipped map, as a screen before any full benchmark.

The earlier results point to more instruments that this paper does not have. We name five directions and claim no
result for any of them. (1) Probes of fidelity to the reference: count the places where a map changes the
reference's own choice, in place of an average distance. (2) Near-tie fragility: judge an item by the reference's own
gap between its two best tokens, so that a flip at a near tie is not read as damage. (3) Drowning tests: for each
repair that adds bits to a group, test the mirror map that removes them, to find groups whose bits buy nothing.
(4) Signals inside the model, validated by planted damage before any use. (5) A gradient map from the reference's
decision margins; its forward pass did not match the engine within the line (0.041 nats against 0.005), so it
stopped before any gradient [E6.13]. For the agent benchmark itself: per-call seeds (Section 4.1), three or more
seeds for each comparison, a temperature-0 arm, and a full arm at effort xhigh on fresh seeds.

## 9. Related work

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
maps with lower KL lost agent work and recall at depth in our tests (Tables 2 and 3), and we keep KL as one gate among
several. We put
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
- Benchmarks: AppWorld, BFCL v3 (the Gorilla project, Apache-2.0), HumanEval, WikiText-2 (test split), used for
  evaluation only. The evidence repository holds outcomes for each entry or task (ids and 0 / 1 flags), with no
  prompt or reply text.

## AI use

Large language model agents, run as coding agents, ran most experiments, wrote most of the code and drafted most of
this text, under the author's direction. The pass lines were written before each run, by the author or by an agent
under his direction, and kept in a timestamped private record. The author checked the results against the logs and
decided what ships and what is claimed. Every number in this paper is read from a log or a scorer output listed in
`evidence/EVIDENCE.md`.

An LLM assistant also wrote the adversarial review of the first draft, at the author's request and with access to
the private record. The three stop-rule tests (B1, the naive file, test_challenge) came from that review. The review
continued for version 1.1: it asked for the baselines and the controls of the second batch, and lab agents ran them.
The review did not ask for the seed-level statistics: the lab found the shared seed stream while it compared the
transcripts of passing and failing runs (Section 4.1). A second LLM review, of version 1.1, gave the text changes of
version 1.2 and two recomputed numbers, which we recomputed from the per-task files [E6.1, E6.2]. Neither review is an
independent peer review.

Version 1.3. LLM agents planned, ran and analysed the studies of Section 4.10 and drafted this text, under the
author's direction. The author approved each study and decided what is claimed. A coordinating agent read some
cells of the return-turn study after it had seen earlier results; the record marks those reads as not blind. To
check them, a separate agent recounted every cell of both studies from the raw rows. Before its first commit it had
read only the plans (for the rules) and the tools that wrote the rows (for the format), and not the entries, the
reports or any result file [E7.14]. The agent that drafted this text counted the cells a third time with its own
code, which is in the evidence repository. Every count of a complete cell agrees. The counts of the private set came after the blind recount; two agents
counted them from the raw rows, and they agree. This is a check of the counting,
not of the design, and it is not an independent peer review.

## Acknowledgements

This work stands on the work of many others (from the public CREDITS of the release): the Qwen team (Qwen3.8-27B);
turboderp (exllamav3, the EXL3 codec, kernels); Albert Tseng, Qingyao Sun, David Hou and Christopher De Sa (QTIP);
Escha Labs (the Escha W2 format and reference, and the first exact 8-bit head export); YourHighnessLA (the L0xRE-V2
bit allocation shipped unchanged, and the
L0xRE runtime); StillDeadcode (the public Hessians, golden set and KL scorer, CC BY 4.0) and the publishers of the
corpora behind them; the ggml / llama.cpp contributors; Ajay9o9, spiritbuun, PrismML, sudoingX (batch-invariant
mode), Chirag Patnaik, alesha-pro, Mirai Labs, usmaneth, zhaoyilun, cklxx, Empressia, lenny76, abdulaziz, Scott
Guyton, Strike3D, Ankit Khandelwal, Mark Schatza, Jason Titus, Tomer Weller (engine work); syv-ai / HyperQwen, z-lab
(DFlash), Inco AI and ggml-org (drafting), Magnitude (mixed K/V idea); Thinking Machines (Inkling 2 Mini, Apache-2.0),
Unsloth (the Inkling build and GGUFs we compared against, and the UD-Q8_K_XL file we used to run the official
weights) and ShinMegamiBoson (LookAtMyQuant); Piotr Wilkin (llama.cpp Inkling 2 support and the DSpark drafter
code we ported); Aldehir Rojas (the llama.cpp Inkling chat parser and the grammar-trigger guard whose absence in our
port caused the harness failures of Section 4.8, and the lazy-grammar reasoning gating); Alexey Fateev (alesha-pro)
and Peasant Smith and the Project Maya contributors (the per-token launch-overhead measurement on Inkling 2 Mini and
the idea to fuse its glue launches); Killy (the voxel-pagoda prompt behind the bad-token check); DogukanUrker (the PatchEval board and its public traces); the authors of HumanEval and
AppWorld.
Testers and reporters of the releases: Lazm, Abdelrahman Waref, Mustafa YILDIRIM, loadnl and luluijn (AMD builds);
Amel (donamelo), Doğukan Ürker, xgxx666, hyf03288 and XeonG (reports and fixes); Elhussein Ahmed (emourad; the Linux
launcher of the Inkling release).
No person or company named here endorses this work.

## References

[AppWorld] H. Trivedi, T. Khot, et al. AppWorld: A Controllable World of Apps and People for Benchmarking Interactive
Coding Agents. ACL 2024. arXiv:2407.18901.

[AWQ] J. Lin et al. AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. MLSys 2024.
arXiv:2306.00978.

[BFCL] S. G. Patil, H. Mao, C. C.-J. Ji, F. Yan, V. Suresh, I. Stoica, J. E. Gonzalez. The Berkeley Function Calling
Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models. ICML 2025. Software and data (v3,
multi-turn): <https://github.com/ShishirPatil/gorilla>.

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

[LookAtMyQuant] ShinMegamiBoson. LookAtMyQuant. Software, <https://github.com/ShinMegamiBoson/lookatmyquant>
(commit 826e5dde for the newer build; the README at commit 92bb52d lists eight ideas taken from the Kintsugi
repositories). Model files: Hugging Face shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-8.9GB,
revision 1318b804e687 (2026-11-21; the file measured in this paper) and revision 6a953d2d1787 (2026-11-23; newer,
not measured here).

[Marchisio et al.] K. Marchisio et al. How Does Quantization Affect Multilingual LLMs? Findings of EMNLP 2024.
arXiv:2407.03211.

[Mekala et al.] A. Mekala, A. Atmakuru, Y. Song, M. Karpinska, M. Iyyer. Does quantization affect models' performance
on long-context tasks? EMNLP 2025. arXiv:2505.20276.

[MoEQuant] X. Hu, Z. Chen, D. Yang, et al. MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language
Models via Expert-Balanced Sampling and Affinity Guidance. ICML 2025. arXiv:2505.03804.

[NIAH] G. Kamradt. Needle In A Haystack. Software, 2023.

[Nosek et al.] B. A. Nosek, C. R. Ebersole, A. C. DeHaven, D. T. Mellor. The preregistration revolution. PNAS
115(11):2600-2606, 2018.

[PatchEval] DogukanUrker. PatchEval: repository-repair tasks for coding agents. Board and public dataset
DogukanUrker/PatchEval, 2026. <https://patcheval.dogukanurker.com/>

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
| KL-table A, KL-table B | V5a, V5c (V5cc-h32 with the release head) | lower-KL maps from the KL table |
| KL-only winner | V5bc-h32 | the lower-KL map that passed every non-agent gate |
| KL-tools | V5tc-h32 (V5t) | the KL-only winner with the down band of layers 20-27 restored |
| band-only map | S-fd | the shipped map with only the down-projections of layers 20-27 at 2 bits |
| KL-table B, down band restored | W2-C g3 | KL-table B with the down-projections of layers 16-31 back at 3 bits |
| uniform baseline, HAWQ-style baseline | U1c-h32, H1c-h32 | the G-B baselines |
| shipped file in BF16's setup | W2-S (arm pv3acB) | the weights-against-setup control |
| head-off arm, 3-bit-head arm | W2-H (pv3ac0), W2-K (pv3ack) | the shipped file without the two-stage head; the same with the 3-bit head and embedding |
| author's fits | L0xRE-V2 body | the original author's fits of the shipped layout |
| proxy map | V3b | Hessian-weighted fit-error map |
| five task-guided maps | C1-C5 | task-damage map candidates |
| expert-layer map, M2 | M1, M2 | the two Inkling maps named before any result |
| uniform Inkling map; Unsloth's file in our engine; official weights on the harder split | U, A (W2-D); FULL (W2-I) | the Inkling baselines of the second batch |
| M2 with a 4-bit head, copy-aware map, borrowed allocation | M2h, M3, M5 | rejected Inkling repairs |
| sparse attention past 200k | S1v2 | rejected 27B runtime repair |
| KL gate, recall gate, harness gate, bad-token check | G1, G2 / G3, G6, GATE P | release gates |
| free-generation check | gate F | thinking length on the 24 needles at 245k |
| graded margin probe | P1 (Qwen), ink_margin gf (Inkling) | teacher-forced recall margins |
| seed-level statistics | G-G, G-G2 | the design-effect test and the cluster-robust interval |
| tests of the first batch (agent ablation, selection effect, K/V ablation, graded K/V, graded baselines) | G-A1, G-A2, G-C, G-C2, G-B2 | test names in `evidence/LINES.md` |
| tests of the second batch | W2-A to W2-U, W3-C, W3-E, W3-T, W4-L | test names in `evidence/LINES.md` |
| KL-tools-2 (the held candidate) | V5td-h32 | KL-tools with 26 projections back at the shipped widths |
| raises-only map, cuts-only map | R8c-h32, C13c-h32 (W6-R) | the KL-only winner with only the 8 raises, or only the 13 cuts, of KL-tools |
| unasked answer; unasked answer only; pass + unasked only | history row A12; A12-only; pass + A12-only (L-S2) | the end-of-task class of Section 4.3 |
| tests of version 1.2 | W6-R, W7-X, W7-A, W7-S, W8-S, W9-V, W10-T, L-S2 | test names in `evidence/LINES.md` |
| KL-tools-3 | V5tf (W11-F) | KL-tools with 29 projections back at the shipped widths; passed the long-copy gates; not shipped |
| second-fit file | V2body (a row of Table 5) | the shipped bit map with a second set of fits |
| return turn; HS\*, HT\* | miss_func return turn; histories HS\*, HT\* | the turn in which a held-back function comes back; the histories made by the shipped file and by KL-tools-3 |
| rule 1, rule 2 | fields `call`, `called_held` | no tool call; the returned function not called |
| cells of Table H1 | s0-* (the 2 x 2 and two repeats), b-* (the 80 GB card), a6-* and s3-* (stripped thinking, effort xhigh), a3-* (wordings va, vb, vc), a2-* (sampling) | cell names in `data/habits/` |
| arms of the locate step, part 1 | S+R, S+C; k-qkv, k-gate, k-sso, k-down, k-up (cells s1-*, s2-*) | the shipped file with the common raises, the common cuts, or one kind of them |
| arms of Table H2, in table order | L1a, L1b, L1c, AS, FF, LO, HI, Kqkv, Kag, Ksso, Katt (cells l1-*, l2-*, l3-*) | tensor copies between the shipped file and the second-fit file |
| the official weights of Inkling 2 Mini, in Section 4.10 | FULL (the record), FP (the confirm) | the official NVFP4 / MXFP8 release, run as Unsloth's UD-Q8_K_XL file (an exact copy of the release, E4.10) |
| first-call arms | P1 (M2), P2 (UD-Q2_K_XL in our engine), FP, R1, C1, C2, FPR; PB1, PB2 (the probe); TW1, TW2 (the cross-over: M2 weights, the 2-bit file's weights) | the arms of the first-call confirm and of its two probes |
| studies of version 1.3 | return-turn hunt (steps 0-3, amendments A1-A8; lines V-1, V-2, P-1, S-1, W-a, W-b, W-c); Inkling turn split; first-call confirm (lines L1-L3, amendments A1-A4); blind recount | study names in `evidence/LINES.md` |

## Appendix B. Version history

**Version 1.3 (2026-11-27).** The central result is now decision-point habits (new Section 4.10), from studies made
after version 1.2:

- A procedure in four steps (detect, freeze and replay, locate, repair). This version does the first three and states
  that the fourth is open: no build of ours was made to remove a habit.
- The return turn of BFCL miss_func on the 27B model: found by a split by turn type while the totals were level
  (E7.1); a replay of one frozen call, valid by its line, with two miss rules given everywhere (E7.2); the mechanism
  (most misses say that the tool is absent) (E7.3); the parent check in one setup (E7.4); stripped thinking, effort
  and three wordings (E7.5, E7.6); sampling on three seeds (E7.7).
- The locate step: a stop line for the common changes of the candidate maps (E7.8), and tensor copies between two
  files with one bit map, reported as suggestive only, with both sets of labels (E7.9). On the development half of a
  fresh private set the control holds, the gap between the two files is not shown, one of the four leads shows again,
  and one graft arm, chosen after the look, is below both files (E7.15). One read of that arm on the sealed half did
  not confirm it: the control did not reach its line, and the graft was not below the shipped file (E7.16).
- Inkling 2 Mini: a turn split that found no class that holds the score gap (E7.10); a pre-registered confirm of a
  first-call decision on fresh tasks, with its three lines passed and the official weights as a report row (E7.11); a
  decision probe that did not locate it (E7.12); a pre-registered cross-over of the thinking, which places the
  decision at the step after the thinking (E7.18); a post hoc correction of what the decision is (the client
  receives one sentence, and the model writes the action in a second block that the server drops; E7.19); and a
  task-cost check that shows no cost in this client (E7.13).
- A blind recount of both studies by a separate agent (E7.14), and per-unit outcome files with a script that repeats
  the counts.
- The coding-agent description in the limitations has the board's later run on release v1.2 (E6.16b).
- The title is the one chosen for this version; the abstract leads with the mechanism. "Full precision" is defined
  in Section 2. The AI-use statement is
  updated. New names in Appendix A; new missed and stop lines in Table 10.

**Version 1.2 (2026-11-26).** Text and evidence. No run was made for this text; the new results come from lab runs
made after version 1.1, and the review items from a second LLM review of version 1.1:

- The title now says what the method does (it gates on tasks); class-level KL, not a task probe, found the repair.
- The temperature claim is narrowed: at temperature 0 the lead is +11 [-5, +28], and one greedy run cannot show
  whether it is smaller than at 1.0; the difference of the two leads does not keep p < 0.05 at the seed level (E6.1).
  A descriptive split of each arm's temperature-1.0 runs by its greedy result is added (Table 1b, E6.2).
- New Section 4.3: the mechanism of the lead over BF16, one end-of-task habit of the base model (the unasked answer),
  with its post hoc limits; the maps at temperature 0; reasoning effort xhigh lowers the habit on a targeted set, one
  targeted sentence does not; the coded bias stays off (E6.4, E6.6-E6.12).
- The repair split: the 8 raised projections carry the gain of KL-tools; with the band test, an interaction (E6.5).
  KL-tools on four seeds (E6.6). A held candidate, KL-tools-2 (E6.14).
- Clean numbers for two doubled shards: uniform -37 (p 0.094), head off +14 (p 0.44), head repair +15 (p 0.46); no
  verdict changes (E6.3).
- Inkling 2 Mini is named as another lineage; new Section 4.9 on readings across compressions of one base.
- Vault 9: the two early departures are damage that both files share, not damage in the new map.
- The AI-use statement is corrected: the lab, not the review, found the shared seed stream.
- A coding-agent audit is added to the limitations as a description (E6.16). The abstract is cut to about 260
  words, and this appendix holds the version history.

**Version 1.1 (2026-11-24).** A second batch of tests (17 new arms, about 14,500 more task-runs, most of them on the
harder agent split) and a correction of the statistics of versions 1.0.x. Those versions tested every AppWorld
comparison with an exact sign test over task-runs; Section 4.1 shows that the task-runs of one seed share one random
stream, so those p-values were overstated. Under the seed-level method:

- Keep p < 0.05: shipped above BF16 on the harder split at temperature 1.0 (four seeds); the same file above BF16 in
  BF16's setup; KL-tools above the KL-only winner; the KL-only winner and KL-table B below the shipped file; our
  Inkling build above the public 2-bit file with the setup equal; the shipped file above the naive 2-bit file. The
  three tests of the stop rule of version 1.0 keep their verdicts at their family alpha of 0.0167 (the two AppWorld
  tests have seed-level p = 0.0056 and 0.0078) [E5.1, E5.28].
- Do not keep it: the losses of the uniform and the HAWQ-style maps on the harder split (Section 4.4); the partial
  recovery of one flip map (Section 4.4); and three results on the selection seeds that versions 1.0.x already
  reported as not replicated or withdrawn (the loss of the KL-only winner there, p 0.019 at the task level and 0.13
  at the seed level; the lead of 21 over BF16, 0.0025 and 0.057; a public head-to-head, 0.0079 and 0.090) [E5.28].

Version 1.1 also replaced the scope sentence of version 1.0.3 (the method found and repaired damage in context), made
the Inkling headline a task result in place of a KL ratio (a uniform map at the same bytes keeps 93 % of the KL gap),
named the measured LookAtMyQuant file exactly, and corrected one error of the record (KL-table B has two cuts against
the shipped map, not one).

**Versions 1.0-1.0.3 (2026-11-23).** The first public version (1.0) with the three tests of the stop rule; then
presentation fixes (1.0.1), text corrections against the held-out recall test (1.0.2) and a scope statement (1.0.3).
No result changed in 1.0.1-1.0.3. `CHANGELOG.md` of the evidence repository gives each item.
