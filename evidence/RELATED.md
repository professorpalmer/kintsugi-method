# Related work (verified citations)

Status: draft in the private repo, 2026-11-21 (cleared for a public version); section 9 added 2026-11-22. Every entry below was checked on 2026-11-21 against the arXiv, proceedings or
project page given (title, authors, year, venue). Where a venue could not be confirmed, the entry says "preprint".
Software without a paper is cited as software, with its URL and the date we read it.

Each entry ends with one sentence on how our method differs ("Ours:").

## 1. Sensitivity-based mixed precision

1. **HAWQ: Hessian AWare Quantization of Neural Networks with Mixed-Precision.** Zhen Dong, Zhewei Yao, Amir Gholami,
   Michael W. Mahoney, Kurt Keutzer. ICCV 2019. https://arxiv.org/abs/1905.03696
   Picks each layer's precision from its Hessian spectrum (top eigenvalue).
   Ours: we do not rank layers by a curvature score; our closest analogue (a Hessian-weighted per-projection fit error,
   V3b) gave +36.7 % KL at equal size, and we rank by measured effects on the output and on task probes instead.
2. **HAWQ-V2: Hessian Aware trace-Weighted Quantization of Neural Networks.** Zhen Dong, Zhewei Yao, Daiyaan Arfeen,
   Amir Gholami, Michael W. Mahoney, Kurt Keutzer. NeurIPS 2020. https://arxiv.org/abs/1911.03852 ;
   https://papers.nips.cc/paper/2020/hash/d77c703536718b95308130ff2e5cf9ee-Abstract.html
   Uses the average Hessian trace as the sensitivity.
   Ours: sensitivity is measured end to end (KL per MiB and recall margins per MiB, one group flipped at a time), and
   the two rankings disagree on which groups are safe.
3. **HAWQ-V3: Dyadic Neural Network Quantization.** Zhewei Yao, Zhen Dong, Zhangcheng Zheng, Amir Gholami, Jiali Yu,
   Eric Tan, Leyuan Wang, Qijing Huang, Yida Wang, Michael W. Mahoney, Kurt Keutzer. ICML 2021 (PMLR 139).
   https://arxiv.org/abs/2011.10680 ; https://proceedings.mlr.press/v139/yao21a.html
   Solves bit allocation as an integer linear program under size and latency limits.
   Ours: the allocation is a constrained choice too, but every candidate must also pass task lines written before the
   run; a better objective value is not enough to ship.
4. **SqueezeLLM: Dense-and-Sparse Quantization.** Sehoon Kim, Coleman Hooper, Amir Gholami, Zhen Dong, Xiuyu Li,
   Sheng Shen, Michael W. Mahoney, Kurt Keutzer. ICML 2024 (PMLR 235). https://arxiv.org/abs/2306.07629
   Sensitivity-weighted non-uniform codes plus a sparse full-precision part for outliers.
   Ours: we keep the codec fixed (trellis K2 / K3) and move whole projections between widths; the question is where,
   judged by tasks.
5. **SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression.** Tim Dettmers, Ruslan Svirschevski,
   Vage Egiazarian, Denis Kuznedelev, Elias Frantar, Saleh Ashkboos, Alexander Borzunov, Torsten Hoefler, Dan Alistarh.
   ICLR 2024. https://arxiv.org/abs/2306.03078
   Keeps outlier weights at higher precision; judged mainly by perplexity.
   Ours: we show cases where lower average error (KL, perplexity, NLL by depth) coexists with task loss, so perplexity
   is used as a gate input, not a target.
6. **Unsloth Dynamic 2.0 GGUFs.** Unsloth AI. Blog and documentation, 2025-04-24. https://unsloth.ai/blog/dynamic-v2 ;
   https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs
   Chooses a llama.cpp quant type per layer and reports mean KL divergence as the main quality metric.
   Ours: we use their UD-Q2_K_XL files as baselines; our maps are lower in KL on Inkling and our runtime / map
   choices are gated by recall and agent tasks, not by KL alone.
7. **LookAtMyQuant.** ShinMegamiBoson. Software (MIT), https://github.com/ShinMegamiBoson/lookatmyquant (read
   2026-11-21; Inkling files on Hugging Face under shinmegamiboson/, Apache-2.0). The file that we measured is the
   8.9 GB pack at Hugging Face revision 1318b804e687 (2026-11-21; 8,874,213,027 B). A newer 8.9 GB file exists
   (revision 6a953d2d1787 of 2026-11-23, code at GitHub commit 826e5dde); we did not measure it, and no sentence here
   applies to it. The author's README (commit 92bb52d) lists eight ideas taken from the Kintsugi repositories, among them per-class KL on chat-templated text and dense expert Hessians.
   Measures how much each stored piece degrades the output, then picks a code per piece under a size budget, with a
   distillation fine-tune of the small tensors.
   Ours: we found the allocation of the old file (M5) does not transfer to our codes (+43 % KL), and the old file is
   equal to ours on average KL but worse on long contexts and copy rows; we choose with task probes and confirm on
   held-out data. Three of our golden documents are copies of public text that is in that project's calibration
   text; the comparison holds without them (EVIDENCE E5.25).
8. **L0xRE-V2-27b-Low.** YourHighnessLA. Hugging Face model release, 2026.
   https://huggingface.co/YourHighnessLA/L0xRE-V2-27b-Low (card licence: "qwen-research-terms-derivative", no text
   shown; noted in the paper's data and licence statement).
   The 400 per-projection bit choices that Kintsugi-27B ships unchanged.
   Ours: we did not invent this allocation; our contribution on Qwen is the gate set that rejected every lower-KL
   map we built (V3s2, V5a-V5t, task-damage candidates).

8a. **escha-mlx (Escha W2 format).** Escha Labs. Software (Apache-2.0), https://github.com/EschaLabs/escha-mlx.
    EXL3 code with Hadamard input / output scales; our converter computes its codec tables with this reference.
    Ours: the format we store our fits in; not our contribution.
8b. **llama.cpp-escha.** Ajay9o9. Software (MIT), https://github.com/Ajay9o9/llama.cpp-escha.
    The Escha op for ggml (CPU and CUDA) that our engine port starts from.
    Ours: our engine builds on it; not our contribution.

## 2. Post-training weight quantization

9. **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** (ICLR title: "OPTQ: Accurate
   Quantization for Generative Pre-trained Transformers"). Elias Frantar, Saleh Ashkboos, Torsten Hoefler,
   Dan Alistarh. ICLR 2023. https://arxiv.org/abs/2210.17323
   One-shot second-order (Hessian-based) weight quantization.
   Ours: we use an existing second-order quantizer (exllamav3) as a black box; the method is about what to do after
   the fits.
10. **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.** Ji Lin, Jiaming Tang, Haotian
    Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han. MLSys 2024.
    https://arxiv.org/abs/2306.00978
    Protects salient channels found from activation statistics.
    Ours: salience is defined by the tasks users run (recall at depth, agent work), measured per group, not by
    activation magnitude.

## 3. Incoherence processing and trellis codes

11. **QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks.** Albert Tseng, Jerry Chee,
    Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa. ICML 2024 (PMLR 235). https://arxiv.org/abs/2402.04396 ;
    https://proceedings.mlr.press/v235/tseng24a.html
    Randomized Hadamard incoherence + E8 lattice codebooks.
    Ours: not a codec paper; our fits inherit the Hadamard input / output scales of this line through EXL3.
12. **QTIP: Quantization with Trellises and Incoherence Processing.** Albert Tseng, Qingyao Sun, David Hou,
    Christopher De Sa. NeurIPS 2024. https://arxiv.org/abs/2406.11235
    Trellis-coded quantization with compute-based Gaussian codes; the codec family we use.
    Ours: we take the codec as given and study allocation and runtime settings on top of it.
13. **ExLlamaV3 / EXL3.** turboderp. Software (MIT), https://github.com/turboderp-org/exllamav3 (format notes in
    doc/exl3.md; read 2026-11-21). A QTIP variant with its own regularization and packing, and the `quantize_exl3`
    quantizer that made every fit in this paper.
    Ours: we use EXL3 fits (K2, K3) as the fit library; no change to the codec.

## 4. K/V-cache quantization

14. **KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.** Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen
    Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, Xia Hu. ICML 2024 (PMLR 235).
    https://arxiv.org/abs/2402.02750
    Per-channel keys, per-token values, a full-precision recent window.
    Ours: we do not propose a K/V codec; we choose among existing types per cache class (global vs sliding-window)
    by a KL / top-1 line, and spend precision where it costs little memory.
15. **KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.** Coleman Hooper, Sehoon Kim,
    Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, Amir Gholami. NeurIPS 2024.
    https://arxiv.org/abs/2401.18079
    Per-channel, pre-RoPE, non-uniform K/V codes; judged by perplexity.
    Ours: our K/V choice is checked by KL at depth and then by recall at 64k-245k with the shipped launcher.

## 5. Long-context recall evaluation

16. **Needle In A Haystack: pressure testing LLMs.** Greg Kamradt. Software, 2023.
    https://github.com/gkamradt/LLMTest_NeedleInAHaystack
    Plants a fact at different depths and asks for it.
    Ours: our needle set is the same idea; we add exact code copies and teacher-forced margins on the gate's own
    prompt, which turn a pass / fail item into a continuous, deterministic score.
17. **RULER: What's the Real Context Size of Your Long-Context Language Models?** Cheng-Ping Hsieh, Simeng Sun, Samuel
    Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, Boris Ginsburg. COLM 2024.
    https://arxiv.org/abs/2404.06654
    Shows that vanilla needle tests are easy and adds multi-needle, tracing and aggregation tasks.
    Ours: we agree that easy items lack power (our wide probes passed a map that lost a needle); we use RULER-type
    breadth as a gap, not yet as a gate.
18. **Lost in the Middle: How Language Models Use Long Contexts.** Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin
    Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. TACL 12, 2024. https://arxiv.org/abs/2307.03172
    Recall depends on position in the context.
    Ours: we place needles at evenly spaced depths; we do not study position effects.

## 6. Evaluating compressed models beyond the average

19. **Accuracy is Not All You Need.** Abhinav Dutta, Sanjeev Krishnan, Nipun Kwatra, Ramachandran Ramjee. NeurIPS 2024.
    https://arxiv.org/abs/2407.09141
    Aggregate accuracy hides "flips"; proposes flips and KL divergence as distance metrics.
    Ours: closest in spirit, but we disagree on one point: they recommend KL (with flips) as the metric; at about 2.5
    bits KL misranked our maps on recall at depth, so KL is one gate among several, and we repair at the layer level
    (see "Disagreement with Dutta et al." in section 9).
20. **Compressing LLMs: The Truth is Rarely Pure and Never Simple.** Ajay Jaiswal, Zhe Gan, Xianzhi Du, Bowen Zhang,
    Zhangyang Wang, Yinfei Yang. ICLR 2024. https://arxiv.org/abs/2310.01382
    LLM-KICK: perplexity is a weak yardstick for compressed models; task suites reveal losses.
    Ours: we put task probes inside the allocation loop (diagnose, locate, repair), not only in the final evaluation.
21. **How Does Quantization Affect Multilingual LLMs?** Kelly Marchisio, Saurabh Dash, Hongyu Chen, Dennis Aumiller,
    Ahmet Ustun, Sara Hooker, Sebastian Ruder. Findings of EMNLP 2024. https://arxiv.org/abs/2407.03211
    Automatic metrics understate harm; non-Latin scripts are hit hardest.
    Ours: consistent with our one non-English long-chat document (doc 529) and the +153 % multilingual KL of M5; we
    use such documents as named checks.
22. **Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models.** Ruikang Liu et al. COLM 2025.
    https://arxiv.org/abs/2504.04823
    Low bit widths hurt hard reasoning tasks more.
    Ours: task-specific damage again; we localize it to layer groups instead of reporting it per method.
23. **"Give Me BF16 or Give Me Death"? Accuracy-Performance Trade-Offs in LLM Quantization.** Eldar Kurtic et al.
    ACL 2025. https://arxiv.org/abs/2411.02355
    Large-scale evaluation of 4-8 bit formats; most are near-lossless on standard suites.
    Ours: we work at about 2.5 bits per weight, where damage is real and local, and on one 12 GB card.

## 7. Selection bias, the winner's curse and held-out confirmation

24. **Anomalies: The Winner's Curse.** Richard H. Thaler. Journal of Economic Perspectives 2(1):191-202, 1988.
    https://doi.org/10.1257/jep.2.1.191
    The winning bid in a common-value auction tends to overpay.
    Ours: the same effect on a benchmark: the map chosen as best on AppWorld seeds 100 / 101 scored 8 fewer on fresh
    seeds, and its gap to BF16 went from +21 to +1.
25. **On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation.** Gavin C. Cawley,
    Nicola L. C. Talbot. JMLR 11:2079-2107, 2010. https://www.jmlr.org/papers/v11/cawley10a.html
    A noisy selection criterion over-fits and biases the reported score of the chosen model.
    Ours: an instance in LLM quantization, with the fix applied (fresh seeds, the same harness, the claim withdrawn).
26. **Do ImageNet Classifiers Generalize to ImageNet?** Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, Vaishaal
    Shankar. ICML 2019 (PMLR 97). https://arxiv.org/abs/1902.10811
    New test sets show accuracy drops, mostly not from adaptive over-fitting.
    Ours: our fresh-seed rerun is a small version of a replication test set; with n = 2 seed pairs we cannot split
    selection bias from seed noise, and we say so.
27. **The reusable holdout: Preserving validity in adaptive data analysis.** Cynthia Dwork, Vitaly Feldman, Moritz Hardt,
    Toniann Pitassi, Omer Reingold, Aaron Roth. Science, 2015. https://cis.upenn.edu/~aaroth/reusable.html ;
    technical version https://arxiv.org/abs/1506.02629
    Reusing a holdout adaptively breaks its guarantees unless access is controlled.
    Ours: we use the simplest control: a fresh evaluation set used once, after selection is closed.

## 8. Statistics, benchmarks and architecture

28. **Statistical Significance Tests for Machine Translation Evaluation.** Philipp Koehn. EMNLP 2004.
    https://aclanthology.org/W04-3250/
    Paired bootstrap resampling. Ours: we use the paired bootstrap over documents for KL and the exact sign test on
    discordant pairs for task counts.
29. **AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents.** Harsh Trivedi,
    Tushar Khot, et al. ACL 2024. https://arxiv.org/abs/2407.18901
    Our agent benchmark (test_normal, 168 tasks).
30. **Evaluating Large Language Models Trained on Code.** Mark Chen et al. arXiv preprint, 2021.
    https://arxiv.org/abs/2107.03374
    HumanEval; saturated for every file we built (158-162 / 164), so we use it as a health gate only.
31. **Gated Delta Networks: Improving Mamba2 with Delta Rule.** Songlin Yang, Jan Kautz, Ali Hatamizadeh. ICLR 2025.
    https://arxiv.org/abs/2412.06464
    The linear-attention (GDN) layer type of the dense hybrid model we compress.

## 9. Added 2026-11-22 after an adversarial review by an LLM assistant (each checked on 2026-11-22 on the arXiv / proceedings page given)

32. **EvoPress: Accurate Dynamic Model Compression via Evolutionary Search.** Oliver Sieberling, Denis Kuznedelev,
    Eldar Kurtic, Dan Alistarh. ICML 2025 (arXiv comment "ICML camera-ready"). https://arxiv.org/abs/2410.14649
    Searches non-uniform per-block / per-layer compression levels under a global target with an evolutionary search.
    Ours: also a search over per-group widths, but a candidate is judged by task probes and pre-written lines as well,
    and a held-out confirmation decides what is claimed.
33. **KVTuner: Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization for Efficient and Nearly Lossless LLM
    Inference.** Xing Li, Zeyu Xing, Yiming Li, Linping Qu, Hui-Ling Zhen, Wulong Liu, Yiwu Yao, Sinno Jialin Pan,
    Mingxuan Yuan. ICML 2025. https://arxiv.org/abs/2502.04420
    Offline multi-objective search for a key / value precision pair per layer; finds keys generally more sensitive.
    Ours: our K/V choice is per cache class (global vs sliding-window) of a looped MoE, chosen by a KL line and then
    checked on recall at 64k-245k; our blanket-type ablation found no recall difference.
34. **Does quantization affect models' performance on long-context tasks?** Anmol Mekala, Anirudh Atmakuru, Yixiao
    Song, Marzena Karpinska, Mohit Iyyer. EMNLP 2025 (to appear per arXiv). https://arxiv.org/abs/2505.20276
    8-bit quantization barely changes long-input task accuracy; 4-bit methods can lose a lot, by method, model and task.
    Ours: we work at about 2.5 bits and measure recall at 117k-245k with teacher-forced margins, per bit map.
35. **HAQ: Hardware-Aware Automated Quantization with Mixed Precision.** Kuan Wang, Zhijian Liu, Yujun Lin, Ji Lin,
    Song Han. CVPR 2019. https://arxiv.org/abs/1811.08886
    Reinforcement learning picks per-layer bit widths with feedback from the target hardware.
    Ours: the per-group choice is made from measured output and task effects at fixed bytes, not from a learned policy.
36. **MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and
    Affinity Guidance.** Xing Hu, Zhixuan Chen, Dawei Yang, et al. ICML 2025 (PMLR 267). https://arxiv.org/abs/2505.03804
    Calibration that covers every expert, and an expert-sample affinity term in the quantization error.
    Ours: we did not change the quantizer for the MoE; we chose which expert matrices get 3 bits by KL on a golden set.
37. **EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models.** Yuanteng Chen,
    Yuantian Shao, Peisong Wang, Jian Cheng. ACL 2025 (long). https://arxiv.org/abs/2508.01625 ;
    https://aclanthology.org/2025.acl-long.633
    Low-bit quantization shifts the router's expert choice; a router calibration (and expert pruning) repairs it.
    Ours: a router shift is one possible reason for our Inkling copy-row KL result; we did not measure router choices.
38. **The preregistration revolution.** Brian A. Nosek, Charles R. Ebersole, Alexander C. DeHaven, David T. Mellor.
    PNAS 115(11):2600-2606, 2018. https://doi.org/10.1073/pnas.1708274114
    Fix the question and the analysis before the outcome is seen, to keep prediction apart from postdiction.
    Ours: our "lines written before the run" are informal preregistration in a private record with timestamps, not a
    registry; we list every line that was missed.
39. **NeurIPS 2021 Workshop on Pre-registration in Machine Learning (Preface).** Samuel Albanie, Joao F. Henriques,
    Luca Bertinetto, Alex Hernandez-Garcia, Hazel Doughty, Gul Varol. PMLR 181, 2022.
    https://proceedings.mlr.press/v181/albanie22a.html
    A review model in which the protocol is reviewed before the results exist.
    Ours: the same discipline applied inside one project, without external review of the protocol.

**Disagreement with Dutta et al. (entry 19).** They recommend distance metrics (KL divergence and flips) next to
accuracy for evaluating compressed models. We agree that KL belongs in the evaluation, but not as the selection
criterion at about 2.5 bits: in our study maps with 7-11 % lower KL lost recall at depth (Table 1 of the draft). The
held-out tests decide how far this goes (see evidence/LINES.md; record section STOP RULE): if they find no task loss, the honest reading is
that KL misranked recall items in our gate set but did not cost tasks on held-out data.

## Positioning (one paragraph for the draft)

Most mixed-precision work chooses bits to minimize a proxy (curvature, activation salience, reconstruction error) or
an average output error (perplexity, KL), and evaluates task accuracy at the end. Recent evaluation papers (Dutta et
al.; Jaiswal et al.; Marchisio et al.) show that averages hide task-level changes. Our contribution sits between
the two: we put task probes (deep recall margins, long-answer documents, an agent benchmark) inside the allocation
and runtime-setting loop, write pass lines before each run, and confirm the winner on data that was not used to
select it. We do not contribute a quantizer, a codec, or (for the 27B model) a bit allocation.
