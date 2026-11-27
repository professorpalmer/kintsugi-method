# Public copy for the Kintsugi-method paper repository (paths point into this repository).
"""Paper v1.3, Section 4.10 (decision-point habits): recount of every count from the per-unit outcome files. CPU only.

    python -X utf8 scripts/v13_habits.py   -> results/v13-habits.txt, results/v13-habits.json

Input: data/habits/return-turn-units.csv and inkling-firstcall-units.csv (outcomes only: the raw rows hold benchmark
content and are not published).

Part 1, the return turn of BFCL v3 multi_turn_miss_func (Qwen3.8-27B). A unit = one entry on one recorded history
(HS = the history the shipped file made, HT = the history the candidate map KL-tools-3 made). Two miss rules:
  rule 1 "no call"            the reply has no tool call (the rule fixed in the study's plan);
  rule 2 "function not called" no call names a returned function (added after the first read; exploratory).
For two cells X and Y: D = misses(X) - misses(Y), a = units where only X misses, b = units where only Y misses,
p = the exact two-sided sign test on (a, b). Pooled over the two histories the ENTRY is the unit: d = the sum over its
histories of (miss X - miss Y), tested on d > 0 against d < 0.

Part 2, the first call of an AppWorld task (Inkling 2 Mini). A unit = one task on one seed. no_code = the reply has no
code block. A reply cut at the output cap is counted against the claim in lines L1 and L2 and by its content elsewhere.
"""
import collections
import csv
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "habits")
OUT = os.path.join(ROOT, "results")
L, J = [], {}


def w(s=""):
    L.append(s)


def sign_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def fp(p):
    return f"{p:.2g}" if p >= 0.001 else f"{p:.1e}"


# ------------------------------------------------------------------ part 1: the return turn
RT = collections.defaultdict(dict)          # cell -> entry -> (rule1, rule2, prompt_tokens, live)
with open(os.path.join(DATA, "return-turn-units.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        RT[r["cell"]][r["entry"]] = (int(r["no_call"]), int(r["held_not_called"]), int(r["prompt_tokens"]), int(r["live_no_call"]))
RULES = (("rule 1 no call", 0), ("rule 2 function not called", 1))


def miss(cell, k):
    return sum(v[k] for v in RT[cell].values())


def pair(x, y, k):
    """x, y = cells of one history; -> n, miss x, miss y, D, a, b, p on the common entries"""
    ids = sorted(set(RT[x]) & set(RT[y]))
    mx, my = sum(RT[x][i][k] for i in ids), sum(RT[y][i][k] for i in ids)
    a = sum(RT[x][i][k] > RT[y][i][k] for i in ids)
    b = sum(RT[x][i][k] < RT[y][i][k] for i in ids)
    return {"n": len(ids), "x": mx, "y": my, "D": mx - my, "a": a, "b": b, "p": sign_p(a, b)}


def pooled(xs, ys, k):
    """xs, ys = lists of cells (one per history or seed), paired in order; the entry is the unit"""
    d = collections.Counter()
    for x, y in zip(xs, ys):
        for i in set(RT[x]) & set(RT[y]):
            d[i] += RT[x][i][k] - RT[y][i][k]
    up, dn = sum(v > 0 for v in d.values()), sum(v < 0 for v in d.values())
    return {"entries": len(d), "D": sum(d.values()), "up": up, "down": dn, "p": sign_p(up, dn)}


def s_pair(q):
    return f"{q['x']} vs {q['y']} of {q['n']}, D {q['D']:+d}, {q['a']} / {q['b']}, p {fp(q['p'])}"


def s_pool(q):
    return f"D {q['D']:+d}, {q['up']} / {q['down']}, p {fp(q['p'])} ({q['entries']} entries)"


def both(tag, name, x, y, hists=("HS", "HT"), pool=True):
    """print a comparison under both rules; x, y are cell stems (history suffix added)"""
    J[tag] = {}
    w(f"[{tag}] {name}")
    for rn, k in RULES:
        parts = []
        for h in hists:
            q = pair(f"{x}-{h}", f"{y}-{h}", k)
            parts.append(f"{h}* {s_pair(q)}")
            J[tag][f"{rn}|{h}"] = q
        if pool and len(hists) == 2:
            q = pooled([f"{x}-{h}" for h in hists], [f"{y}-{h}" for h in hists], k)
            parts.append(f"pooled by entry {s_pool(q)}")
            J[tag][f"{rn}|pooled"] = q
        w(f"    {rn}: " + "; ".join(parts))


w("PART 1. The return turn (BFCL v3 multi_turn_miss_func), Qwen3.8-27B. Counts are misses.")
w("Cell names: s0 = the 2 x 2; KL-tools-3 is 'V5tf' in the records. HS* = 200 entries, HT* = 198 entries.")
w()
w("(1a) Cells: units / rule 1 misses / rule 2 misses")
for c in sorted(RT):
    w(f"    {c}: {len(RT[c])} / {miss(c, 0)} / {miss(c, 1)}")
J["cells"] = {c: [len(RT[c]), miss(c, 0), miss(c, 1)] for c in sorted(RT)}
w()
w("(1b) Replay noise: the shipped cell on HS* run again (R1 same seeds, R2 other seeds)")
for rn, k in RULES:
    w(f"    {rn}: R1 {s_pair(pair('s0-shipped-HS-R1', 's0-shipped-HS', k))}; R2 {s_pair(pair('s0-shipped-HS-R2', 's0-shipped-HS', k))}")
w()
w("(1c) The live runs that made the histories (first reply of the return turn; the rig's repeat of the benchmark turn)")
hs, ht = RT["s0-shipped-HS"], RT["s0-V5tf-HT"]
ids = sorted(set(hs) & set(ht))
a = sum(ht[i][3] > hs[i][3] for i in ids)
b = sum(ht[i][3] < hs[i][3] for i in ids)
both_ = sum(ht[i][3] and hs[i][3] for i in ids)
w(f"    shipped {sum(v[3] for v in hs.values())} of {len(hs)}; KL-tools-3 {sum(v[3] for v in ht.values())} of {len(ht)}; "
  f"paired on {len(ids)} entries: KL-tools-3 only {a}, shipped only {b}, both {both_}, p {fp(sign_p(a, b))}")
J["live"] = {"shipped": [sum(v[3] for v in hs.values()), len(hs)], "V5tf": [sum(v[3] for v in ht.values()), len(ht)], "a": a, "b": b,
             "both": both_, "p": sign_p(a, b)}
for cell, nm in (("s0-shipped-HS", "shipped on HS*"), ("s0-V5tf-HT", "KL-tools-3 on HT*")):
    eq = sum(v[0] == v[3] for v in RT[cell].values())
    w(f"    replay equal to the live call, {nm}: {eq} of {len(RT[cell])} ({100 * eq / len(RT[cell]):.1f} %)")
w()
w("(1d) Main comparisons, both rules")
both("2x2", "KL-tools-3 vs shipped (RTX 4070 cells)", "s0-V5tf", "s0-shipped")
both("parent", "BF16 vs shipped, one A100, one engine and setup", "b-BF16", "b-shipped", hists=("HS",))
both("parent-v5tf", "KL-tools-3 vs shipped on the A100", "b-V5tf", "b-shipped", hists=("HS",))
both("parent-bf16-v5tf", "BF16 vs KL-tools-3 on the A100", "b-BF16", "b-V5tf", hists=("HS",))
J["setup"] = {}
w("[setup] shipped on the A100 vs shipped on the RTX 4070 box, HS* (same requests)")
for rn, k in RULES:
    q = pair("b-shipped-HS", "s0-shipped-HS", k)
    J["setup"][rn] = q
    w(f"    {rn}: {s_pair(q)}; entries that differ {q['a'] + q['b']}")
both("strip-shipped", "shipped with earlier thinking stripped vs shipped", "a6-shipped-nothink", "s0-shipped")
both("strip-v5tf", "KL-tools-3 with earlier thinking stripped vs KL-tools-3", "s3-V5tf-nothink", "s0-V5tf")
both("strip-gap", "KL-tools-3 stripped vs shipped stripped", "s3-V5tf-nothink", "a6-shipped-nothink")
both("strip-v2body", "refit stripped vs shipped stripped", "a6-V2body-nothink", "a6-shipped-nothink", hists=("HT",))
both("strip-v2body-self", "refit stripped vs refit", "a6-V2body-nothink", "s1-V2body", hists=("HT",))
both("xhigh", "KL-tools-3 at effort xhigh vs KL-tools-3", "s3-V5tf-xhigh", "s0-V5tf")
both("xhigh-shipped", "KL-tools-3 at effort xhigh vs shipped", "s3-V5tf-xhigh", "s0-shipped")
for v, text in (("va", "wording (a) paraphrase"), ("vb", "wording (b) the withheld request again, no announcement"),
                ("vc", "wording (c) the function is named")):
    both(f"word-{v}", f"{text}: KL-tools-3 vs shipped", f"a3-V5tf-{v}", f"a3-shipped-{v}")
    w(f"    shipped under this wording vs shipped under the benchmark line (other requests by design): " + "; ".join(
        f"{rn}: " + ", ".join(f"{h}* {miss(f'a3-shipped-{v}-{h}', k)} vs {miss(f's0-shipped-{h}', k)}" for h in ("HS", "HT")) for rn, k in RULES))
both("word-vc-refit", "wording (c): refit vs shipped", "l1-V2body-vc", "a3-shipped-vc")
w()
w("(1e) Sampling at the launcher default (temperature 1.0, top-k 20, top-p 0.95, min-p 0.05), three seed bases")
J["sampling"] = {}
for rn, k in RULES:
    ds = []
    for sb in (20261201, 20261202, 20261203):
        x, y = f"a2-V5tf-s{sb}", f"a2-shipped-s{sb}"
        q = pooled([f"{x}-HS", f"{x}-HT"], [f"{y}-HS", f"{y}-HT"], k)
        ds.append(q["D"])
        w(f"    {rn}, seed base {sb}: HS* {s_pair(pair(x + '-HS', y + '-HS', k))}; HT* {s_pair(pair(x + '-HT', y + '-HT', k))}; pooled {s_pool(q)}")
    xs = [f"a2-V5tf-s{sb}-{h}" for sb in (20261201, 20261202, 20261203) for h in ("HS", "HT")]
    ys = [f"a2-shipped-s{sb}-{h}" for sb in (20261201, 20261202, 20261203) for h in ("HS", "HT")]
    q = pooled(xs, ys, k)
    m = sum(ds) / 3
    sd = math.sqrt(sum((d - m) ** 2 for d in ds) / 2)
    half = 4.302653 * sd / math.sqrt(3)            # t(0.975, 2 df)
    tx, ty = sum(miss(c, k) for c in xs), sum(miss(c, k) for c in ys)
    n = sum(len(RT[c]) for c in xs)
    w(f"    {rn}, three seeds: KL-tools-3 {tx} vs shipped {ty} of {n} replies; by entry {s_pool(q)}; "
      f"D per seed {ds}, mean {m:+.1f}, 95 % t interval (2 df) [{m - half:+.1f}, {m + half:+.1f}]")
    J["sampling"][rn] = {"D_seed": ds, "mean": m, "lo": m - half, "hi": m + half, "x": tx, "y": ty, "n": n, "by_entry": q}
w()
w("(1f) Prompt length: rule 1 misses in the shorter and the longer half of the prompts (split at the median)")
for cell in ("s0-V5tf-HT", "s0-shipped-HT", "b-BF16-HS"):
    v = sorted(RT[cell].values(), key=lambda t: t[2])
    h = len(v) // 2
    w(f"    {cell}: shorter {sum(t[0] for t in v[:h])} of {h}; longer {sum(t[0] for t in v[h:])} of {len(v) - h}")
w()
w("(1g) The locate step. G1 = the gap of KL-tools-3 on HT* under rule 1 (the plan's analysis set); G2 = the pooled gap of")
w("     the refit (same bit map as shipped, other fits) under rule 1. 'plan label' = the rule of the study's plan applied")
w("     on HT* against G1: carries iff share >= 0.60 and p < 0.05 and D > 4; clear iff share <= 0.30 and p >= 0.05; else")
w("     partial. 'half-gap' = the rule of the later amendment: pooled D >= G2 / 2 and pooled p < 0.05.")
G1 = pair("s0-V5tf-HT", "s0-shipped-HT", 0)["D"]
G2 = pooled(["s1-V2body-HS", "s1-V2body-HT"], ["s0-shipped-HS", "s0-shipped-HT"], 0)
w(f"    G1 = {G1:+d}; G2 = {s_pool(G2)}; half of G2 = {G2['D'] / 2:g}")
ARMS = [("s1-V2body", "refit (all 400 fits and 53 other tensors differ)"), ("s1-SR", "shipped + the 40 common raises of the candidate maps"),
        ("s1-SC", "shipped + the 33 common cuts"), ("s2-k-qkv", "shipped + raises: attn_qkv (17)"), ("s2-k-gate", "shipped + raises: ffn_gate (8)"),
        ("s2-k-sso", "shipped + raises: ssm_out (15)"), ("s2-k-down", "shipped + cuts: ffn_down (28)"), ("s2-k-up", "shipped + cuts: ffn_up (5)"),
        ("l1-L1a", "shipped + the refit's 53 non-fit tensors"), ("l1-L1b", "refit + shipped's 53 non-fit tensors (the refit's 400 fits)"),
        ("l1-L1c", "shipped + the refit's output head"), ("l2-AS", "shipped + the refit's fits: attention and ssm kinds (208)"),
        ("l2-FF", "shipped + the refit's fits: ffn kinds (192)"), ("l2-LO", "shipped + the refit's fits: layers 0-31 (200)"),
        ("l2-HI", "shipped + the refit's fits: layers 32-63 (200)"), ("l3-Kqkv", "shipped + the refit's fits: attn_qkv (48)"),
        ("l3-Kag", "shipped + the refit's fits: attn_gate (48)"), ("l3-Ksso", "shipped + the refit's fits: ssm_out (48)"),
        ("l3-Katt", "shipped + the refit's fits: attn_q, k, v, output (64)")]
J["locate"] = {}
for stem, text in ARMS:
    q1h = pair(f"{stem}-HT", "s0-shipped-HT", 0)
    s = q1h["D"] / G1
    label = "carries" if (s >= 0.60 and q1h["p"] < 0.05 and q1h["D"] > 4) else ("clear" if (s <= 0.30 and q1h["p"] >= 0.05) else "partial")
    p1 = pooled([f"{stem}-HS", f"{stem}-HT"], ["s0-shipped-HS", "s0-shipped-HT"], 0)
    p2 = pooled([f"{stem}-HS", f"{stem}-HT"], ["s0-shipped-HS", "s0-shipped-HT"], 1)
    q2h = pair(f"{stem}-HT", "s0-shipped-HT", 1)
    half = "met" if (p1["D"] >= G2["D"] / 2 and p1["p"] < 0.05) else "not met"
    J["locate"][stem] = {"rule1_HS": pair(f"{stem}-HS", "s0-shipped-HS", 0), "rule1_HT": q1h, "rule1_pooled": p1, "rule2_HT": q2h,
                         "rule2_pooled": p2, "share_G1": s, "plan_label": label, "half_gap": half}
    w(f"    {stem} = {text}")
    w(f"        rule 1: HS* {s_pair(pair(stem + '-HS', 's0-shipped-HS', 0))}; HT* {s_pair(q1h)}; pooled {s_pool(p1)}")
    w(f"        rule 2: HS* {s_pair(pair(stem + '-HS', 's0-shipped-HS', 1))}; HT* {s_pair(q2h)}; pooled {s_pool(p2)}")
    w(f"        plan label on HT* (share {s:.2f}): {label}; half-gap rule: {half}")
for tag, x, y in (("refit - L1b", "s1-V2body", "l1-L1b"), ("refit - L1a", "s1-V2body", "l1-L1a"), ("LO - HI", "l2-LO", "l2-HI"),
                  ("AS - FF", "l2-AS", "l2-FF"), ("AS - Ksso", "l2-AS", "l3-Ksso"), ("Ksso - Kag", "l3-Ksso", "l3-Kag")):
    w(f"    direct, rule 1, pooled: {tag}: {s_pool(pooled([x + '-HS', x + '-HT'], [y + '-HS', y + '-HT'], 0))}")

# ------------------------------------------------------------------ part 2: the first call (Inkling 2 Mini)
FC = collections.defaultdict(dict)          # (arm, dataset, seed) -> task -> (no_code, cut)
with open(os.path.join(DATA, "inkling-firstcall-units.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        FC[(r["arm"], r["dataset"], int(r["seed"]))][r["task"]] = (int(r["no_code"]), int(r["cut"]))
NAMES = {"FP": "official weights (run as UD-Q8_K_XL), f16 K/V, 48 GB card", "P2": "public 2-bit file (UD-Q2_K_XL), our engine, 12 GB card",
         "P1": "our map (M2), 12 GB card"}
w()
w("PART 2. The first call of an AppWorld task, Inkling 2 Mini. Counts are first replies with no code block.")
w()
w("(2a) test_challenge, 417 tasks per seed: no code / cut at the cap, per seed (cut replies scored by their content)")
SEEDS = sorted({s for (a, d, s) in FC if d == "test_challenge"})
J["firstcall"] = {"seeds": SEEDS, "arms": {}}
for arm in ("FP", "P2", "P1"):
    cells, tn, tc, tu = [], 0, 0, 0
    for s in SEEDS:
        u = FC.get((arm, "test_challenge", s))
        if u is None:
            cells.append(f"seed {s}: not run")
            continue
        n, c = sum(v[0] for v in u.values()), sum(v[1] for v in u.values())
        cells.append(f"seed {s}: {n} of {len(u)} ({100 * n / len(u):.1f} %), cut {c}")
        tn, tc, tu = tn + n, tc + c, tu + len(u)
    J["firstcall"]["arms"][arm] = {"no_code": tn, "cut": tc, "n": tu,
                                   "per_seed": {s: sum(v[0] for v in FC[(arm, "test_challenge", s)].values())
                                                for s in SEEDS if (arm, "test_challenge", s) in FC}}
    w(f"    {arm} = {NAMES[arm]}: " + "; ".join(cells) + f"; all: {tn} of {tu} ({100 * tn / tu:.1f} %), cut {tc}")


def by_task(A, B, fa, fb, seeds):
    """x = (B events) - (A events) per task over the seeds; -> B>A, A>B, p; and the same by scenario"""
    x = collections.Counter()
    for s in seeds:
        ua, ub = FC[(A, "test_challenge", s)], FC[(B, "test_challenge", s)]
        for t in set(ua) & set(ub):
            x[t] += fb(ub[t]) - fa(ua[t])
    up, dn = sum(v > 0 for v in x.values()), sum(v < 0 for v in x.values())
    sc = collections.Counter()
    for t, v in x.items():
        sc[t.split("_")[0]] += v
    su, sd = sum(v > 0 for v in sc.values()), sum(v < 0 for v in sc.values())
    return {"tasks": len(x), "up": up, "down": dn, "p": sign_p(up, dn), "scen": len(sc), "s_up": su, "s_down": sd, "s_p": sign_p(su, sd)}


plain = lambda v: v[0]
against_p1 = lambda v: int(v[0] or v[1])        # P1: a cut reply counts as no code
against_p2 = lambda v: int(v[0] and not v[1])   # P2: a cut reply counts as code
LINE_SEEDS = (102, 103)          # the seeds of the plan: L1 and L2 are scored on these, whatever seeds come later
both_seeds = [s for s in LINE_SEEDS if all((a, "test_challenge", s) in FC for a in ("P1", "P2"))]
all_seeds = [s for s in SEEDS if all((a, "test_challenge", s) in FC for a in ("P1", "P2"))]
w()
w(f"(2b) The lines written before the run (seeds {both_seeds})")
p2 = sum(against_p2(v) for s in both_seeds for v in FC[("P2", "test_challenge", s)].values())
p1 = sum(against_p1(v) for s in both_seeds for v in FC[("P1", "test_challenge", s)].values())
n2 = sum(len(FC[("P2", "test_challenge", s)]) for s in both_seeds)
n1 = sum(len(FC[("P1", "test_challenge", s)]) for s in both_seeds)
w(f"    L1 (the public 2-bit file's rate is 20 % or more; a cut reply counts as code): {p2} of {n2} = {100 * p2 / n2:.1f} % -> {'PASS' if p2 / n2 >= 0.20 else 'FAIL'}")
q = by_task("P1", "P2", against_p1, against_p2, both_seeds)
ratio = (p1 / n1) / (p2 / n2)
ok = q["up"] > q["down"] and q["p"] < 0.001 and ratio <= 0.5
w(f"    L2 (our map is below it: sign test by task p < 0.001 and rate ratio <= 0.5; cut replies counted against): our map {p1} of {n1} = "
  f"{100 * p1 / n1:.1f} %, ratio {ratio:.3f}; tasks 2-bit file higher / our map higher {q['up']} / {q['down']}, p {fp(q['p'])}; "
  f"scenarios {q['s_up']} / {q['s_down']} of {q['scen']}, p {fp(q['s_p'])} -> {'PASS' if ok else 'FAIL'}")
J["firstcall"]["L1"] = [p2, n2]
J["firstcall"]["L2"] = {"p1": p1, "n1": n1, "ratio": ratio, **q}
w("    per seed (2-bit file minus our map, plain rule): " + ", ".join(
    f"{s}: {sum(plain(v) for v in FC[('P2', 'test_challenge', s)].values()) - sum(plain(v) for v in FC[('P1', 'test_challenge', s)].values()):+d}" for s in both_seeds))
q = by_task("P1", "P2", plain, plain, both_seeds)
w(f"    plain rule by task: {q['up']} / {q['down']}, p {fp(q['p'])}")
for arm in ("P1", "P2"):
    ev = collections.Counter()
    for s in both_seeds:
        for t, v in FC[(arm, "test_challenge", s)].items():
            ev[t] += v[0]
    w(f"    tasks with the event on one seed or more, {arm}: {sum(v > 0 for v in ev.values())} of {len(ev)}")
fp_seeds = [s for s in both_seeds if ("FP", "test_challenge", s) in FC]
if fp_seeds:
    w()
    w(f"(2c) The official weights against the two 12 GB files (report only; plain rule; seeds {fp_seeds})")
    for A, B, text in (("P1", "FP", "official weights higher / our map higher"), ("FP", "P2", "2-bit file higher / official weights higher")):
        q = by_task(A, B, plain, plain, fp_seeds)
        ds = ", ".join(f"{s}: {sum(plain(v) for v in FC[(B, 'test_challenge', s)].values()) - sum(plain(v) for v in FC[(A, 'test_challenge', s)].values()):+d}" for s in fp_seeds)
        w(f"    {A} vs {B}: by task {text} {q['up']} / {q['down']}, p {fp(q['p'])}; by scenario {q['s_up']} / {q['s_down']} of {q['scen']}, p {fp(q['s_p'])}; per seed ({B} - {A}) {ds}")
        J["firstcall"][f"{A}-{B}"] = q
if len(all_seeds) > len(both_seeds):
    def t_p(ds):
        """two-sided p of a one-sample t-test of the per-seed differences (numeric integral of the t density)"""
        n = len(ds)
        m = sum(ds) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in ds) / (n - 1))
        if sd == 0:
            return float("nan")
        t0, df = abs(m / (sd / math.sqrt(n))), n - 1
        c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
        k, h = 20000, t0 / 20000
        area = sum(c * (1 + ((i + 0.5) * h) ** 2 / df) ** (-(df + 1) / 2) for i in range(k)) * h
        return max(0.0, 1 - 2 * area)

    w()
    w(f"(2c2) All seeds (report only; the lines above stay as scored on seeds {list(LINE_SEEDS)}); plain rule; seeds {all_seeds}")
    J["firstcall"]["all_seeds"] = {"seeds": all_seeds}
    for A, B, text in (("P1", "P2", "2-bit file higher / our map higher"), ("P1", "FP", "official weights higher / our map higher"),
                       ("FP", "P2", "2-bit file higher / official weights higher")):
        ss = [s for s in all_seeds if all((x, "test_challenge", s) in FC for x in (A, B))]
        if len(ss) <= len(both_seeds):
            w(f"    {A} vs {B}: no seed beyond the line seeds")
            continue
        q = by_task(A, B, plain, plain, ss)
        ds = [sum(plain(v) for v in FC[(B, "test_challenge", s)].values()) - sum(plain(v) for v in FC[(A, "test_challenge", s)].values()) for s in ss]
        w(f"    {A} vs {B}, seeds {ss}: by task {text} {q['up']} / {q['down']}, p {fp(q['p'])}; by scenario {q['s_up']} / {q['s_down']} of {q['scen']}, "
          f"p {fp(q['s_p'])}; per seed ({B} - {A}) {ds}; t-test over the seeds p {fp(t_p(ds))}")
        J["firstcall"]["all_seeds"][f"{A}-{B}"] = {**q, "per_seed": ds, "t_p": t_p(ds)}
w()
w("(2d) test_normal, seed 102, 168 tasks (the replay of the record, and two controls on our map)")
r1 = FC[("R1", "test_normal", 102)]
n = 168
pmf = [math.comb(n, k) * (50 / 672) ** k * (1 - 50 / 672) ** (n - k) for k in range(n + 1)]
cdf, lo, hi = 0.0, None, n
for k in range(n + 1):
    cdf += pmf[k]
    if lo is None and cdf >= 0.025:
        lo = k
    if cdf >= 0.975:
        hi = k
        break
c = sum(v[0] for v in r1.values())
w(f"    L3 (our map's replay is inside the exact 95 % interval of the record's 50 of 672 at n 168 = [{lo}, {hi}]): {c} of {len(r1)} "
  f"({100 * c / len(r1):.1f} %), cut {sum(v[1] for v in r1.values())} -> {'PASS' if lo <= c <= hi else 'FAIL'}")
J["firstcall"]["L3"] = {"count": c, "lo": lo, "hi": hi}
for arm, text in (("C1", "control: f16 K/V"), ("C2", "control: no reasoning budget"), ("FPR", "official weights (48 GB card)")):
    u = FC.get((arm, "test_normal", 102))
    if u is None:
        continue
    a = sum(u[t][0] > r1[t][0] for t in u)
    b = sum(u[t][0] < r1[t][0] for t in u)
    w(f"    {arm} ({text}): {sum(v[0] for v in u.values())} of {len(u)} ({100 * sum(v[0] for v in u.values()) / len(u):.1f} %), cut {sum(v[1] for v in u.values())}; "
      f"against our map's replay by task: only this arm {a}, only our map {b}, p {fp(sign_p(a, b))}")
    J["firstcall"][arm] = {"count": sum(v[0] for v in u.values()), "a": a, "b": b}

t = "\n".join(L)
print(t)
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "v13-habits.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write(t + "\n")
with open(os.path.join(OUT, "v13-habits.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(J, f, indent=1, default=float)
