"""G-G statistics pass (docs/paper/GAPS.md # G-G). CPU only, deterministic.
    python -X utf8 scripts/gg_stats.py      -> results/gg-stats.txt

For every AppWorld count in the draft that has per-task files in data/appworld/:
- the count per arm with a 95 % Wilson interval (task-runs, pooled over the seeds of the row; the Wilson interval
  treats task-runs as independent, which they are not quite: the same task appears once per seed);
- the paired comparison: discordant split, exact two-sided sign test, and the minimum detectable difference (MDD):
  the smallest difference a - b, with a + b = the observed number of discordant pairs, that gives p < 0.05;
- a 95 % paired bootstrap interval of the difference in task-runs (resample the tasks, keep every seed of a task
  together; 10,000 resamples, numpy default_rng(20261122)), only over the tasks finished in both arms on every seed
  of the row.
- G-A2 check: d_sel recomputed as the batch scorer does, the share of resamples <= 0, and the BF16 (S - F)
  on the tasks where its fresh shards exist.
HumanEval counts (no per-task files used): Wilson intervals only.
"""
import csv
import json
import math
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
AW = os.path.join(ROOT, "data", "appworld")
OUT = os.path.join(ROOT, "results", "gg-stats.txt")
SEED, NB = 20261122, 10000

# arm -> {seed: file}
F = {
    "V3ac-h32 (S)": {100: "v3ac.csv", 101: "v3ac-s101.csv"},
    "V5bc-h32 (S)": {100: "v5bc.csv", 101: "v5bc-s101.csv"},
    "V5t (S)": {100: "v5t-s100.csv", 101: "v5t-s101.csv"},
    "L0xRE-V2 body (S)": {100: "v2body.csv", 101: "v2body-s101.csv"},
    "BF16 our engine (S)": {100: "bf16o-s100.csv", 101: "bf16o-s101.csv"},
    "naive UD-Q2_K_XL our engine (S)": {100: "q2kxlo-s100.csv", 101: "q2kxlo-s101.csv"},
    "Inkling Unsloth setup (100+101)": {100: "ink-s100.csv", 101: "ink-s101.csv"},
    "Kintsugi shipped (F 102+103)": {102: "kship-s102.csv", 103: "kship-s103.csv"},
    "Kintsugi weights-only (F 102+103)": {102: "kfw-s102.csv", 103: "kfw-s103.csv"},
    "BF16 (F 102+103)": {102: "bf16f-s102.csv", 103: "bf16f-s103.csv"},
    "Inkling Unsloth setup (102+103)": {102: "ink-s102.csv", 103: "ink-s103.csv"},
    "Inkling Unsloth setup (104+105)": {104: "ink-s104.csv", 105: "ink-s105.csv"},
    "Inkling OURS M2 (102+103)": {102: "inkm-s102.csv", 103: "inkm-s103.csv"},
    "Inkling OURS M2 (104+105)": {104: "inkm-s104.csv", 105: "inkm-s105.csv"},
    "Inkling OURS M2 (102-105)": {s: f"inkm-s{s}.csv" for s in (102, 103, 104, 105)},
    "Inkling exact weights RAW (102-105)": {s: f"raw-s{s}.csv" for s in (102, 103, 104, 105)},
    "Inkling q4_0 global cache Q65 (102-105)": {s: f"q65-s{s}.csv" for s in (102, 103, 104, 105)},
    "Inkling OURS M2 (102+103) for Q262": {102: "inkm-s102.csv", 103: "inkm-s103.csv"},
    "Inkling Q262 (102+103)": {102: "q262-s102.csv", 103: "q262-s103.csv"},
    "V3ac-h32 (B 104+105)": {s: f"pv3ac-s{s}.csv" for s in (104, 105)},
    "V5bc-h32 (B 104+105)": {s: f"pv5bc-s{s}.csv" for s in (104, 105)},
    "V5tc-h32 (B 104+105)": {s: f"pv5tc-s{s}.csv" for s in (104, 105)},
    "L0xRE-V2 body (B 104+105)": {s: f"pv2body-s{s}.csv" for s in (104, 105)},
    "BF16 (B 104+105)": {s: f"pbf16-s{s}.csv" for s in (104, 105)},
    "naive UD-Q2_K_XL our engine (B 104+105)": {s: f"pnaive-s{s}.csv" for s in (104, 105)},
    "V3ac-h32 (T3 challenge 106+107)": {s: f"t3-pv3ac-s{s}.csv" for s in (106, 107)},
    "V5bc-h32 (T3 challenge 106+107)": {s: f"t3-pv5bc-s{s}.csv" for s in (106, 107)},
    "V3ac-h32 (T3 challenge 106)": {106: "t3-pv3ac-s106.csv"},
    "V5bc-h32 (T3 challenge 106)": {106: "t3-pv5bc-s106.csv"},
    "V3ac-h32 (T3 challenge 107)": {107: "t3-pv3ac-s107.csv"},
    "V5bc-h32 (T3 challenge 107)": {107: "t3-pv5bc-s107.csv"},
}
# (label, arm X, arm Y, evidence row)
CMP = [
    ("C1 V5bc-h32 vs V3ac-h32, selection seeds", "V5bc-h32 (S)", "V3ac-h32 (S)", "E1.8"),
    ("C1 V5t vs V3ac-h32, selection seeds", "V5t (S)", "V3ac-h32 (S)", "E1.9"),
    ("C3 V2 body vs V3ac-h32, selection seeds", "L0xRE-V2 body (S)", "V3ac-h32 (S)", "E3.8"),
    ("C3 V3ac-h32 vs BF16, selection seeds", "V3ac-h32 (S)", "BF16 our engine (S)", "E3.6"),
    ("C1 V3ac-h32 vs naive (ours), selection seeds", "V3ac-h32 (S)", "naive UD-Q2_K_XL our engine (S)", "E1.20"),
    ("C3 Kintsugi vs Inkling Unsloth, 100+101", "V3ac-h32 (S)", "Inkling Unsloth setup (100+101)", "E3.9"),
    ("C3 shipped vs BF16, 102+103", "Kintsugi shipped (F 102+103)", "BF16 (F 102+103)", "E3.7"),
    ("C3 weights-only vs BF16, 102+103", "Kintsugi weights-only (F 102+103)", "BF16 (F 102+103)", "E3.7"),
    ("C2 shipped vs weights-only, 102+103", "Kintsugi shipped (F 102+103)", "Kintsugi weights-only (F 102+103)", "E2.14"),
    ("C3 Kintsugi vs Inkling Unsloth, 102+103", "Kintsugi shipped (F 102+103)", "Inkling Unsloth setup (102+103)", "E3.9"),
    ("C3 Inkling OURS vs Unsloth setup, 102+103", "Inkling OURS M2 (102+103)", "Inkling Unsloth setup (102+103)", "E3.12"),
    ("C3 Inkling OURS vs Unsloth setup, 104+105", "Inkling OURS M2 (104+105)", "Inkling Unsloth setup (104+105)", "E3.13"),
    ("C4 Inkling OURS vs exact weights, 102-105", "Inkling OURS M2 (102-105)", "Inkling exact weights RAW (102-105)", "E4.8"),
    ("C2 Inkling Q65 vs exact weights, 102-105", "Inkling q4_0 global cache Q65 (102-105)", "Inkling exact weights RAW (102-105)", "E4.12"),
    ("C2 Inkling Q65 vs OURS (q8_0 cache), 102-105", "Inkling q4_0 global cache Q65 (102-105)", "Inkling OURS M2 (102-105)", "E4.12"),
    ("C2 Inkling Q262 vs OURS, 102+103", "Inkling Q262 (102+103)", "Inkling OURS M2 (102+103) for Q262", "E4.12"),
    ("G-A1 V5bc-h32 vs V3ac-h32, 104+105", "V5bc-h32 (B 104+105)", "V3ac-h32 (B 104+105)", "E1.31"),
    ("G-A2 V5tc-h32 vs V3ac-h32, 104+105", "V5tc-h32 (B 104+105)", "V3ac-h32 (B 104+105)", "E3.16"),
    ("G-A2 V2 body vs V3ac-h32, 104+105", "L0xRE-V2 body (B 104+105)", "V3ac-h32 (B 104+105)", "E3.16"),
    ("G-A2 BF16 vs V3ac-h32, 104+105 (PARTIAL if < 336)", "BF16 (B 104+105)", "V3ac-h32 (B 104+105)", "E3.16"),
    ("STOP RULE 3 V5bc-h32 vs V3ac-h32, test_challenge 106+107", "V5bc-h32 (T3 challenge 106+107)", "V3ac-h32 (T3 challenge 106+107)", "E1.36"),
    ("STOP RULE 3 V5bc-h32 vs V3ac-h32, test_challenge 106", "V5bc-h32 (T3 challenge 106)", "V3ac-h32 (T3 challenge 106)", "E1.36"),
    ("STOP RULE 3 V5bc-h32 vs V3ac-h32, test_challenge 107", "V5bc-h32 (T3 challenge 107)", "V3ac-h32 (T3 challenge 107)", "E1.36"),
    ("G-B3 / STOP RULE 2 V3ac-h32 vs naive, 104+105", "V3ac-h32 (B 104+105)", "naive UD-Q2_K_XL our engine (B 104+105)", "E1.32"),
]
HE = [("V3ac-h32 rig", 161), ("V3ac-h32 box", 159), ("V5bc-h32", 162), ("Inkling M2", 158), ("Inkling Unsloth setup", 160)]


def load(fn):
    with open(os.path.join(AW, fn), encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    return {t: s == "1" for t, s in rows}


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h) * n, (c + h) * n


def sign(a, b):
    n, k = a + b, min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def pfmt(p):
    return f"{p:.4f}" if p >= 1e-4 else f"{p:.2e}"


def mdd(n):
    for k in range(n % 2, n + 1, 2):
        if k and sign((n + k) // 2, (n - k) // 2) < 0.05:
            return k
    return None


D = {arm: {s: load(fn) for s, fn in seeds.items()} for arm, seeds in F.items()}
rng = np.random.default_rng(SEED)
out = []
w = out.append
w("G-G statistics pass (tools/paper/gg_stats.py). AppWorld test_normal; counts are task-runs pooled over the seeds named.")
w("Wilson = 95 % Wilson interval of the count. Paired: exact two-sided sign test on discordant (task, seed) pairs;")
w("MDD = minimum difference with p < 0.05 at the observed number of discordant pairs; boot = 95 % paired bootstrap")
w(f"interval of the difference over tasks (seeds of a task kept together, {NB} resamples, seed {SEED}).")
w("")
w("== Arms ==")
for arm, by in D.items():
    k = sum(sum(v.values()) for v in by.values())
    n = sum(len(v) for v in by.values())
    lo, hi = wilson(k, n)
    w(f"  {arm:42s} seeds {'+'.join(str(s) for s in by)}: {k} / {n} ({100 * k / n:.1f} %), Wilson [{lo:.1f}, {hi:.1f}]"
      + ("" if n in (168 * len(by), 417 * len(by)) else "  [PARTIAL / unfinished runs]"))
w("")
w("== Paired comparisons (X vs Y) ==")
for label, x, y, ev in CMP:
    X, Y = D[x], D[y]
    seeds = sorted(set(X) & set(Y))
    units = [(s, t) for s in seeds for t in sorted(set(X[s]) & set(Y[s]))]  # pairs per (task, seed), as aw_base.pair
    a = sum(X[s][t] for s, t in units)
    b = sum(Y[s][t] for s, t in units)
    ox = sum(X[s][t] and not Y[s][t] for s, t in units)
    oy = sum(Y[s][t] and not X[s][t] for s, t in units)
    n = len(units)
    tasks = sorted({t for s, t in units})
    per = {t: 0 for t in tasks}
    for s, t in units:
        per[t] += int(X[s][t]) - int(Y[s][t])
    diff = np.array([per[t] for t in tasks], dtype=float)
    bs = diff[rng.integers(0, len(tasks), size=(NB, len(tasks)))].sum(axis=1)
    blo, bhi = np.percentile(bs, [2.5, 97.5])
    m = mdd(ox + oy)
    w(f"  [{ev}] {label}: {a} vs {b} of {n} paired task-runs ({len(tasks)} tasks, seeds {'+'.join(map(str, seeds))}); discordant "
      f"{ox} / {oy}, p = {pfmt(sign(ox, oy))}; diff {a - b:+d}, boot [{blo:+.0f}, {bhi:+.0f}]; MDD at {ox + oy} discordant = {m}")
w("")

# ---- G-A2 check ---------------------------------------------------------------------------------------------
w("== G-A2 check (as the batch scorer; winner V3ac-h32; others V5t(c), V2 body, V5bc; S = 100+101, F = 104+105) ==")
G = [("V3ac-h32 (S)", "V3ac-h32 (B 104+105)"), ("V5t (S)", "V5tc-h32 (B 104+105)"),
     ("L0xRE-V2 body (S)", "L0xRE-V2 body (B 104+105)"), ("V5bc-h32 (S)", "V5bc-h32 (B 104+105)")]
ids = sorted(set.intersection(*[set(D[s][q]) for s, f in G for q in D[s]] + [set(D[f][q]) for s, f in G for q in D[f]]))
M = {s: np.array([sum(D[s][q][t] for q in D[s]) - sum(D[f][q][t] for q in D[f]) for t in ids], dtype=float) for s, f in G}
keys = [s for s, f in G]


def dsel(idx):
    return M[keys[0]][idx].sum() - np.mean([M[k][idx].sum() for k in keys[1:]])


rng2 = np.random.default_rng(SEED)  # same stream as aw_paper.py
bt = np.array([dsel(rng2.integers(0, len(ids), len(ids))) for _ in range(NB)])
lo, hi = np.percentile(bt, [2.5, 97.5])
w(f"  tasks {len(ids)}; (S - F): " + ", ".join(f"{k.split(' (')[0]} {M[k].sum():+.0f}" for k in keys))
w(f"  d_sel = {dsel(np.arange(len(ids))):+.1f}, 95 % [{lo:+.1f}, {hi:+.1f}]; share of resamples <= 0: {np.mean(bt <= 0):.3f}; "
  f"< 0: {np.mean(bt < 0):.3f}")
bf_s, bf_f = D["BF16 our engine (S)"], D["BF16 (B 104+105)"]
bt_ids = sorted(set(bf_s[100]) & set(bf_s[101]) & set(bf_f[104]) & set(bf_f[105]))
bsf = sum(bf_s[100][t] + bf_s[101][t] - bf_f[104][t] - bf_f[105][t] for t in bt_ids)
w(f"  BF16 (never selected): (S - F) {bsf:+d} on the {len(bt_ids)} tasks with both fresh shards"
  f"{' (PARTIAL: 168 needed)' if len(bt_ids) < 168 else ''}")
w("")
w("== HumanEval (164 problems; Wilson only) ==")
for name, k in HE:
    lo, hi = wilson(k, 164)
    w(f"  {name}: {k} / 164, Wilson [{lo:.1f}, {hi:.1f}]")
w("")
w("== Resolution of the sign test (power note) ==")
for n in (20, 30, 33, 40, 45, 49, 52, 60):
    w(f"  {n} discordant pairs: p < 0.05 needs a difference of at least {mdd(n)} ({(n + mdd(n)) // 2} / {(n - mdd(n)) // 2})")

t = "\n".join(out)
print(t)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    f.write(t + "\n")
