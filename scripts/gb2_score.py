"""G-B2 scorer (docs/paper/GAPS.md # G-B2; statistic and line written before any run). Deterministic.
    python -I -X utf8 tools/gb2/gb2_score.py      -> logs/gb2/gb2-result.txt

Data: logs/tdm/gb2-{shipped,U1,H1}/p1.jsonl (probe P1, tools/tdm/p1.py: the 44 G2 items at 117,000, one /completion
per answer token, top-20 margins). Arms: the release file Kintsugi-27B-12G.gguf ("shipped"), U1c-h32, H1c-h32.

Pairing (replaces the first draft of this scorer, which keyed records by item name): the copy bench asks for the same
function in two or more adjacent slots (20 copy slots hold 13 distinct functions at 117k), so name keys collapse 44
items to 37. Items pair by position; every arm sent the same 44 requests in the same order (asserted on the names).

Statistic (GAPS): m = min margin over the item's answer tokens, capped at 5 nats; d = m(X) - m(shipped); mean over
items; 95 % percentile bootstrap over items, 10,000 resamples, numpy default_rng(20261122). Primary set: the 43 items
without `_check_tzname`. Reported beside it: the 44-item set, `_check_tzname`'s own margins (min, first-token margin,
position of the minimum), items with m < 0 per arm, and the copy family's mean per-token margin difference
(secondary; same bootstrap over the 20 copy slots). Index matrices are drawn once per set size, in a fixed order, and
shared by both arms.
Line: "X loses recall margin" only if the 43-item interval lies entirely below 0; if it contains 0: "no general
recall-margin loss shown; the G2 miss is the fragile item only"; if above 0: "X has more recall margin than the
shipped map" (reported, no other claim).
"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "results", "gb2-result.txt")
CAP, FRAG, SEED, NB, S1_REF = 5.0, "_check_tzname", 20261122, 10000, 207.334
ARMS = {"shipped": "Kintsugi-27B-12G.gguf (0510bae8...)", "U1": "U1c-h32 (d02dabb4...)", "H1": "H1c-h32 (fa49974c...)"}


def load(tag):
    with open(os.path.join(ROOT, "data", "margins", "gb2", tag + ".jsonl"), encoding="utf-8") as f:
        rs = [json.loads(x) for x in f if x.strip()]
    items = [r for r in rs if "item" in r]
    summ = [r for r in rs if r.get("summary")]
    assert len(items) == 44 and len(summ) == 1, tag
    return items, summ[0]


def cap(r):
    return min(r["min"], CAP)


R, SUM = {}, {}
for t in ARMS:
    R[t], SUM[t] = load(t)
NAMES = [(r["family"], r["item"]) for r in R["shipped"]]
for t in ARMS:
    assert [(r["family"], r["item"]) for r in R[t]] == NAMES, f"order differs: {t}"
FI = [i for i, k in enumerate(NAMES) if k == ("copy", FRAG)]
assert len(FI) == 1
FI = FI[0]
ALL = list(range(44))
P43 = [i for i in ALL if i != FI]
COPY = [i for i in ALL if NAMES[i][0] == "copy"]

rng = np.random.default_rng(SEED)
IDX = {n: rng.integers(0, n, size=(NB, n)) for n in (43, 44, 20)}  # fixed draw order: 43, 44, 20


def mean_ci(v):
    v = np.asarray(v, dtype=float)
    b = v[IDX[len(v)]].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


out = []
w = out.append
w("G-B2 result (tools/gb2/gb2_score.py; GAPS.md # G-B2). Probe P1 at 117,000, rig RTX 4070, sprint131 profile, "
  "bin-p10-5599dede7, thinking off, layer off.")
w(f"Capped min margin m (cap {CAP} nats). Bootstrap: {NB} resamples of the items, seed {SEED}, percentile 95 %.")
w("")
w("== Determinism check ==")
w(f"  shipped S1 = {SUM['shipped']['S1']} (11-21: {S1_REF}) -> {'EQUAL: PASS' if SUM['shipped']['S1'] == S1_REF else 'DIFFERS: FAIL'}")
w("")
w("== Per arm ==")
for t, desc in ARMS.items():
    rs = R[t]
    neg = [f"{NAMES[i][0]} item {i}: {NAMES[i][1]} ({rs[i]['min']:+.3f})" for i in ALL if rs[i]["min"] < 0]
    w(f"  {t} {desc}: S1 {SUM[t]['S1']}; mean capped m {np.mean([cap(r) for r in rs]):.3f} (needles "
      f"{np.mean([cap(rs[i]) for i in ALL if i not in COPY]):.3f}, copies {np.mean([cap(rs[i]) for i in COPY]):.3f}); "
      f"items m < 0: {len(neg)} {neg}; tokens < 0: {sum(r['n_neg'] for r in rs)}; bound flags {sum(r['below_top'] for r in rs)}")
w("")
w(f"== {FRAG} (item {FI} of 44; copy slot {FI - 24}) in each arm ==")
for t in ARMS:
    r = R[t][FI]
    m = r["margins"]
    w(f"  {t}: min {r['min']:+.3f}; first-token margin {m[0]:+.3f}; minimum at position {int(np.argmin(m))} of "
      f"{r['answer_tokens']}; tokens < 0: {r['n_neg']}; mean per-token {r['mean']:.3f}")
w("")
w("== Paired vs shipped ==")
RES = {}
for t in ("U1", "H1"):
    X, S = R[t], R["shipped"]
    d = [cap(X[i]) - cap(S[i]) for i in ALL]
    p43 = mean_ci([d[i] for i in P43])
    p44 = mean_ci(d)
    cd = mean_ci([X[i]["mean"] - S[i]["mean"] for i in COPY])
    RES[t] = p43
    flips = [f"{NAMES[i][1]} (item {i}) {S[i]['min']:+.3f} -> {X[i]['min']:+.3f}" for i in ALL if X[i]["min"] < 0 <= S[i]["min"]]
    w(f"  {t} vs shipped:")
    w(f"    PRIMARY, 43 items without {FRAG}: mean d {p43[0]:+.3f} nats, 95 % [{p43[1]:+.3f}, {p43[2]:+.3f}]")
    w(f"    all 44 items: mean d {p44[0]:+.3f}, 95 % [{p44[1]:+.3f}, {p44[2]:+.3f}]")
    w(f"    by family (information, 43-item set): needles {np.mean([d[i] for i in P43 if i not in COPY]):+.3f}, "
      f"copies {np.mean([d[i] for i in P43 if i in COPY]):+.3f}")
    w(f"    secondary, copy family mean per-token margin difference (20 copies): {cd[0]:+.3f} nats [{cd[1]:+.3f}, {cd[2]:+.3f}]")
    w(f"    items m < 0 in {t} but >= 0 in shipped: {flips if flips else 'none'}")
    nz = [f"{NAMES[i][1]} (item {i}) {d[i]:+.3f}" for i in ALL if d[i] != 0]
    w(f"    items with d != 0 (the rest are capped at {CAP} in both arms or equal): {len(nz)} {nz}")
w("")
w("== Sensitivity (information only, no line): repeated copy requests counted once (first slot), 43-item set ==")
first = []
seen = set()
for i in P43:
    if NAMES[i] in seen:
        continue
    seen.add(NAMES[i]); first.append(i)
for t in ("U1", "H1"):
    d = np.array([cap(R[t][i]) - cap(R["shipped"][i]) for i in first])
    b = d[np.random.default_rng(SEED).integers(0, len(d), size=(NB, len(d)))].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    w(f"  {t}: {len(first)} distinct items, mean d {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
w("")
w("== Line (as written before any run) ==")
for t in ("U1", "H1"):
    p, lo, hi = RES[t]
    if hi < 0:
        v = f"{t} LOSES RECALL MARGIN"
    elif lo > 0:
        v = f"{t} HAS MORE RECALL MARGIN THAN THE SHIPPED MAP (reported, no other claim)"
    else:
        v = "NO GENERAL RECALL-MARGIN LOSS SHOWN; the G2 miss is the fragile item only"
    w(f"  {t}: 43-item interval [{lo:+.3f}, {hi:+.3f}] -> {v}")

t = "\n".join(out)
print(t)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    f.write(t + "\n")
