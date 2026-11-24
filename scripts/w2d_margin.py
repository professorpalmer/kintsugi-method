"""W2-D graded recall scorer (evidence/LINES.md section 6; G-C2's statistic). Deterministic.
    python -X utf8 scripts/w2d_margin.py   -> results/w2d-margin-result.txt (+ .json)

Data: data/margins/w2d/gm-<arm>-<depth>-<family>.jsonl (arms m2, a, u; 117000 / 245000; needles 24 + copies 20), rig, engine
a20cc419b, launcher defaults (q4_0 global + f16 SWA), method gf. Units = (family, slot, depth), paired by request order
(asserted on the item names); items = (family, slot), 44, both depths move together in the bootstrap.
m = min margin over the answer tokens, capped at 5 nats; d = m(X) - m(M2) per unit; mean d over the 88 units;
percentile bootstrap over the 44 items, 10,000 resamples, numpy default_rng(20261122), one shared index matrix;
97.5 % (the line, Bonferroni over the two baselines) and 95 % intervals. Noise floor 0.000 (G-C2).
Line per baseline: "less recall margin than M2" if the 97.5 % interval lies wholly below 0; "more" if wholly above 0;
otherwise "not separated". Information: M2 here vs G-C2's arm (d) (same file and settings, engine ad657a606 on a box).
"""
import json, os
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
L = os.path.join(ROOT, "data", "margins", "w2d")
RES = os.path.join(ROOT, "results")
CAP, SEED, NB = 5.0, 20261122, 10000
ARMS = {"m2": "M2 (shipped)", "a": "A: Unsloth UD-Q2_K_XL", "u": "U: uniform map, equal bytes"}
DEPTHS = (117000, 245000)
FAM = (("needle", 24), ("copy", 20))


def load(path):
    with open(path, encoding="utf-8") as f:
        return [r for r in (json.loads(x) for x in f if x.strip()) if "item" in r]


R, NAMES = {}, {}
for arm in ARMS:
    R[arm] = {}
    for d in DEPTHS:
        for fam, n in FAM:
            fp = os.path.join(L, f"gm-{arm}-{d}-{fam}.jsonl")
            if not os.path.exists(fp):   # interim scoring (a ticket still in the rig queue): that block is missing
                print(f"MISSING {arm} {d} {fam}: interim result for {arm}")
                continue
            rs = load(fp)
            assert len(rs) == n, (arm, d, fam, len(rs))
            names = [r["item"] for r in rs]
            assert names == NAMES.setdefault((fam, d), names), f"request order differs: {arm} {d} {fam}"
            for j, r in enumerate(rs):
                R[arm][(fam, j, d)] = r
ITEMS = [(fam, j) for fam, n in FAM for j in range(n)]
UNITS = [(fam, j, d) for fam, j in ITEMS for d in DEPTHS]
IDX = np.random.default_rng(SEED).integers(0, len(ITEMS), size=(NB, len(ITEMS)))
cap = lambda r: min(r["min"], CAP)


def V(f):
    M = np.zeros((len(ITEMS), len(DEPTHS)))
    for i, (fam, j) in enumerate(ITEMS):
        for k, d in enumerate(DEPTHS):
            M[i, k] = f((fam, j, d))
    return M


out, res = [], {}
w = out.append
w("W2-D graded recall (scripts/w2d_margin.py; evidence/LINES.md section 6). Rig RTX 4070, engine a20cc419b, launcher defaults, gf.")
w(f"Capped min margin (cap {CAP}); bootstrap {NB} x 44 items, seed {SEED}.")
for arm, desc in ARMS.items():
    US = [u for u in UNITS if u in R[arm]]
    neg = [f"{u[0][0]}{u[1]}@{u[2] // 1000}k {R[arm][u]['item']} {R[arm][u]['min']:+.3f}" for u in US if R[arm][u]["min"] < 0]
    w(f"  {arm} ({desc}): {len(US)} units; mean capped m {np.mean([cap(R[arm][u]) for u in US]):.3f}; units m < 0: {len(neg)} {neg}; "
      f"bound flags {sum(R[arm][u].get('below_top', 0) for u in US)}")
for x in ("a", "u"):
    M = V(lambda u: (cap(R[x][u]) - cap(R["m2"][u])) if u in R[x] else np.nan)
    if np.isnan(M).any():
        w(f"  {x}: INTERIM, {int(np.isnan(M).sum())} of 88 units missing")
    b = np.nanmean(M[IDX], axis=(1, 2))
    c95, c975 = np.percentile(b, [2.5, 97.5]), np.percentile(b, [1.25, 98.75])
    pt = float(np.nanmean(M))
    v = "LESS recall margin than M2" if c975[1] < 0 else ("MORE recall margin than M2" if c975[0] > 0 else "not separated")
    res[x] = {"mean_d": pt, "ci95": list(map(float, c95)), "ci975": list(map(float, c975)), "verdict": v}
    w(f"  {x} - M2: mean d {pt:+.3f}, 95 % [{c95[0]:+.3f}, {c95[1]:+.3f}], 97.5 % [{c975[0]:+.3f}, {c975[1]:+.3f}] -> {v}")
    for fam, _ in FAM:
        sel = np.array([f == fam for f, _ in ITEMS])
        Mf = M[sel]; bf = np.nanmean(Mf[np.random.default_rng(SEED).integers(0, len(Mf), size=(NB, len(Mf)))], axis=(1, 2))
        w(f"      {fam}: {np.nanmean(Mf):+.3f} [{np.percentile(bf, 2.5):+.3f}, {np.percentile(bf, 97.5):+.3f}] (information)")
gc2 = os.path.join(ROOT, "data", "margins", "gc2")
try:
    diffs = []
    for d in DEPTHS:
        for fam, n in FAM:
            rs = load(os.path.join(gc2, f"d-{d}-{fam}.jsonl"))
            for j, r in enumerate(rs):
                if r["item"] == R["m2"][(fam, j, d)]["item"]:
                    diffs.append(abs(cap(r) - cap(R["m2"][(fam, j, d)])))
    w(f"  information: M2 here vs G-C2 arm (d) (ad657a606, box): {len(diffs)} units, mean |diff| {np.mean(diffs):.4f}, "
      f"max {np.max(diffs):.4f}, equal {sum(1 for x in diffs if x == 0)}")
except (OSError, KeyError) as e:
    w(f"  information: G-C2 arm (d) comparison not available ({e})")
t = "\n".join(out)
print(t)
os.makedirs(RES, exist_ok=True)
with open(os.path.join(RES, "w2d-margin-result.txt"), "w", encoding="utf-8", newline="\n") as f:
    f.write(t + "\n")
with open(os.path.join(RES, "w2d-margin-result.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(res, f, indent=1)
