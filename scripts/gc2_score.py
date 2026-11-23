"""G-C2 scorer (docs/paper/GAPS.md # G-C2; statistic and lines written before any run). Deterministic.
    python -I -X utf8 tools/gc2/gc2_score.py      -> logs/gc2/gc2-result.txt

Data (Machine change, 16:2x): every statistic uses the box records logs/gc2/box/out/<arm>-<depth>-<family>.jsonl
(arms a-d, 117000 / 245000, needles 24 + copies 20) and the arm-(a) 117k repeat logs/gc2/box/out/repeat/. The two rig
records logs/gc2/a-117000-{needle,copy}.jsonl are a rig-vs-box cross-check of arm (a) ONLY: printed, used in no
statistic.

Pairing (replaces the item-name keys of the 15:1x draft of this scorer, which collapsed repeated copy items):
- A unit is (family, slot, depth). Needle slot k = "vault k". Copy slot j = the j-th copy request of the bench at that
  depth (ink_margin.copy_items: the candidate nearest to position (j + 0.5) / 20 of the context). The bench picks the
  copy functions from the corpus slice of each depth, so copy slot j is a different function at 117k and at 245k, and
  the same function can fill two or more adjacent slots of one depth (the same request repeated; after the first, it
  runs on the cached path). Units pair across arms by request order (every arm sent the same requests in the same
  order; asserted below on the item names).
- An "item" for the bootstrap = (family, slot): 24 needles + 20 copy slots = 44 items, each with its 2 depths
  (88 units). Resampling an item moves both of its depths together.

Statistic (GAPS): m = min margin over the answer tokens, capped at 5 nats. d = m(X) - m(a) per unit; mean d over the 88
units; 95 % percentile bootstrap resampling the 44 items, 10,000 resamples, numpy default_rng(20261122). ONE index
matrix (10,000 x 44) is drawn once and shared by every statistic and arm, so all intervals use the same resamples.
Secondary, same resamples: copy log-probability difference (sum log p(correct) over the answer tokens; mean over the
copy units in the resample, 20 copy slots x 2 depths), and the count of units with m < 0 per arm (difference to a).
Noise floor (added 15:1x, before any result): mean |m(repeat) - m(a)| over the repeat units (117k: 24 needles + copy
slots 0-9, same order). A line is MET only if its interval also clears that floor ("below 0" -> "below -floor").
"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
L = os.path.join(ROOT, "data", "margins", "gc2")
BOX = L
OUT = os.path.join(ROOT, "results", "gc2-result.txt")
CAP, DELTA, SEED, NB = 5.0, 0.25, 20261122, 10000
ARMS = {"a": "release v1.0: q8_0 global + f16 SWA", "b": "blanket q4_0 (all 48 caches)",
        "c": "blanket q8_0 (all 48 caches)", "d": "new default: q4_0 global + f16 SWA"}
DEPTHS = (117000, 245000)
FAM = (("needle", 24), ("copy", 20))


def load(path):
    with open(path, encoding="utf-8") as f:
        return [r for r in (json.loads(x) for x in f if x.strip()) if "item" in r]


def cap(r):
    return min(r["min"], CAP)


# ---- load and check ------------------------------------------------------------------------------------------
R = {}  # R[arm][(family, slot, depth)] = record
NAMES = {}
for arm in ARMS:
    R[arm] = {}
    for depth in DEPTHS:
        for fam, n in FAM:
            rs = load(os.path.join(BOX, f"{arm}-{depth}-{fam}.jsonl"))
            assert len(rs) == n, (arm, depth, fam, len(rs))
            names = [r["item"] for r in rs]
            ref = NAMES.setdefault((fam, depth), names)
            assert names == ref, f"request order differs: {arm} {depth} {fam}"
            for j, r in enumerate(rs):
                assert r["depth"] == depth and r["family"] == fam and r["method"] == "gf"
                R[arm][(fam, j, depth)] = r
ITEMS = [(fam, j) for fam, n in FAM for j in range(n)]  # 44
UNITS = [(fam, j, d) for fam, j in ITEMS for d in DEPTHS]  # 88
for arm in ARMS:
    assert set(R[arm]) == set(UNITS)

rng = np.random.default_rng(SEED)
IDX = rng.integers(0, len(ITEMS), size=(NB, len(ITEMS)))  # shared resamples


def per_item(f, fams=("needle", "copy")):
    """value matrix V[item, depth] (nan where the item's family is not selected)."""
    V = np.full((len(ITEMS), len(DEPTHS)), np.nan)
    for i, (fam, j) in enumerate(ITEMS):
        if fam in fams:
            for k, d in enumerate(DEPTHS):
                V[i, k] = f((fam, j, d))
    return V


def mean_ci(V):
    """point mean over the selected units and the 95 % bootstrap interval over the shared item resamples."""
    point = float(np.nanmean(V))
    S = V[IDX]  # NB x 44 x 2
    s = np.nansum(S, axis=(1, 2))
    c = np.sum(~np.isnan(S), axis=(1, 2))
    b = s / c
    lo, hi = np.percentile(b, [2.5, 97.5])
    return point, float(lo), float(hi)


out = []
w = out.append
w("G-C2 result (scripts/gc2_score.py; evidence/LINES.md # 4). One rented box, 8x RTX 4070, engine ad657a606 (b10935), method gf.")
w(f"Capped min margin m (cap {CAP} nats). Bootstrap: {NB} resamples of the 44 items, seed {SEED}, percentile 95 %.")
w("")

# ---- per arm summary -------------------------------------------------------------------------------------------
w("== Per arm and depth (88 units per arm) ==")
for arm, desc in ARMS.items():
    for d in DEPTHS:
        for fam, n in FAM:
            rs = [R[arm][(fam, j, d)] for j in range(n)]
            neg = [f"{fam[0]}{j}:{r['item']}({r['min']:+.3f})" for j, r in enumerate(rs) if r["min"] < 0]
            w(f"  {arm} {d} {fam:6s}: mean capped m {np.mean([cap(r) for r in rs]):.3f}; min m {min(r['min'] for r in rs):+.3f}; "
              f"units m<0: {len(neg)} {neg}; tokens<0 {sum(r['n_neg'] for r in rs)}; "
              f"bound flags {sum(r['below_top'] for r in rs)}; wrong-in-top20 {sum(r['positions_with_wrong_in_top20'] for r in rs)}"
              f"/{sum(r['answer_tokens'] for r in rs)}")
    w(f"  {arm} ({desc}): units with m < 0: {sum(R[arm][u]['min'] < 0 for u in UNITS)} of 88")
w("")

# ---- noise floor -----------------------------------------------------------------------------------------------
REP = {}
for fam, fn in (("needle", "a-117000-needle.jsonl"), ("copy", "a-117000-copy-0-10.jsonl")):
    rs = load(os.path.join(BOX, "repeat", fn))
    for j, r in enumerate(rs):
        assert r["item"] == NAMES[(fam, 117000)][j], ("repeat order", fam, j)
        REP[(fam, j, 117000)] = r
rd = {u: cap(REP[u]) - cap(R["a"][u]) for u in REP}
absd = np.array([abs(v) for v in rd.values()])
NOISE = float(absd.mean())
w("== Noise floor: arm (a) repeated at 117k on the same box (24 needles + copy slots 0-9, same order) ==")
w(f"  units {len(rd)}; mean |repeat difference| of m = {NOISE:.3f} nats; max {absd.max():.3f}; "
  f"needles mean {np.mean([abs(v) for u, v in rd.items() if u[0] == 'needle']):.3f}, "
  f"copies mean {np.mean([abs(v) for u, v in rd.items() if u[0] == 'copy']):.3f}; "
  f"units exactly equal {int(np.sum(absd == 0))}")
w("  copy repeat per slot (m repeat - m a): " + ", ".join(
    f"{j}:{R['a'][('copy', j, 117000)]['item']} {rd[('copy', j, 117000)]:+.3f}" for j in range(10)))
nz = [f"{u[0][0]}{u[1]} {v:+.3f}" for u, v in rd.items() if u[0] == "needle" and v != 0]
w(f"  needle repeat differences not 0: {nz if nz else 'none'}")
w("")

# ---- rig vs box cross-check (arm a, 117k) --------------------------------------------------------------------
w("== Rig vs box cross-check of arm (a) at 117k (information only; used in no statistic) ==")
for fam, fn in (("needle", "a-117000-needle.jsonl"), ("copy", "a-117000-copy.jsonl")):
    rs = load(os.path.join(L, "rig", fn))
    assert [r["item"] for r in rs] == NAMES[(fam, 117000)], ("rig order", fam)
    diffs = [rs[j]["min"] - R["a"][(fam, j, 117000)]["min"] for j in range(len(rs))]
    cd = [cap(rs[j]) - cap(R["a"][(fam, j, 117000)]) for j in range(len(rs))]
    w(f"  {fam}: {len(rs)} items; rig - box min margin: mean |diff| {np.mean(np.abs(diffs)):.4f}, max |diff| "
      f"{np.max(np.abs(diffs)):.4f}; capped mean |diff| {np.mean(np.abs(cd)):.4f}; exactly equal {sum(x == 0 for x in diffs)}")
    w("    per item (rig - box): " + ", ".join(f"{j}:{rs[j]['item']} {x:+.4f}" for j, x in enumerate(diffs)))
w("")

# ---- primary and secondary statistics -------------------------------------------------------------------------
w("== Paired vs arm (a): d = m(X) - m(a) over 88 units; interval over the 44 items ==")
RES = {}
for arm in ("b", "c", "d"):
    D = per_item(lambda u: cap(R[arm][u]) - cap(R["a"][u]))
    p, lo, hi = mean_ci(D)
    RES[arm] = (p, lo, hi)
    sd = float(np.std(D.mean(axis=1), ddof=1))
    C = per_item(lambda u: R[arm][u]["logp_sum"] - R["a"][u]["logp_sum"], fams=("copy",))
    cp, clo, chi = mean_ci(C)
    N = per_item(lambda u: float(R[arm][u]["min"] < 0) - float(R["a"][u]["min"] < 0))
    S = N[IDX]
    nb = np.sum(S, axis=(1, 2)) * 88.0 / np.sum(~np.isnan(S), axis=(1, 2))
    nlo, nhi = np.percentile(nb, [2.5, 97.5])
    nn = per_item(lambda u: float(R[arm][u]["min"] < 0))
    na = per_item(lambda u: float(R["a"][u]["min"] < 0))
    flips = [f"{u[0]} slot {u[1]} {R[arm][u]['item']} @{u[2]} ({R['a'][u]['min']:+.3f} -> {R[arm][u]['min']:+.3f})"
             for u in UNITS if R[arm][u]["min"] < 0 <= R["a"][u]["min"]]
    back = [f"{u[0]} slot {u[1]} {R[arm][u]['item']} @{u[2]} ({R['a'][u]['min']:+.3f} -> {R[arm][u]['min']:+.3f})"
            for u in UNITS if R["a"][u]["min"] < 0 <= R[arm][u]["min"]]
    nd = per_item(lambda u: cap(R[arm][u]) - cap(R["a"][u]), fams=("needle",))
    cd = per_item(lambda u: cap(R[arm][u]) - cap(R["a"][u]), fams=("copy",))
    w(f"  {arm} ({ARMS[arm]}) vs a:")
    w(f"    PRIMARY mean d = {p:+.3f} nats, 95 % [{lo:+.3f}, {hi:+.3f}]  (SD of per-item d {sd:.3f})")
    w(f"    by family (information): needles {np.nanmean(nd):+.3f} [{mean_ci(nd)[1]:+.3f}, {mean_ci(nd)[2]:+.3f}], "
      f"copies {np.nanmean(cd):+.3f} [{mean_ci(cd)[1]:+.3f}, {mean_ci(cd)[2]:+.3f}]; "
      f"by depth: 117k {np.nanmean(D[:, 0]):+.3f}, 245k {np.nanmean(D[:, 1]):+.3f}")
    w(f"    secondary: copy log-prob difference per copy unit {cp:+.3f} nats [{clo:+.3f}, {chi:+.3f}] (40 units)")
    w(f"    secondary: units with m < 0: {int(np.nansum(nn))} vs a {int(np.nansum(na))} of 88; difference "
      f"{int(np.nansum(N)):+d} [{nlo:+.1f}, {nhi:+.1f}]")
    w(f"    m < 0 in {arm} but >= 0 in a: {flips if flips else 'none'}")
    w(f"    m < 0 in a but >= 0 in {arm}: {back if back else 'none'}")
w("")

# ---- sensitivity (information, no line): repeated copy requests counted once ----------------------------------
w("== Sensitivity (information only, no line): repeated copy requests of one depth counted once ==")
w("  Adjacent copy slots that hold the same function send the same request; after the first it runs on the cached")
w("  path. Here each (depth, function) keeps its first slot only; the needles are unchanged. Bootstrap: units")
w("  resampled as clusters by (family, slot), the same seed.")
keep = set()
for d in DEPTHS:
    seen = set()
    for j, nm in enumerate(NAMES[("copy", d)]):
        if nm not in seen:
            keep.add(("copy", j, d)); seen.add(nm)
    for j in range(24):
        keep.add(("needle", j, d))
w(f"  distinct copy functions: 117k {len(set(NAMES[('copy', 117000)]))} of 20, 245k {len(set(NAMES[('copy', 245000)]))} of 20; units kept {len(keep)}")
for arm in ("b", "c", "d"):
    D = per_item(lambda u: (cap(R[arm][u]) - cap(R["a"][u])) if u in keep else np.nan)
    p, lo, hi = mean_ci(D)
    w(f"  {arm} vs a: mean d {p:+.3f} [{lo:+.3f}, {hi:+.3f}]")
w("")

# ---- lines -----------------------------------------------------------------------------------------------------
w("== Lines (as written before any run; noise floor applied as GAPS # Smoke finding says) ==")
bl, cl = RES["b"], RES["c"]
b_below, c_below = bl[2] < 0, cl[2] < 0
b_clear, c_clear = bl[2] < -NOISE, cl[2] < -NOISE
met1 = b_clear or c_clear
w(f"  L1 'the per-class choice matters for recall (graded)': needs the interval of (b) or (c) vs (a) entirely below 0"
  f" AND below -{NOISE:.3f} (noise floor).")
w(f"     (b) [{bl[1]:+.3f}, {bl[2]:+.3f}]: below 0 {b_below}, below -floor {b_clear}; "
  f"(c) [{cl[1]:+.3f}, {cl[2]:+.3f}]: below 0 {c_below}, below -floor {c_clear}")
w("     -> " + ("MET" if met1 else "NOT MET: even graded recall margins do not separate the blanket settings from the "
                                    "per-class one at this resolution"))
dp, dlo, dhi = RES["d"]
if dhi < -NOISE:
    v2 = "COSTS RECALL MARGIN"
elif dhi < 0:
    v2 = "INCONCLUSIVE (interval below 0 but does not clear the noise floor)"
elif dlo >= -DELTA:
    v2 = "NO MEASURABLE MARGIN LOSS"
else:
    v2 = "INCONCLUSIVE"
w(f"  L2 (d) vs (a): 'costs' if the interval is below 0 (and below -floor); 'no measurable margin loss' only if the "
  f"interval lies inside [-{DELTA}, +inf); else 'inconclusive'.")
w(f"     (d) [{dlo:+.3f}, {dhi:+.3f}], point {dp:+.3f} -> {v2}")
if NOISE > DELTA:
    w(f"     note: the noise floor ({NOISE:.3f}) is larger than delta ({DELTA}); single-run noise per unit exceeds the "
      f"non-inferiority margin, so the mean-level verdict rests on averaging over 88 units.")
w("  L3 no line on the copy log-probability (reported above); no 64k depth (not run).")

t = "\n".join(out)
print(t)
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    f.write(t + "\n")
