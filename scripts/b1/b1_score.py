"""B1 scorer (docs/paper/GAPS.md # B1; statistic and lines written before any run). Deterministic.
    python -I -X utf8 tools/b1/b1_score.py --dir <box out dir> [--out logs/b1/b1-result.txt]

Input: <dir>/<arm>-doc<HH>.jsonl (tools/b1/b1_probe.py, method gf), arms below; <dir>/repeat/shipped-doc00.jsonl (the
run-noise repeat). A document counts only if every arm has all of its records (24 depth items + its twins).

Statistic: m = min margin over the answer tokens, capped at 5 nats. d = m(X) - m(shipped), paired by (doc, depth,
family, slot); twins paired by the item they copy. Per arm X:
  L = mean d over the items at 117,000 and 245,000 (the long items);
  S = mean d over their twins (the same items on a ~3.9k-token haystack: the shallow control);
  C = mean over the long items of (d(item) - d(its twin)) = L - S (the depth contrast: long-range damage);
  64k mean d reported, no line.
Bootstrap: resample the documents (every item and twin of a document together), 10,000 resamples, numpy
default_rng(20261122); one index matrix shared by every arm and statistic; percentile intervals.
Multiplicity: B1 is one of the three tests of the STOP RULE (family alpha 0.0167). Inside B1, Bonferroni over the six
lower-KL maps: alpha per map 0.0167 / 6 = 0.00278, so every line reads a two-sided 99.72 % interval (percentiles
0.139 and 99.861). Requiring (a), (b) and (c) together for one map needs no further correction.
Floors: run floor = mean |m(repeat) - m(first run)| over the repeat ticket; refit floor = |L| and |C| of the refit arm
(V2body: the same bit layout with another fit). F_L = max(run floor, |L(V2body)|), F_C = max(run floor, |C(V2body)|).
Line "map X loses recall at depth": (a) the interval of L lies entirely below -F_L, AND (b) the interval of C lies
entirely below -F_C, AND (c) the interval of S does NOT lie entirely below -F_L (no loss shown at the shallow control).
B1 PASSES (STOP RULE test 1; framing B for this claim) only if >= 1 of the six lower-KL maps meets the line.
Otherwise B1 does not pass. H1 (higher KL) and V2body are reported, not counted.
"""
import argparse
import glob
import json
import os

import numpy as np

ARMS = ["shipped", "V3s2", "V5a", "V5c", "V5bc", "V5tc", "U1", "H1", "V2body"]
LOWER_KL = ["V3s2", "V5a", "V5c", "V5bc", "V5tc", "U1"]
REFIT = "V2body"
CAP, SEED, NB = 5.0, 20261122, 10000
FAMILY_ALPHA = 0.05 / 3
ALPHA = FAMILY_ALPHA / len(LOWER_KL)
PLO, PHI = 100 * ALPHA / 2, 100 * (1 - ALPHA / 2)
LONG = ("117000", "245000")
MIN_DOCS = 12


def key(r):
    return ("twin", tuple(r["of"])) if r["depth"] == "twin" else ("item", (int(r["depth"]), r["family"], r["slot"]))


def load(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for x in f:
            r = json.loads(x)
            if "min" in r:
                out[key(r)] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "b1-result.txt"))
    a = ap.parse_args()
    R = {}
    for arm in ARMS:
        for p in sorted(glob.glob(os.path.join(a.dir, f"{arm}-doc[0-9][0-9].jsonl"))):
            R[(arm, int(p[-8:-6]))] = load(p)
    docs_all = sorted({h for _, h in R})
    ref_keys = {h: set(R[("shipped", h)]) for h in docs_all if ("shipped", h) in R}
    docs = [h for h in docs_all if h in ref_keys and len(ref_keys[h]) >= 24 and
            all((arm, h) in R and set(R[(arm, h)]) == ref_keys[h] for arm in ARMS)]
    out = []
    w = out.append
    w(f"B1 result (tools/b1/b1_score.py; GAPS.md # B1). Complete documents: {len(docs)} of {len(docs_all)} {docs}")
    w(f"Cap {CAP} nats; bootstrap over documents, {NB} resamples, seed {SEED}; line intervals at {100 * (1 - ALPHA):.2f} % "
      f"(alpha {ALPHA:.5f} = family {FAMILY_ALPHA:.4f} / {len(LOWER_KL)} maps); 95 % intervals for information.")
    if len(docs) < MIN_DOCS:
        w(f"NO VERDICT: fewer than {MIN_DOCS} complete documents. B1 does not pass.")
        return finish(out, a.out)
    rng = np.random.default_rng(SEED)
    IDX = rng.integers(0, len(docs), size=(NB, len(docs)))

    def cap(r):
        return min(r["min"], CAP)

    def per_doc(f):
        """list over docs of (sum, count) arrays -> bootstrap means."""
        s = np.array([sum(v for v in f(h)) for h in docs], dtype=float)
        c = np.array([len(list(f(h))) for h in docs], dtype=float)
        pt = s.sum() / c.sum()
        b = s[IDX].sum(axis=1) / c[IDX].sum(axis=1)
        return pt, np.percentile(b, [PLO, PHI]), np.percentile(b, [2.5, 97.5])

    def dvals(arm, h, sel):
        X, S = R[(arm, h)], R[("shipped", h)]
        return [cap(X[k]) - cap(S[k]) for k in sorted(S) if sel(k)]

    is_long = lambda k: k[0] == "item" and str(k[1][0]) in LONG
    is_64 = lambda k: k[0] == "item" and k[1][0] == 64000
    is_twin = lambda k: k[0] == "twin"

    def contrast(arm, h):
        X, S = R[(arm, h)], R[("shipped", h)]
        v = []
        for k in sorted(S):
            if is_long(k):
                t = ("twin", (k[1][0], k[1][1], k[1][2]))
                if t in S:
                    v.append((cap(X[k]) - cap(S[k])) - (cap(X[t]) - cap(S[t])))
        return v

    # run floor
    rep_p = os.path.join(a.dir, "repeat", "shipped-doc00.jsonl")
    run_floor = None
    if os.path.exists(rep_p) and ("shipped", 0) in R:
        rep = load(rep_p)
        ks = [k for k in rep if k in R[("shipped", 0)]]
        run_floor = float(np.mean([abs(cap(rep[k]) - cap(R[("shipped", 0)][k])) for k in ks])) if ks else None
        w(f"run floor: shipped doc 0 repeated, {len(ks)} records, mean |difference| {run_floor}")
    if run_floor is None:
        w("run floor: repeat not landed -> NO VERDICT (the line needs it). B1 does not pass.")
        return finish(out, a.out)
    res = {}
    w("")
    w("== Per arm vs shipped (point, line interval, 95 % interval) ==")
    for arm in ARMS[1:]:
        L = per_doc(lambda h: dvals(arm, h, is_long))
        S_ = per_doc(lambda h: dvals(arm, h, is_twin))
        C = per_doc(lambda h: contrast(arm, h))
        M64 = per_doc(lambda h: dvals(arm, h, is_64))
        res[arm] = (L, S_, C)
        fmt = lambda t: f"{t[0]:+.3f} [{t[1][0]:+.3f}, {t[1][1]:+.3f}] (95 % [{t[2][0]:+.3f}, {t[2][1]:+.3f}])"
        w(f"  {arm}: L {fmt(L)}; S {fmt(S_)}; C {fmt(C)}; 64k {M64[0]:+.3f}")
        neg = {}
        for h in docs:
            for k, r in R[(arm, h)].items():
                g = "twin" if k[0] == "twin" else str(k[1][0])
                neg[g] = neg.get(g, 0) + (r["min"] < 0)
        negs = {}
        for h in docs:
            for k, r in R[("shipped", h)].items():
                g = "twin" if k[0] == "twin" else str(k[1][0])
                negs[g] = negs.get(g, 0) + (r["min"] < 0)
        cp = per_doc(lambda h: [R[(arm, h)][k]["logp_sum"] - R[("shipped", h)][k]["logp_sum"]
                                for k in sorted(R[("shipped", h)]) if is_long(k) and k[1][1] == "copy"])
        w(f"     items m < 0 by depth: {arm} {dict(sorted(neg.items()))} vs shipped {dict(sorted(negs.items()))}; "
          f"long copies log-prob diff {cp[0]:+.3f} (95 % [{cp[2][0]:+.3f}, {cp[2][1]:+.3f}])")
    FL = max(run_floor, abs(res[REFIT][0][0]))
    FC = max(run_floor, abs(res[REFIT][2][0]))
    w("")
    w(f"Floors: F_L = max(run {run_floor:.3f}, |L(V2body)| {abs(res[REFIT][0][0]):.3f}) = {FL:.3f}; "
      f"F_C = max(run {run_floor:.3f}, |C(V2body)| {abs(res[REFIT][2][0]):.3f}) = {FC:.3f}")
    w("")
    w("== Line per map (a) L hi < -F_L, (b) C hi < -F_C, (c) NOT S hi < -F_L ==")
    passed = []
    for arm in ARMS[1:]:
        L, S_, C = res[arm]
        ca, cb, cc = L[1][1] < -FL, C[1][1] < -FC, not (S_[1][1] < -FL)
        met = ca and cb and cc
        tag = "counted" if arm in LOWER_KL else "reported only"
        w(f"  {arm} ({tag}): (a) {ca}, (b) {cb}, (c) {cc} -> {'LOSES RECALL AT DEPTH' if met else 'not shown'}")
        if met and arm in LOWER_KL:
            passed.append(arm)
    w("")
    w("B1 LINE: " + (f"PASSES: lower-KL map(s) that lose recall at depth but not at the shallow control: {passed}"
                     if passed else "DOES NOT PASS: no lower-KL map loses recall at depth by the line"))
    return finish(out, a.out)


def finish(out, path):
    t = "\n".join(out)
    print(t)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(t + "\n")


if __name__ == "__main__":
    main()
