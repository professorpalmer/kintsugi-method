"""B1c scorer (docs/paper/GAPS.md # B1c, line written before any B1c run). Deterministic.
    python -I -X utf8 tools/b1/b1c_score.py --dir logs/b1c/box/out [--out logs/b1/b1c-result.txt]

The B1 statistic of tools/b1/b1_score.py, unchanged, for the two B1c arms only (shipped and V5c on ONE box):
m = min margin capped at 5 nats; d = m(V5c) - m(shipped) paired by (doc, depth, family, slot), twins by their item;
L = mean d over the 117k + 245k items, S = mean d over their twins, C = mean over the long items of (d(item) - d(twin));
bootstrap over documents, 10,000 resamples, numpy default_rng(20261122), percentile intervals at 99.72 % (the B1
alpha 0.0167 / 6, kept as written) and 95 % for information. The functions load / key and the constants are imported
from b1_score.py; per_doc / dvals / contrast are the same code.
Floors (B1c line): run floor = mean |m(repeat) - m(first run)| over the shipped doc-0 repeat on another GPU of the same
box; F_L = max(run floor, 0.212), F_C = max(run floor, 0.295) (the B1 refit floors |L(V2body)|, |C(V2body)|).
Line: B1c CONFIRMS the B1 pass only if L hi < -F_L AND C hi < -F_C AND NOT (S hi < -F_L).
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b1_score import CAP, SEED, NB, PLO, PHI, ALPHA, LONG, MIN_DOCS, load  # noqa: E402

ARMS = ["shipped", "V5c"]
REFIT_L, REFIT_C = 0.212, 0.295   # B1 RESULT: |L(V2body)|, |C(V2body)| (logs/b1/b1-result.txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "b1c-result.txt"))
    a = ap.parse_args()
    R = {}
    for arm in ARMS:
        for p in sorted(glob.glob(os.path.join(a.dir, f"{arm}-doc[0-9][0-9].jsonl"))):
            R[(arm, int(p[-8:-6]))] = load(p)
    docs_all = sorted({h for _, h in R})
    docs = [h for h in docs_all if ("shipped", h) in R and len(R[("shipped", h)]) >= 24 and
            all((arm, h) in R and set(R[(arm, h)]) == set(R[("shipped", h)]) for arm in ARMS)]
    out = []
    w = out.append
    w(f"B1c result (tools/b1/b1c_score.py; GAPS.md # B1c). Complete documents: {len(docs)} of {len(docs_all)} {docs}")
    w(f"Cap {CAP} nats; bootstrap over documents, {NB} resamples, seed {SEED}; line intervals at {100 * (1 - ALPHA):.2f} %; "
      f"95 % for information.")
    if len(docs) < MIN_DOCS:
        w(f"NO VERDICT: fewer than {MIN_DOCS} complete documents.")
        return finish(out, a.out)
    rng = np.random.default_rng(SEED)
    IDX = rng.integers(0, len(docs), size=(NB, len(docs)))
    cap = lambda r: min(r["min"], CAP)

    def per_doc(f):
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

    rep_p = os.path.join(a.dir, "repeat", "shipped-doc00.jsonl")
    if not (os.path.exists(rep_p) and ("shipped", 0) in R):
        w("run floor: repeat not landed -> NO VERDICT.")
        return finish(out, a.out)
    rep = load(rep_p)
    ks = [k for k in rep if k in R[("shipped", 0)]]
    run_floor = float(np.mean([abs(cap(rep[k]) - cap(R[("shipped", 0)][k])) for k in ks]))
    nz = sum(abs(rep[k]["min"] - R[("shipped", 0)][k]["min"]) > 1e-9 for k in ks)
    w(f"run floor (same box, another GPU): shipped doc 0 repeated, {len(ks)} records, mean |difference| {run_floor}; "
      f"raw mins that differ: {nz} of {len(ks)}")
    arm = "V5c"
    L = per_doc(lambda h: dvals(arm, h, is_long))
    S_ = per_doc(lambda h: dvals(arm, h, is_twin))
    C = per_doc(lambda h: contrast(arm, h))
    M64 = per_doc(lambda h: dvals(arm, h, is_64))
    fmt = lambda t: f"{t[0]:+.3f} [{t[1][0]:+.3f}, {t[1][1]:+.3f}] (95 % [{t[2][0]:+.3f}, {t[2][1]:+.3f}])"
    w(f"  V5c vs shipped: L {fmt(L)}; S {fmt(S_)}; C {fmt(C)}; 64k {M64[0]:+.3f}")
    neg = {}
    for h in docs:
        for who in ARMS:
            for k, r in R[(who, h)].items():
                g = "twin" if k[0] == "twin" else str(k[1][0])
                neg.setdefault(who, {}).setdefault(g, 0)
                neg[who][g] += r["min"] < 0
    w(f"     items m < 0 by depth: {', '.join(f'{x} {dict(sorted(neg[x].items()))}' for x in ARMS)}")
    for fam in ("needle", "copy"):
        for D in (64000, 117000, 245000):
            v = per_doc(lambda h: dvals(arm, h, lambda k: k[0] == "item" and k[1][0] == D and k[1][1] == fam))
            w(f"     mean d {fam} {D}: {v[0]:+.3f} (95 % [{v[2][0]:+.3f}, {v[2][1]:+.3f}])")
    FL, FC = max(run_floor, REFIT_L), max(run_floor, REFIT_C)
    w(f"Floors: F_L = max(run {run_floor:.3f}, {REFIT_L}) = {FL:.3f}; F_C = max(run {run_floor:.3f}, {REFIT_C}) = {FC:.3f}")
    ca, cb, cc = L[1][1] < -FL, C[1][1] < -FC, not (S_[1][1] < -FL)
    w(f"Line: (a) L hi < -F_L {ca}, (b) C hi < -F_C {cb}, (c) NOT S hi < -F_L {cc}")
    w("B1c LINE: " + ("CONFIRMS the B1 pass: V5c loses recall at depth vs shipped on one box" if ca and cb and cc else
                      "DOES NOT CONFIRM: B1 passed on a cross-box pairing, not confirmed on one box"))
    return finish(out, a.out)


def finish(out, path):
    t = "\n".join(out)
    print(t)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(t + "\n")


if __name__ == "__main__":
    main()
