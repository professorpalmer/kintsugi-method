"""W2-C scorer (evidence/LINES.md section 6; lines written before any build or run). Deterministic.
    python -X utf8 scripts/w2c_score.py --dir data/w2c/out [--out results/w2c-result.txt]
Arms W2Cg1..W2Cg4 (V5c's kmap with group g set back to V3s2) from --dir; shipped = B1's run (data/b1/out, RTX 4070);
V5c = B1c's same-box run (data/b1c/out, RTX 4070). B1's statistic unchanged (b1_score.py's load
and constants; per_doc / dvals / contrast as there): d = m(map) - m(shipped), m = min margin capped at 5; L (117k +
245k items), S (their twins), C = L - S; bootstrap over documents, 10,000 resamples, default_rng(20261122).
Run floor = mean |m(repeat) - m(B1 shipped)| over the W2-C box's shipped doc-0 repeat (<dir>/repeat/shipped-doc00.jsonl).
F_L = max(run floor, 0.212), F_C = max(run floor, 0.295). R_g = L(M_g) - L(V5c), item-paired.
Line (98.75 % intervals = Bonferroni over 4 maps): "group g carries the held-out recall loss" only if (i) the interval
of L(M_g) does NOT lie entirely below -F_L, AND (ii) the interval of R_g lies entirely above 0.
Reported: B1's line (a)-(c) per map at 99.72 %, R_g / |L(V5c)|, needle / copy split."""
import argparse, glob, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "b1"))
from b1_score import CAP, SEED, NB, LONG, load  # noqa: E402

MAPS = ["W2Cg1", "W2Cg2", "W2Cg3", "W2Cg4"]
GROUP = {"W2Cg1": "G1 attn_qkv (48, K3 -> K2)", "W2Cg2": "G2 attn_output (12, K2 -> K3)", "W2Cg3": "G3 ffn_down (16, K2 -> K3)",
         "W2Cg4": "G4 ffn_gate (9, K2 -> K3)"}
A4 = 0.05 / 4
B1A = 0.05 / 3 / 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "w2c-result.txt"))
    a = ap.parse_args()
    R = {}
    for h in range(20):
        R[("shipped", h)] = load(os.path.join(ROOT, "data", "b1", "out", f"shipped-doc{h:02d}.jsonl"))
        R[("V5c", h)] = load(os.path.join(ROOT, "data", "b1c", "out", f"V5c-doc{h:02d}.jsonl"))
        for m in MAPS:
            p = os.path.join(a.dir, f"{m}-doc{h:02d}.jsonl")
            if os.path.exists(p):
                R[(m, h)] = load(p)
    docs = [h for h in range(20) if all((x, h) in R and set(R[(x, h)]) == set(R[("shipped", h)]) for x in MAPS + ["V5c"])]
    out = [f"W2-C result (scripts/w2c_score.py; evidence/LINES.md section 6). Complete documents: {len(docs)} of 20 {docs}"]
    w = out.append
    if len(docs) < 12:
        w("NO VERDICT: fewer than 12 complete documents."); return fin(out, a.out)
    rng = np.random.default_rng(SEED)
    IDX = rng.integers(0, len(docs), size=(NB, len(docs)))
    cap = lambda r: min(r["min"], CAP)
    is_long = lambda k: k[0] == "item" and str(k[1][0]) in LONG
    is_twin = lambda k: k[0] == "twin"

    def per_doc(f, plo, phi):
        s = np.array([sum(f(h)) for h in docs], dtype=float)
        c = np.array([len(list(f(h))) for h in docs], dtype=float)
        b = s[IDX].sum(axis=1) / c[IDX].sum(axis=1)
        return s.sum() / c.sum(), np.percentile(b, [plo, phi])

    def dv(x, y, h, sel):
        X, Y = R[(x, h)], R[(y, h)]
        return [cap(X[k]) - cap(Y[k]) for k in sorted(R[("shipped", h)]) if sel(k)]

    def contrast(x, h):
        X, S = R[(x, h)], R[("shipped", h)]
        v = []
        for k in sorted(S):
            if is_long(k):
                t = ("twin", (k[1][0], k[1][1], k[1][2]))
                v.append((cap(X[k]) - cap(S[k])) - (cap(X[t]) - cap(S[t])))
        return v

    rep_p = os.path.join(a.dir, "repeat", "shipped-doc00.jsonl")
    if not os.path.exists(rep_p):
        w("run floor: repeat not landed -> NO VERDICT"); return fin(out, a.out)
    rep = load(rep_p)
    ks = [k for k in rep if k in R[("shipped", 0)]]
    rf = float(np.mean([abs(cap(rep[k]) - cap(R[("shipped", 0)][k])) for k in ks]))
    nz = sum(abs(rep[k]["min"] - R[("shipped", 0)][k]["min"]) > 1e-9 for k in ks)
    FL, FC = max(rf, 0.212), max(rf, 0.295)
    w(f"run floor (W2-C box repeat vs the B1 run): {len(ks)} records, mean |d| {rf:.4f}, raw mins that differ {nz}; F_L {FL:.3f}, F_C {FC:.3f}")
    lo4, hi4 = 100 * A4 / 2, 100 * (1 - A4 / 2)
    lo1, hi1 = 100 * B1A / 2, 100 * (1 - B1A / 2)
    Lv = per_doc(lambda h: dv("V5c", "shipped", h, is_long), lo4, hi4)
    w(f"V5c (B1c same-box) L {Lv[0]:+.3f} [{Lv[1][0]:+.3f}, {Lv[1][1]:+.3f}] (98.75 %)")
    carriers = []
    for m in MAPS:
        L4 = per_doc(lambda h: dv(m, "shipped", h, is_long), lo4, hi4)
        Rg = per_doc(lambda h: dv(m, "V5c", h, is_long), lo4, hi4)
        L1 = per_doc(lambda h: dv(m, "shipped", h, is_long), lo1, hi1)
        S1 = per_doc(lambda h: dv(m, "shipped", h, is_twin), lo1, hi1)
        C1 = per_doc(lambda h: contrast(m, h), lo1, hi1)
        nd = per_doc(lambda h: dv(m, "shipped", h, lambda k: is_long(k) and k[1][1] == "needle"), lo4, hi4)
        cp = per_doc(lambda h: dv(m, "shipped", h, lambda k: is_long(k) and k[1][1] == "copy"), lo4, hi4)
        ci, cii = not (L4[1][1] < -FL), Rg[1][0] > 0
        if ci and cii:
            carriers.append(m)
        w(f"\n{m} = V5c with {GROUP[m]} set back to V3s2:")
        w(f"  L {L4[0]:+.3f} [{L4[1][0]:+.3f}, {L4[1][1]:+.3f}] (98.75 %); R_g = L - L(V5c) {Rg[0]:+.3f} [{Rg[1][0]:+.3f}, {Rg[1][1]:+.3f}]; "
          f"recovery R_g / |L(V5c)| {Rg[0] / abs(Lv[0]):.2f}")
        w(f"  long needles {nd[0]:+.3f}, long copies {cp[0]:+.3f}")
        w(f"  B1 line at 99.72 %: L {L1[0]:+.3f} [{L1[1][0]:+.3f}, {L1[1][1]:+.3f}], S {S1[0]:+.3f} [{S1[1][0]:+.3f}, {S1[1][1]:+.3f}], "
          f"C {C1[0]:+.3f} [{C1[1][0]:+.3f}, {C1[1][1]:+.3f}]; (a) {L1[1][1] < -FL}, (b) {C1[1][1] < -FC}, (c) {not (S1[1][1] < -FL)}")
        w(f"  W2-C line: (i) L not entirely below -F_L {ci}; (ii) R_g entirely above 0 {cii} -> "
          + ("CARRIES the held-out recall loss" if ci and cii else "does not carry it alone"))
    w("")
    w("W2-C LINE: " + (f"the held-out recall loss is carried by {', '.join(GROUP[m] for m in carriers)}" if carriers
                       else "NOT CARRIED BY ONE GROUP"))
    return fin(out, a.out)


def fin(out, path):
    t = "\n".join(out)
    print(t)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8", newline="\n").write(t + "\n")


if __name__ == "__main__":
    main()
