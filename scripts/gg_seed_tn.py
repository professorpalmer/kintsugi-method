"""G-G2: seed-level re-analysis of the AppWorld test_normal comparisons of paper v1.0.x (evidence/LINES.md section 6;
lines written before any number). CPU only, deterministic. Method = scripts/gg_seed.py part (2), unchanged.
    python -X utf8 scripts/gg_seed_tn.py      -> results/gg-seed-tn.txt, results/gg-seed-tn.json
A file of data/appworld/ without a seed suffix (v3ac, v5bc, v2body) is seed 100."""
import csv, json, math, os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "data", "appworld")
RESULTS = os.path.join(ROOT, "results")
NB, RNG_SEED = 10000, 20261122
PHI_GG = {"27B": 2.75, "Inkling": 1.29}      # G-G (test_challenge / wave 2), results/gg-seed.txt "phiD used"

# arm -> (family, {seed: file stem})
A = {
    "shipped (S)": ("27B", {100: "v3ac", 101: "v3ac-s101"}),
    "V5bc (S)": ("27B", {100: "v5bc", 101: "v5bc-s101"}),
    "V5t (S)": ("27B", {100: "v5t-s100", 101: "v5t-s101"}),
    "V2 body (S)": ("27B", {100: "v2body", 101: "v2body-s101"}),
    "BF16 (S)": ("27B", {100: "bf16o-s100", 101: "bf16o-s101"}),
    "naive (S)": ("27B", {100: "q2kxlo-s100", 101: "q2kxlo-s101"}),
    "shipped (102+103)": ("27B", {102: "kship-s102", 103: "kship-s103"}),
    "weights-only (102+103)": ("27B", {102: "kfw-s102", 103: "kfw-s103"}),
    "BF16 (102+103)": ("27B", {102: "bf16f-s102", 103: "bf16f-s103"}),
    "shipped (104+105)": ("27B", {104: "pv3ac-s104", 105: "pv3ac-s105"}),
    "V5bc (104+105)": ("27B", {104: "pv5bc-s104", 105: "pv5bc-s105"}),
    "V5tc (104+105)": ("27B", {104: "pv5tc-s104", 105: "pv5tc-s105"}),
    "V2 body (104+105)": ("27B", {104: "pv2body-s104", 105: "pv2body-s105"}),
    "BF16 (104+105)": ("27B", {104: "pbf16-s104", 105: "pbf16-s105"}),
    "naive (104+105)": ("27B", {104: "pnaive-s104", 105: "pnaive-s105"}),
    "Inkling Unsloth setup (100+101)": ("Inkling", {100: "ink-s100", 101: "ink-s101"}),
    "Inkling Unsloth setup (102+103)": ("Inkling", {102: "ink-s102", 103: "ink-s103"}),
    "Inkling Unsloth setup (104+105)": ("Inkling", {104: "ink-s104", 105: "ink-s105"}),
    "Inkling Unsloth setup (102-105)": ("Inkling", {s: f"ink-s{s}" for s in (102, 103, 104, 105)}),
    "Inkling M2 (102+103)": ("Inkling", {102: "inkm-s102", 103: "inkm-s103"}),
    "Inkling M2 (104+105)": ("Inkling", {104: "inkm-s104", 105: "inkm-s105"}),
    "Inkling M2 (102-105)": ("Inkling", {s: f"inkm-s{s}" for s in (102, 103, 104, 105)}),
    "Inkling official weights (102-105)": ("Inkling", {s: f"raw-s{s}" for s in (102, 103, 104, 105)}),
    "Inkling M2 4-bit global cache (102-105)": ("Inkling", {s: f"q65-s{s}" for s in (102, 103, 104, 105)}),
}
# (evidence row, label, X, Y, claim): "+" X > Y, "-" X < Y, "" report only (a "level / not shown" row)
COMP = [
    ("E1.8", "KL-only winner vs shipped, selection seeds", "V5bc (S)", "shipped (S)", "-"),
    ("E1.9", "KL-tools vs shipped, selection seeds", "V5t (S)", "shipped (S)", ""),
    ("E3.8", "author's fits vs shipped, selection seeds", "V2 body (S)", "shipped (S)", ""),
    ("E3.6", "shipped vs BF16, selection seeds", "shipped (S)", "BF16 (S)", "+"),
    ("E1.20", "shipped vs naive, selection seeds", "shipped (S)", "naive (S)", "+"),
    ("E3.9", "Kintsugi-27B vs Inkling Unsloth setup, 100+101", "shipped (S)", "Inkling Unsloth setup (100+101)", "+"),
    ("E3.7", "shipped vs BF16, 102+103", "shipped (102+103)", "BF16 (102+103)", ""),
    ("E3.7", "weights-only vs BF16, 102+103", "weights-only (102+103)", "BF16 (102+103)", ""),
    ("E2.14", "shipped vs weights-only, 102+103", "shipped (102+103)", "weights-only (102+103)", ""),
    ("E3.9", "Kintsugi-27B vs Inkling Unsloth setup, 102+103", "shipped (102+103)", "Inkling Unsloth setup (102+103)", ""),
    ("E1.31", "KL-only winner vs shipped, 104+105", "V5bc (104+105)", "shipped (104+105)", ""),
    ("E3.16", "KL-tools vs shipped, 104+105", "V5tc (104+105)", "shipped (104+105)", ""),
    ("E3.16", "author's fits vs shipped, 104+105", "V2 body (104+105)", "shipped (104+105)", ""),
    ("E3.16", "shipped vs BF16, 104+105", "shipped (104+105)", "BF16 (104+105)", ""),
    ("E1.32", "shipped vs naive, 104+105", "shipped (104+105)", "naive (104+105)", "+"),
    ("E3.12", "Inkling M2 setup vs Unsloth setup, 102+103", "Inkling M2 (102+103)", "Inkling Unsloth setup (102+103)", "+"),
    ("E3.13", "Inkling M2 setup vs Unsloth setup, 104+105 (finished pairs)", "Inkling M2 (104+105)", "Inkling Unsloth setup (104+105)", "+"),
    ("E3.12+E3.13", "Inkling M2 setup vs Unsloth setup, 102-105 (finished pairs)", "Inkling M2 (102-105)", "Inkling Unsloth setup (102-105)", "+"),
    ("E4.8", "Inkling M2 vs official weights, 102-105", "Inkling M2 (102-105)", "Inkling official weights (102-105)", ""),
    ("E4.12", "Inkling M2 4-bit global cache vs official weights, 102-105", "Inkling M2 4-bit global cache (102-105)", "Inkling official weights (102-105)", ""),
    ("E4.12", "Inkling M2 4-bit global cache vs 8-bit cache, 102-105", "Inkling M2 4-bit global cache (102-105)", "Inkling M2 (102-105)", ""),
]


def sign_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def norm_p(z):
    return math.erfc(abs(z) / math.sqrt(2))


def t_p(t, df):
    """two-sided p of Student t (df 1 and 3 in closed form; the only cases here)"""
    if df == 1:
        return 1 - 2 * math.atan(abs(t)) / math.pi
    if df == 3:
        x = abs(t) / math.sqrt(3)
        return 1 - 2 / math.pi * (math.atan(x) + x / (1 + x * x))
    raise ValueError(df)


_c = {}


def res(stem):
    if stem not in _c:
        with open(os.path.join(LOGS, f"{stem}.csv"), encoding="utf-8") as fh:
            _c[stem] = {t: int(s) for t, s in list(csv.reader(fh))[1:]}
    return _c[stem]


def pair(x, y, s):
    a, b = res(A[x][1][s]), res(A[y][1][s])
    ids = sorted(set(a) & set(b))
    return np.array([a[t] for t in ids]), np.array([b[t] for t in ids])


def phi_tn(fam):
    """mean h over all arm pairs of the family that share a seed pair (two seeds each)"""
    hs = []
    for sp in ((100, 101), (102, 103), (104, 105)):
        nm = [n for n, (f, ss) in A.items() if f == fam and tuple(sorted(ss)) == sp]
        for i, x in enumerate(nm):
            for y in nm[i + 1:]:
                D, N = [], []
                for s in sp:
                    xv, yv = pair(x, y, s)
                    D.append(int((xv - yv).sum())); N.append(int((xv != yv).sum()))
                if np.mean(N) > 0:
                    hs.append((f"{x} - {y}", float(np.var(D, ddof=1) / np.mean(N))))
    return hs


def main():
    L, J = [], {"phi": {}, "comparisons": []}
    L.append("G-G2 (seed level, test_normal): evidence/LINES.md section 6; CPU, deterministic")
    L.append("\n(1) Design effect on test_normal: h = sample var of D_s over the two seeds of a pair / mean discordant n_s")
    phi = {}
    for fam in ("27B", "Inkling"):
        hs = phi_tn(fam)
        v = [h for _, h in hs]
        m = float(np.mean(v))
        phi[fam] = max(1.0, PHI_GG[fam], m)
        L.append(f"  {fam}: {len(v)} arm pairs, mean h {m:.2f}, median {np.median(v):.2f}, range {min(v):.2f}-{max(v):.2f}; "
                 f"G-G value {PHI_GG[fam]:.2f}; phi used = the larger = {phi[fam]:.2f}")
        J["phi"][fam] = {"pairs": len(v), "mean_h": m, "gg": PHI_GG[fam], "used": phi[fam]}
    L.append("\n(2) Seed-level re-analysis (D = X - Y task passes; V = var of sum D from a within-seed task bootstrap)")
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for ev, lab, x, y, claim in COMP:
        ss = sorted(set(A[x][1]) & set(A[y][1]))
        ph = max(phi[A[x][0]], phi[A[y][0]])
        Ds, per, ntot, a_all, b_all = [], [], 0, 0, 0
        boot = np.zeros(NB)
        for s in ss:
            xv, yv = pair(x, y, s)
            d = xv - yv
            a, b = int((d > 0).sum()), int((d < 0).sum())
            Ds.append(int(d.sum())); ntot += len(d); a_all += a; b_all += b
            per.append(f"s{s} {int(xv.sum())}-{int(yv.sum())}={int(d.sum()):+d} ({a}/{b})")
            boot += np.array([d[rng.integers(0, len(d), len(d))].sum() for _ in range(NB)])
        T, V, S = sum(Ds), float(boot.var()), len(ss)
        z = T / math.sqrt(ph * V) if V > 0 else 0.0
        pz, half = norm_p(z), 1.96 * math.sqrt(ph * V)
        m, sd = float(np.mean(Ds)), float(np.std(Ds, ddof=1))
        pt = t_p(m / (sd / math.sqrt(S)), S - 1) if sd > 0 else (0.0 if m != 0 else 1.0)
        npos, nneg = sum(1 for d in Ds if d > 0), sum(1 for d in Ds if d < 0)
        if claim:
            want = 1 if claim == "+" else -1
            ok = all((d > 0 if want > 0 else d < 0) for d in Ds) and pz < 0.05 and np.sign(T) == want
            verdict = "KEEPS p < 0.05 at the seed level" if ok else "does NOT keep p < 0.05 at the seed level"
        else:
            verdict = "report only"
        L.append(f"  [{ev}] {lab} [{claim or 'report'}]: " + "; ".join(per))
        L.append(f"     total {T:+d} of {ntot}; task-level sign p {sign_p(a_all, b_all):.3g} ({a_all}/{b_all}); seed signs +{npos} / -{nneg}; "
                 f"t across seeds p {pt:.3g}; design-effect test (phi {ph:.2f}) z {z:+.2f} p {pz:.3g}; "
                 f"cluster-robust 95 % [{T - half:+.1f}, {T + half:+.1f}] -> {verdict}")
        J["comparisons"].append({"row": ev, "label": lab, "claim": claim, "seeds": ss, "D": Ds, "total": T, "n": ntot,
                                 "task_p": sign_p(a_all, b_all), "t_p": pt, "phi": ph, "z": z, "p_design": pz,
                                 "ci": [T - half, T + half], "verdict": verdict})
        rows.append((ev, lab, claim, T, ntot, sign_p(a_all, b_all), pz, T - half, T + half, npos, nneg, verdict))
    L.append("\nVERDICT TABLE: row | comparison | claim | total | task-level p (v1.0.x) | design-effect p | cluster-robust 95 % | seed signs | verdict")
    for ev, lab, claim, T, n, p0, pz, lo, hi, npos, nneg, v in rows:
        L.append(f"  {ev} | {lab} | {claim or 'report'} | {T:+d} / {n} | {p0:.2g} | {pz:.2g} | [{lo:+.0f}, {hi:+.0f}] | +{npos}/-{nneg} | {v}")
    t = "\n".join(L)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "gg-seed-tn.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(t + "\n")
    with open(os.path.join(RESULTS, "gg-seed-tn.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(J, f, indent=1, default=float)
    print(t)


if __name__ == "__main__":
    main()
