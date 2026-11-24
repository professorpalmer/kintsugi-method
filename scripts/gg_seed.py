"""G-G (seed clustering); the lines are in evidence/LINES.md section 6 (written before any number). CPU only,
deterministic. The AppWorld client sends one seed on every LM call and llama-server re-seeds its sampler per request,
so the task-runs of one seed share one random stream (for every task and every arm). This script measures the
clustering and re-tests every headline AppWorld comparison with the seed as the unit where it matters.
    python -X utf8 scripts/gg_seed.py      -> results/gg-seed.txt, results/gg-seed.json
Public copy: outcomes from data/appworld/<arm>-s<seed>.csv; the first word and a hash of the first 8 words of each
first reply from data/appworld/openings/ (no reply text is published)."""
import collections, csv, json, math, os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "data", "appworld")
OPENINGS = os.path.join(LOGS, "openings")
RESULTS = os.path.join(ROOT, "results")
NB, RNG_SEED, MARGIN, N_REF = 10000, 20261122, 25, 834

# 27B arms (AppWorld test_challenge, 417 tasks): name -> {seed: experiment}
A27 = {
    "shipped": {106: "t3-pv3ac", 107: "t3-pv3ac", 108: "w2-pv3ac", 109: "w2-pv3ac"},
    "BF16": {s: "w2-pbf16" for s in (106, 107, 108, 109)},
    "W2-S": {106: "w2-pv3acB", 107: "w2-pv3acB"},
    "V5tc": {106: "w2-pv5tc", 107: "w2-pv5tc"},
    "V5bc": {106: "t3-pv5bc", 107: "t3-pv5bc"},
    "V5cc": {106: "w2-pv5cc", 107: "w2-pv5cc"},
    "U1": {106: "w2-puni", 107: "w2-puni"},
    "H1": {106: "w2-phawq", 107: "w2-phawq"},
    "g3": {106: "w2-pw2cg3", 107: "w2-pw2cg3"},
    "S-fd": {106: "w2-psfd", 107: "w2-psfd"},
    "noH2": {106: "w2-pv3ac0", 107: "w2-pv3ac0"},
    "W2-K": {106: "w2-pv3ack", 107: "w2-pv3ack"},
}
# Inkling families (seed sets differ by split): W2-D test_normal 168 tasks, W2-I test_challenge 417 tasks
AINK = {
    "D:M2": {s: "w2dm2" for s in (108, 109, 110, 111)}, "D:A": {s: "w2da" for s in (108, 109, 110, 111)},
    "D:U": {s: "w2du" for s in (108, 109, 110, 111)},
    "I:M2": {s: "w2im2" for s in (112, 113)}, "I:U": {s: "w2iu" for s in (112, 113)}, "I:FULL": {s: "w2-pfull" for s in (112, 113)},
}
# (label, X, Y, claim): claim "+" X > Y, "-" X < Y, "NI" X not worse than Y (margin -25 of 834), "" report only
COMP = [
    ("shipped vs BF16 (W2-A + W2-A2)", "shipped", "BF16", "+", None),
    ("shipped vs BF16, not worse (W2-A + W2-A2)", "shipped", "BF16", "NI", None),
    ("W2-S vs BF16", "W2-S", "BF16", "+", (106, 107)),
    ("W2-S vs shipped", "W2-S", "shipped", "", (106, 107)),
    ("V5tc vs V5bc", "V5tc", "V5bc", "+", None),
    ("V5tc vs shipped", "V5tc", "shipped", "", (106, 107)),
    ("V5bc vs shipped", "V5bc", "shipped", "-", (106, 107)),
    ("V5cc vs shipped", "V5cc", "shipped", "-", (106, 107)),
    ("U1 vs shipped", "U1", "shipped", "-", (106, 107)),
    ("H1 vs shipped", "H1", "shipped", "-", (106, 107)),
    ("g3 vs shipped", "g3", "shipped", "-", (106, 107)),
    ("g3 vs V5cc", "g3", "V5cc", "+", None),
    ("S-fd vs shipped", "S-fd", "shipped", "", (106, 107)),
    ("noH2 vs shipped", "noH2", "shipped", "", (106, 107)),
    ("noH2 vs BF16", "noH2", "BF16", "+", (106, 107)),
    ("Inkling W2-D M2 vs A (test_normal)", "D:M2", "D:A", "+", None),
    ("Inkling W2-D M2 vs U (test_normal)", "D:M2", "D:U", "", None),
    ("Inkling W2-I U vs M2", "I:U", "I:M2", "", None),
    ("Inkling W2-I M2 vs FULL, not worse", "I:M2", "I:FULL", "NI", None),
    # W2-K rows last (scored 21:5x, after the first G-G run): the rows above keep their bootstrap draws
    ("W2-K: noH2 (exact head) vs W2-K (3-bit head)", "noH2", "W2-K", "", (106, 107)),
    ("W2-K vs shipped", "W2-K", "shipped", "", (106, 107)),
]


# ---------- small exact / special functions (no scipy) ----------
def sign_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def norm_p(z):
    return math.erfc(abs(z) / math.sqrt(2))


def betacf(a, b, x):
    """continued fraction of the incomplete beta (Numerical Recipes betacf)"""
    tiny = 1e-300
    fix = lambda v: tiny if abs(v) < tiny else v  # noqa: E731
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 / fix(1 - qab * x / qap)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 / fix(1 + aa * d); c = fix(1 + aa / c); h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 / fix(1 + aa * d); c = fix(1 + aa / c); de = d * c; h *= de
        if abs(de - 1) < 1e-14:
            break
    return h


def betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    return bt * betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1 - bt * betacf(b, a, 1 - x) / b


def t_p(t, df):
    """two-sided p of Student t"""
    return betai(df / 2, 0.5, df / (df + t * t))


def chi2_sf(x, k):
    """upper tail of chi-square(k) via the regularized gamma Q(k/2, x/2)"""
    a, x = k / 2, x / 2
    if x < a + 1:
        s, term, n = 1 / a, 1 / a, a
        for _ in range(500):
            n += 1; term *= x / n; s += term
            if abs(term) < abs(s) * 1e-14:
                break
        return 1 - s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b, c, d = x + 1 - a, 1e300, 1 / (x + 1 - a); h = d
    for i in range(1, 500):
        an = -i * (i - a); b += 2
        d = an * d + b; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = b + an / c if abs(b + an / c) > 1e-300 else 1e-300
        de = d * c; h *= de
        if abs(de - 1) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


# ---------- data ----------
_cache = {}


def res(exp, seed):
    k = (exp, seed)
    if k not in _cache:
        f = os.path.join(LOGS, f"{exp}-s{seed}.csv")
        if not os.path.exists(f):
            _cache[k] = None
        else:
            with open(f, encoding="utf-8") as fh:
                _cache[k] = {t: int(s) for t, s in list(csv.reader(fh))[1:]}
    return _cache[k]


def arm(name):
    return A27.get(name) or AINK[name]


def pair(x, y, s):
    a, b = res(arm(x)[s], s), res(arm(y)[s], s)
    ids = sorted(set(a) & set(b))
    return np.array([a[t] for t in ids]), np.array([b[t] for t in ids]), ids


def first_words(exp, seed, n=8):
    """per task: (first word, hash of the first n words) of the first reply; equal tuples = the same first n words"""
    out = {}
    f = os.path.join(OPENINGS, f"{exp}-s{seed}.csv")
    if not os.path.exists(f):
        return out
    with open(f, encoding="utf-8") as fh:
        for tid, word, h in list(csv.reader(fh))[1:]:
            out[tid] = (word, h)
    return out


def main():
    L, J = [], {"opening": {}, "crn": {}, "phi1": {}, "phiD": {}, "comparisons": []}
    L.append("G-G (seed clustering): evidence/LINES.md section 6; CPU, deterministic")
    # ---- (1a) opening modes ----
    L.append("\n(1a) Opening modes: first word of the first reply (reasoning), share of the task-runs of each arm-seed")
    fw = {}
    for name, ss in A27.items():
        for s, exp in ss.items():
            w = first_words(exp, s)
            if w:
                fw[(name, s)] = w
    for name in A27:
        seeds = [s for s in (106, 107, 108, 109) if (name, s) in fw]
        if not seeds:
            continue
        rows, modes = [], collections.Counter()
        for s in seeds:
            c = collections.Counter((v[0] if v else "") for v in fw[(name, s)].values())
            modes.update(c)
            n = sum(c.values())
            rows.append((s, n, c))
        top = [m for m, _ in modes.most_common(3)]
        for s, n, c in rows:
            sh = ", ".join(f"'{m}' {100 * c[m] / n:.1f} %" for m in top)
            L.append(f"  {name:8s} s{s}: {n} task-runs; {sh}; other {100 * (n - sum(c[m] for m in top)) / n:.1f} %")
            J["opening"][f"{name}:{s}"] = {m: c[m] for m in top} | {"n": n}
        if 106 in seeds and 107 in seeds:   # chi-square mode x seed
            c6, c7 = rows[seeds.index(106)][2], rows[seeds.index(107)][2]
            cats = top + ["other"]
            def cnt(c, m):
                return c[m] if m != "other" else sum(c.values()) - sum(c[t] for t in top)
            tab = np.array([[cnt(c6, m) for m in cats], [cnt(c7, m) for m in cats]], float)
            tab = tab[:, tab.sum(0) > 0]
            e = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
            x2 = float(((tab - e) ** 2 / e).sum()); k = tab.shape[1] - 1
            L.append(f"  {name:8s} mode x seed (106 vs 107): chi2 {x2:.1f}, df {k}, p {chi2_sf(x2, k) if k else 1:.3g}")
    # ---- (1b) common random numbers: same first 8 words ----
    L.append("\n(1b) Same first 8 words of the first reply, share of tasks (arm pairs on the same seed vs across seeds)")
    names = [n for n in A27 if (n, 106) in fw and (n, 107) in fw]   # 21:5x: W2-K complete (the first run had its partial 105 + 70 tasks)
    same, cross, selfx = [], [], []
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            for s in (106, 107):
                ids = set(fw[(x, s)]) & set(fw[(y, s)])
                same.append(np.mean([fw[(x, s)][t] == fw[(y, s)][t] for t in ids]))
            ids = set(fw[(x, 106)]) & set(fw[(y, 107)])
            cross.append(np.mean([fw[(x, 106)][t] == fw[(y, 107)][t] for t in ids]))
        ids = set(fw[(x, 106)]) & set(fw[(x, 107)])
        selfx.append(np.mean([fw[(x, 106)][t] == fw[(x, 107)][t] for t in ids]))
    L.append(f"  different arms, SAME seed: mean {100 * np.mean(same):.1f} % (range {100 * min(same):.1f}-{100 * max(same):.1f}, {len(same)} arm-pair-seeds)")
    L.append(f"  different arms, seed 106 vs 107: mean {100 * np.mean(cross):.1f} % (range {100 * min(cross):.1f}-{100 * max(cross):.1f})")
    L.append(f"  same arm, seed 106 vs 107: mean {100 * np.mean(selfx):.1f} % (range {100 * min(selfx):.1f}-{100 * max(selfx):.1f})")
    J["crn"] = {"same_seed": float(np.mean(same)), "cross_seed": float(np.mean(cross)), "self_cross_seed": float(np.mean(selfx))}
    # ---- (1c) phi1 ----
    L.append("\n(1c) Design effect of one arm's seed total: z = (a - b) / sqrt(a + b) over the tasks of two seeds")
    def phi1(fam, spairs):
        zs = []
        for name, ss in fam.items():
            for s1, s2 in spairs:
                if s1 in ss and s2 in ss and res(ss[s1], s1) and res(ss[s2], s2):
                    r1, r2 = res(ss[s1], s1), res(ss[s2], s2)
                    ids = set(r1) & set(r2)
                    if len(ids) < 0.95 * max(len(r1), len(r2)):
                        continue
                    a = sum(1 for t in ids if r1[t] and not r2[t]); b = sum(1 for t in ids if r2[t] and not r1[t])
                    z = (a - b) / math.sqrt(a + b) if a + b else 0.0
                    zs.append((f"{name} s{s1}/s{s2}", a, b, z))
        return zs
    z27 = phi1({k: v for k, v in A27.items() if k != "W2-K"}, [(106, 107), (108, 109)])   # W2-K out: phi1 / phiD as first reported
    zin = phi1(AINK, [(108, 109), (110, 111), (112, 113)])
    for lab, zs in (("27B", z27), ("Inkling", zin)):
        for n_, a, b, z in zs:
            L.append(f"  {lab:7s} {n_:22s} only-first {a:3d} / only-second {b:3d}: z {z:+.2f}")
        ssum = sum(z * z for *_, z in zs); k = len(zs); ph = ssum / k
        p = chi2_sf(ssum, k)
        L.append(f"  {lab}: phi1 = mean z^2 = {ph:.2f} over {k}; sum z^2 {ssum:.1f}, chi2({k}) p {p:.3g}; rho = {(ph - 1) / 416:+.4f}")
        J["phi1"][lab] = {"phi1": ph, "k": k, "sum_z2": ssum, "p": p}
    ph1 = J["phi1"]["27B"]
    L.append("  LINE (1): the task-runs of a seed are " + ("CLUSTERED" if ph1["phi1"] > 1 and ph1["p"] < 0.05 else "NOT SHOWN to be clustered")
             + f" (27B phi1 {ph1['phi1']:.2f}, chi2 p {ph1['p']:.3g}; approximate: the arms share each seed's random stream)")
    # ---- (1d) phiD ----
    L.append("\n(1d) Design effect of a paired difference: h = sample var of D_s over seeds / mean discordant n_s")
    def phid(fam, seeds):
        hs = []
        nm = [n for n in fam if all(s in fam[n] and res(fam[n][s], s) for s in seeds)]
        for i, x in enumerate(nm):
            for y in nm[i + 1:]:
                D, N = [], []
                for s in seeds:
                    a, b = res(fam[x][s], s), res(fam[y][s], s)
                    ids = set(a) & set(b)
                    D.append(sum(a[t] - b[t] for t in ids)); N.append(sum(1 for t in ids if a[t] != b[t]))
                hs.append((f"{x}-{y}", float(np.var(D, ddof=1) / np.mean(N))))
        return hs
    h27 = phid({k: v for k, v in A27.items() if k != "W2-K"}, (106, 107))
    hD = phid({k: v for k, v in AINK.items() if k.startswith("D:")}, (108, 109, 110, 111))
    hI = phid({k: v for k, v in AINK.items() if k.startswith("I:")}, (112, 113))
    hsb = [("shipped-BF16 (4 seeds)", None)]
    D4 = []; N4 = []
    for s in (106, 107, 108, 109):
        x, y, _ = pair("shipped", "BF16", s); D4.append(int((x - y).sum())); N4.append(int((x != y).sum()))
    hsb = float(np.var(D4, ddof=1) / np.mean(N4))
    for lab, hs in (("27B (55 pairs, 106/107)", h27), ("Inkling W2-D (4 seeds)", hD), ("Inkling W2-I (2 seeds)", hI)):
        v = [h for _, h in hs]
        L.append(f"  {lab}: mean h {np.mean(v):.2f}, median {np.median(v):.2f}, range {min(v):.2f}-{max(v):.2f}, share h > 1: {np.mean(np.array(v) > 1):.2f}")
        J["phiD"][lab] = {"mean": float(np.mean(v)), "median": float(np.median(v)), "n": len(v)}
    L.append(f"  shipped vs BF16 over 4 seeds: D_s {D4}, n_s {N4}: h {hsb:.2f}")
    phiD27 = max(1.0, float(np.mean([h for _, h in h27])))
    phiDink = max(1.0, float(np.mean([h for _, h in hD + hI])))
    L.append(f"  phiD used: 27B {phiD27:.2f}; Inkling {phiDink:.2f} (W2-D + W2-I pairs pooled); floored at 1")
    J["phiD"]["used"] = {"27B": phiD27, "Inkling": phiDink}
    # ---- (2) comparisons ----
    L.append("\n(2) Seed-level re-analysis (D = X - Y task passes; V = var of sum D from a within-seed task bootstrap)")
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for lab, x, y, claim, seeds in COMP:
        ax, ay = arm(x), arm(y)
        ss = seeds or tuple(s for s in sorted(set(ax) & set(ay)))
        ss = tuple(s for s in ss if res(ax[s], s) and res(ay[s], s))
        if x == "W2-K" and (len(ss) < 2 or any(len(res(ax[s], s)) < 400 for s in ss)):
            L.append(f"  {lab}: not scored yet -> left out"); continue
        if len(ss) < 2:
            L.append(f"  {lab}: fewer than 2 seeds -> left out"); continue
        Ds, Ns, Vs, per, ntot, a_all, b_all = [], [], [], [], 0, 0, 0
        boot = np.zeros(NB)
        for s in ss:
            xv, yv, ids = pair(x, y, s)
            d = xv - yv
            Ds.append(int(d.sum())); Ns.append(int((d != 0).sum())); ntot += len(ids)
            a, b = int((d > 0).sum()), int((d < 0).sum()); a_all += a; b_all += b
            per.append(f"s{s} {int(xv.sum())}-{int(yv.sum())}={int(d.sum()):+d} ({a}/{b}, p {sign_p(a, b):.3g})")
            bs = np.array([d[rng.integers(0, len(d), len(d))].sum() for _ in range(NB)])
            Vs.append(float(bs.var())); boot += bs
        T = sum(Ds); V = float(boot.var()); S = len(ss)
        phi = phiDink if x.startswith(("D:", "I:")) else phiD27
        z = T / math.sqrt(phi * V) if V > 0 else 0.0
        pz = norm_p(z)
        half = 1.96 * math.sqrt(phi * V)
        lo, hi = T - half, T + half
        m, sd = float(np.mean(Ds)), float(np.std(Ds, ddof=1))
        tt = m / (sd / math.sqrt(S)) if sd > 0 else float("inf") * (1 if m > 0 else -1 if m < 0 else 0)
        pt = t_p(tt, S - 1) if sd > 0 else (0.0 if m != 0 else 1.0)
        npos, nneg = sum(1 for d in Ds if d > 0), sum(1 for d in Ds if d < 0)
        p_seed_sign = sign_p(npos, nneg)
        p_task = sign_p(a_all, b_all)
        betw, within = float(np.var(Ds, ddof=1)), float(np.mean(Vs))
        if claim in ("+", "-"):
            want = 1 if claim == "+" else -1
            consistent = all((d > 0 if want > 0 else d < 0) for d in Ds)
            verdict = "KEEPS p < 0.05 at the seed level" if consistent and pz < 0.05 and np.sign(T) == want else "does NOT keep p < 0.05 at the seed level"
        elif claim == "NI":
            marg = -MARGIN * ntot / N_REF
            verdict = (f"KEEPS 'not worse' (lower end {lo:+.1f} > {marg:+.1f})" if lo > marg
                       else f"does NOT keep 'not worse' (lower end {lo:+.1f} <= {marg:+.1f})")
        else:
            verdict = "report only"
        L.append(f"  {lab} [{claim or 'report'}]: " + "; ".join(per))
        L.append(f"     total {T:+d} of {ntot}; task-level sign p {p_task:.3g} (as before); seed signs +{npos} / -{nneg}, "
                 f"sign test across seeds p {p_seed_sign:.3g}; t across seeds t {tt:+.2f} df {S - 1} p {pt:.3g}")
        L.append(f"     between-seed var of D_s {betw:.1f} vs within-seed var {within:.1f} (ratio {betw / within if within else float('nan'):.2f}); "
                 f"design-effect test (phiD {phi:.2f}) z {z:+.2f} p {pz:.3g}; cluster-robust 95 % [{lo:+.1f}, {hi:+.1f}]")
        L.append(f"     -> {verdict}")
        J["comparisons"].append({"label": lab, "claim": claim, "seeds": list(ss), "D": Ds, "n": Ns, "total": T, "task_p": p_task,
                                 "seed_sign_p": p_seed_sign, "t": tt, "t_p": pt, "phiD": phi, "z": z, "p_design": pz,
                                 "ci": [lo, hi], "between_var": betw, "within_var": within, "verdict": verdict})
        rows.append((lab, claim, T, ntot, p_task, pz, lo, hi, npos, nneg, pt, verdict))
    L.append("\nVERDICT TABLE: comparison | claim | total | task-level p (as before) | design-effect p | cluster-robust 95 % | seed signs | t across seeds p | verdict")
    for lab, claim, T, n, pt_, pz, lo, hi, npos, nneg, ptt, v in rows:
        L.append(f"  {lab} | {claim or 'report'} | {T:+d} / {n} | {pt_:.2g} | {pz:.2g} | [{lo:+.0f}, {hi:+.0f}] | +{npos}/-{nneg} | {ptt:.2g} | {v}")
    t = "\n".join(L)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "gg-seed.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(t + "\n")
    with open(os.path.join(RESULTS, "gg-seed.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(J, f, indent=1, default=float)
    print(t)


if __name__ == "__main__":
    main()
