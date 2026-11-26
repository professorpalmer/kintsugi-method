# Public copy for the Kintsugi-method paper repository (paths point into this repository).
"""Paper v1.2, items 1 and 7 of docs/paper/V1.2-CHANGES.md (CPU only, deterministic; per-task files on disk).

Item 1: is the T 1.0 lead of shipped over BF16 larger than the T 0 lead? Difference of leads
  Delta = mean over the four T 1.0 seeds of (shipped - BF16) - (shipped T0 - BF16 T0), in task-runs of 417.
  (a) scenario bootstrap (resample AppWorld scenarios = task-id prefix, all their tasks, 10,000 draws);
  (b) the same draws with the G-G design effect phiD = 2.75 on the T 1.0 part only (T 0 is greedy: no shared
      random stream), V = phiD Var(L1) + Var(L0) - 2 Cov(L1, L0);
  (c) the same against each single T 1.0 seed.
Item 7: split each arm's T 1.0 runs (seeds 106-109) by the arm's own greedy (T 0) result: share of T 1.0 runs that
  FAIL among the tasks greedy passes; share that PASS among the tasks greedy fails. Description, no line.
    python -X utf8 scripts/v12_t0.py   -> results/v12-t0.txt, results/v12-t0.json
"""
import csv, json, math, os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, "data", "appworld")
OUT = os.path.join(ROOT, "results")
NB, RNG_SEED, PHI = 10000, 20261122, 2.75
T1 = {"shipped": {106: "t3-pv3ac", 107: "t3-pv3ac", 108: "w2-pv3ac", 109: "w2-pv3ac"},
      "BF16": {s: "w2-pbf16" for s in (106, 107, 108, 109)}}
T0 = {"shipped": "w2-pv3acT", "BF16": "w2-pbf16T"}
SEEDS = (106, 107, 108, 109)


def res(exp, seed):
    with open(os.path.join(LOGS, f"{exp}-s{seed}.csv"), encoding="utf-8") as f:
        return {t: int(v) for t, v in list(csv.reader(f))[1:]}


def norm_p(z):
    return math.erfc(abs(z) / math.sqrt(2))


def main():
    L, J = [], {}
    r1 = {a: {s: res(T1[a][s], s) for s in SEEDS} for a in T1}
    r0 = {a: res(T0[a], 106) for a in T0}
    ids = sorted(set.intersection(*[set(r1[a][s]) for a in r1 for s in SEEDS], set(r0["shipped"]), set(r0["BF16"])))
    n = len(ids)
    # per-task leads
    d1 = {s: np.array([r1["shipped"][s][t] - r1["BF16"][s][t] for t in ids], float) for s in SEEDS}
    d1m = np.mean([d1[s] for s in SEEDS], axis=0)          # mean T 1.0 lead per task (per seed)
    d0 = np.array([r0["shipped"][t] - r0["BF16"][t] for t in ids], float)
    scen = sorted({t.split("_")[0] for t in ids})
    sidx = {sc: [i for i, t in enumerate(ids) if t.split("_")[0] == sc] for sc in scen}
    L.append(f"Paper v1.2 items 1 and 7; {n} tasks in all six runs, {len(scen)} scenarios; NB {NB}, rng {RNG_SEED}")
    L.append("Totals: " + ", ".join(f"{a} T1 s{s} {sum(r1[a][s][t] for t in ids)}" for a in r1 for s in SEEDS)
             + f"; shipped T0 {sum(r0['shipped'][t] for t in ids)}, BF16 T0 {sum(r0['BF16'][t] for t in ids)}")
    rng = np.random.default_rng(RNG_SEED)
    draws = [np.concatenate([sidx[scen[j]] for j in rng.integers(0, len(scen), len(scen))]) for _ in range(NB)]

    def block(lead1, label):
        L1 = lead1.sum(); L0 = d0.sum(); D = L1 - L0
        b1 = np.array([lead1[ix].sum() for ix in draws]) * n / np.array([len(ix) for ix in draws])
        b0 = np.array([d0[ix].sum() for ix in draws]) * n / np.array([len(ix) for ix in draws])
        bd = b1 - b0
        lo, hi = np.percentile(bd, [2.5, 97.5])
        v1, v0, c = b1.var(ddof=1), b0.var(ddof=1), np.cov(b1, b0)[0, 1]
        se_plain = math.sqrt(v1 + v0 - 2 * c)
        se_phi = math.sqrt(PHI * v1 + v0 - 2 * c)
        p_plain, p_phi = norm_p(D / se_plain), norm_p(D / se_phi)
        L.append(f"  {label}: T1 lead {L1:+.2f}, T0 lead {L0:+.0f}, Delta {D:+.2f}; scenario bootstrap 95 % "
                 f"[{lo:+.1f}, {hi:+.1f}], normal p {p_plain:.3f}; with phiD {PHI} on the T1 part: "
                 f"[{D - 1.96 * se_phi:+.1f}, {D + 1.96 * se_phi:+.1f}], p {p_phi:.3f}")
        return {"T1_lead": L1, "T0_lead": L0, "Delta": D, "boot95": [lo, hi], "p_plain": p_plain,
                "ci_phi": [D - 1.96 * se_phi, D + 1.96 * se_phi], "p_phi": p_phi}

    L.append("\nItem 1: lead at T 1.0 minus lead at T 0 (task-runs of 417; T0 = one greedy run, client seed 106)")
    J["item1"] = {"mean4": block(d1m, "mean of the 4 T1 seeds")}
    for s in SEEDS:
        J["item1"][str(s)] = block(d1[s], f"T1 seed {s} alone")
    # per-seed leads and a t view
    leads = [int(d1[s].sum()) for s in SEEDS]
    L.append(f"  per-seed T1 leads {leads}; T0 lead {d0.sum():+.0f}")

    L.append("\nItem 7: each arm's T 1.0 runs (4 seeds) split by its own greedy result (description, no line)")
    J["item7"] = {}
    for a in ("BF16", "shipped"):
        gp = [t for t in ids if r0[a][t] == 1]; gf = [t for t in ids if r0[a][t] == 0]
        fail_gp = sum(1 - r1[a][s][t] for t in gp for s in SEEDS); pass_gf = sum(r1[a][s][t] for t in gf for s in SEEDS)
        ngp, ngf = 4 * len(gp), 4 * len(gf)
        L.append(f"  {a}: greedy passes {len(gp)}, fails {len(gf)}; T1 runs that FAIL among greedy passes "
                 f"{fail_gp} / {ngp} = {100 * fail_gp / ngp:.1f} %; T1 runs that PASS among greedy fails "
                 f"{pass_gf} / {ngf} = {100 * pass_gf / ngf:.1f} %")
        J["item7"][a] = {"greedy_pass": len(gp), "greedy_fail": len(gf), "fail_of_gp": [fail_gp, ngp],
                         "pass_of_gf": [pass_gf, ngf]}
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "v12-t0.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump(J, open(os.path.join(OUT, "v12-t0.json"), "w", encoding="utf-8"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
