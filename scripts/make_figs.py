"""Figures of the Kintsugi-method paper, from data already in the record only (no new runs). Deterministic.
    python -X utf8 scripts/make_figs.py      -> paper/figs/fig{1..7}-<name>.{png,pdf} (numbered in order of appearance)

Sources by output file (the paper numbers figures by the file prefix; every number is also in evidence/EVIDENCE.md):
- fig1-workflow: the workflow (no data).
- fig2-bf16-per-seed: AppWorld test_challenge per seed for the shipped file, BF16 and the shipped file in BF16's setup
  (per-task score files; E5.2, E5.4, E5.5) and the seed-level differences with cluster-robust intervals (the JSON output of
  the seed-clustering script; E5.1, E5.4-E5.6).
- fig3-kl-vs-task: KL change and recall-gate results per map (E1.3, E1.6-E1.9, E1.28, E1.29); test_challenge difference to the
  shipped file with cluster-robust intervals (E5.2, E5.12, E5.13).
- fig4-b1-recall-population: data/ (scripts/b1/b1_score.py) and data/ (KL-table B = V5c by depth; E1.34).
- fig5-item-margins: per-item capped margins, data/ (G-B2, E1.33) and data/ (G-C2,
  E2.23), paired by request order as the two scorers do.
- fig6-seed-pairs: AppWorld test_normal counts per seed pair (E3.1-E3.8, E3.15, E3.16) and the shipped - BF16 gap with
  cluster-robust intervals (E5.28).
- fig7-inkling-appworld: Inkling AppWorld test_normal per seed: seeds 102-105 (E4.8, E4.12, E3.12, E3.13) and seeds 108-111 with
  the setup equal (E5.22).
Style: light theme, Okabe-Ito colours (colour-blind safe) plus distinct markers / hatches (readable in grayscale). Legends
are outside the plot area.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "paper", "figs")
OK = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73", "yellow": "#F0E442",
      "blue": "#0072B2", "verm": "#D55E00", "purple": "#CC79A7", "grey": "#777777"}
CAP = 5.0
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
                     "figure.dpi": 100, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.hashsalt": "kintsugi"})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight", metadata={"Software": None})
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight", metadata={"CreationDate": None, "Creator": None,
                                                                              "Producer": None})
    plt.close(fig)


def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


AW = os.path.join(ROOT, "data", "appworld")
GGSEED = os.path.join(ROOT, "results", "gg-seed.json")


def aw_score(stem):
    """(passed, tasks) of one arm-seed"""
    with open(os.path.join(AW, f"{stem}.csv"), encoding="utf-8") as f:
        rows = [x.strip().split(",") for x in f.readlines()[1:] if x.strip()]
    return sum(int(r[1]) for r in rows), len(rows)


def seed_level(label):
    """one comparison of the seed-clustering script: total, cluster-robust interval, design-effect p, per-seed D"""
    with open(GGSEED, encoding="utf-8") as f:
        for c in json.load(f)["comparisons"]:
            if c["label"] == label:
                return c
    raise KeyError(label)


def pfmt(p):
    return f"{p:.2g}" if p >= 0.001 else f"{p:.1e}"


# ------------------------------------------------------------------ Fig 1
def fig1():
    # (name, KL change % vs shipped/start, recall items lost at 117k, kind, note)
    maps = [("shipped", 0.0, 0, "ref"), ("KL-group", -9.3, 2, "real"), ("KL-table A*", -8.5, 1, "fragile"),
            ("KL-table B*", -10.6, 1, "margin"), ("KL-only winner", -7.0, 0, "pass"), ("KL-tools", -7.9, 0, "pass"),
            ("Uniform", -3.8, 1, "fragile"), ("HAWQ-style", 12.7, 1, "fragile")]
    sty = {"ref": dict(marker="*", s=140, c=OK["black"], ec=OK["black"]),
           "real": dict(marker="X", s=80, c=OK["verm"], ec=OK["black"]),
           "margin": dict(marker="D", s=55, c=OK["verm"], ec=OK["black"]),
           "fragile": dict(marker="o", s=60, c="white", ec=OK["orange"], linewidths=1.8),
           "pass": dict(marker="s", s=55, c=OK["blue"], ec=OK["black"])}
    lab = {"ref": "shipped (reference)", "real": "lost a needle and a copy (gate)", "margin": "needle below 0 (margin probe)",
           "fragile": "lost only the fragile copy item", "pass": "full recall (24/24 + 20/20)"}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
    seen = set()
    off = {"KL-only winner": 0.1, "KL-tools": -0.1}
    pos = {"shipped": (8, -12, "left"), "KL-group": (8, 0, "left"), "KL-table A*": (0, -14, "center"),
           "KL-table B*": (0, 9, "center"), "KL-only winner": (0, 11, "center"), "KL-tools": (-8, -3, "right"),
           "Uniform": (0, 9, "center"), "HAWQ-style": (0, 9, "center")}
    for n, x, y, k in maps:
        ax[0].scatter(x, y + off.get(n, 0), label=None if k in seen else lab[k], zorder=3, **sty[k]); seen.add(k)
        dx, dy, ha = pos[n]
        ax[0].annotate(n, (x, y + off.get(n, 0)), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                       fontsize=7.5)
    ax[0].axvline(0, color=OK["grey"], lw=0.8, ls=":")
    ax[0].set_xlabel("KL change vs the shipped allocation (%)\n(* = on the build subset T only)")
    ax[0].set_ylabel("recall items lost at 117k (of 44)")
    ax[0].set_yticks([0, 1, 2]); ax[0].set_ylim(-0.4, 2.6); ax[0].set_xlim(-13, 15)
    ax[0].set_title("(a) Recall gate at 117k")
    ax[0].legend(loc="upper center", frameon=False, fontsize=7, ncol=2, bbox_to_anchor=(0.5, -0.3),
                 columnspacing=0.8, handletextpad=0.3)
    # (b) AppWorld test_challenge, seeds 106 + 107: difference to the shipped file against the KL change (p1)
    rows = [("KL-tools", -7.9, "V5tc vs shipped", "top"), ("band only", 4.1, "S-fd vs shipped", "top"),
            ("HAWQ-style", 12.7, "H1 vs shipped", "top"), ("uniform", -3.8, "U1 vs shipped", "bottom"),
            ("KL-only winner", -7.0, "V5bc vs shipped", "bottom-right"), ("KL-table B", -8.8, "V5cc vs shipped", "left")]
    seen = set()
    for n, x, key, side in rows:
        c = seed_level(key)
        tot, (lo, hi) = c["total"], c["ci"]
        keeps = c["p_design"] < 0.05 and (all(d > 0 for d in c["D"]) or all(d < 0 for d in c["D"]))
        lab = "seed-level p < 0.05, same sign on both seeds" if keeps else "not significant at the seed level"
        ax[1].errorbar(x, tot, yerr=[[tot - lo], [hi - tot]], fmt="o" if keeps else "s", ms=7,
                       color=OK["verm"] if keeps else OK["blue"], mfc=OK["verm"] if keeps else "white", mew=1.6, capsize=3,
                       label=None if lab in seen else lab)
        seen.add(lab)
        if side == "bottom-right":
            ax[1].annotate(n, (x, lo), xytext=(-3, -5), textcoords="offset points", ha="left", va="top", fontsize=7)
        elif side == "left":
            ax[1].annotate(n, (x, tot), xytext=(-8, 0), textcoords="offset points", ha="right", va="center", fontsize=7)
        else:
            ax[1].annotate(n, (x, hi if side == "top" else lo), xytext=(0, 4 if side == "top" else -4),
                           textcoords="offset points", ha="center", va="bottom" if side == "top" else "top", fontsize=7)
    ax[1].scatter([0], [0], marker="*", s=140, c=OK["black"], zorder=3)
    ax[1].annotate("shipped", (0, 0), xytext=(0, 8), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    ax[1].axhline(0, color=OK["grey"], lw=0.8, ls=":"); ax[1].axvline(0, color=OK["grey"], lw=0.8, ls=":")
    ax[1].set_xlim(-15, 17); ax[1].set_ylim(-160, 80)
    ax[1].set_xlabel("KL change vs the shipped allocation (%)")
    ax[1].set_ylabel("task-runs vs shipped (of 834)\ncluster-robust 95 %")
    ax[1].set_title("(b) Harder agent split")
    ax[1].legend(loc="upper center", frameon=False, ncol=1, fontsize=7, bbox_to_anchor=(0.5, -0.3))
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig3-kl-vs-task")


# ------------------------------------------------------------------ Fig 2
def fig2():
    pairs = ["100 + 101\n(selection)", "102 + 103\n(fresh)", "104 + 105\n(fresh)"]
    arms = [("shipped (selected)", [302, 294, 292], OK["black"], "*", "-"),
            ("BF16 (never selected)", [281, 293, 294], OK["blue"], "s", "-"),
            ("KL-tools", [301, np.nan, 299], OK["orange"], "^", "--"),
            ("author's fits", [288, np.nan, 289], OK["green"], "v", "--"),
            ("KL-only winner", [285, np.nan, 296], OK["verm"], "D", "--")]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.5))
    x = np.arange(3)
    for n, v, c, m, ls in arms:
        v = np.array(v, dtype=float)
        ok = ~np.isnan(v)
        ax[0].plot(x[ok], v[ok], ls=ls, color=c, marker=m, ms=7 if m != "*" else 11, mfc=c if m in "*s" else "white",
                   mew=1.5, label=n)
    ax[0].set_xticks(x); ax[0].set_xticklabels(pairs)
    ax[0].set_ylabel("AppWorld test_normal task-runs (of 336)")
    ax[0].set_title("(a) Score per seed pair")
    ax[0].legend(frameon=False, fontsize=7, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.24), columnspacing=0.8, handletextpad=0.4)
    g = [(21, -1, 43, 0.057), (1, -22, 24, 0.93), (-2, -25, 21, 0.87)]   # cluster-robust 95 %, seed-level p (E5.28)
    for i, (p, lo, hi, pv) in enumerate(g):
        ax[1].errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="o", ms=8, color=OK["black"] if i == 0 else OK["blue"],
                       mfc=OK["black"] if i == 0 else "white", mew=1.6, capsize=4)
        ax[1].annotate(f"{p:+d}\np = {pv:g}", (i, hi), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5)
    ax[1].axhline(0, color=OK["grey"], lw=0.8, ls=":")
    ax[1].set_xticks(x); ax[1].set_xticklabels(pairs); ax[1].set_xlim(-0.5, 2.5); ax[1].set_ylim(-32, 58)
    ax[1].set_ylabel("shipped - BF16 (task-runs of 336)\ncluster-robust 95 %")
    ax[1].set_title("(b) The +21 lead on the selection seeds")
    fig.tight_layout()
    save(fig, "fig6-seed-pairs")


# ------------------------------------------------------------------ Fig 3
def fig3():
    S = [r for r in jl(os.path.join(ROOT, "data/margins/gb2/shipped.jsonl")) if "item" in r]
    gb = {}
    for t in ("U1", "H1"):
        X = [r for r in jl(os.path.join(ROOT, f"data/margins/gb2/{t}.jsonl")) if "item" in r]
        assert [r["item"] for r in X] == [r["item"] for r in S]
        gb[t] = np.array([min(a["min"], CAP) - min(b["min"], CAP) for a, b in zip(X, S)])
    box = os.path.join(ROOT, "data/margins/gc2")
    gc = {}
    for arm in "bcd":
        d = []
        for D in (117000, 245000):
            for fam in ("needle", "copy"):
                A = jl(os.path.join(box, f"a-{D}-{fam}.jsonl")); X = jl(os.path.join(box, f"{arm}-{D}-{fam}.jsonl"))
                assert [r["item"] for r in A] == [r["item"] for r in X]
                d += [min(x["min"], CAP) - min(a["min"], CAP) for a, x in zip(A, X)]
        gc[arm] = np.array(d)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for t, c, m, n in (("U1", OK["orange"], "o", "uniform baseline"), ("H1", OK["blue"], "s", "HAWQ-style baseline")):
        v = np.sort(gb[t])
        ax[0].plot(np.arange(len(v)), v, ls="-", lw=1, color=c, marker=m, ms=3.5, mfc="white" if m == "o" else c,
                   label=f"{n}: {int(np.sum(v != 0))} of 44 items differ")
    ax[0].axhline(0, color=OK["grey"], lw=0.8, ls=":")
    ax[0].set_xlabel("items, sorted by difference (Qwen, 117k)")
    ax[0].set_ylabel("capped item margin vs shipped (nats)")
    ax[0].set_title("(a) Qwen baselines: the change sits in few items")
    ax[0].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    for arm, c, m, n in (("b", OK["verm"], "^", "blanket q4_0"), ("c", OK["green"], "v", "blanket q8_0"),
                         ("d", OK["purple"], "D", "q4_0 global + f16 SWA")):
        v = np.sort(gc[arm])
        ax[1].plot(np.arange(len(v)), v, ls="-", lw=1, color=c, marker=m, ms=3, mfc="white", mew=1,
                   label=f"{n}: mean {v.mean():+.3f}")
    ax[1].axhline(0, color=OK["grey"], lw=0.8, ls=":")
    ax[1].set_xlabel("item-depth units, sorted (Inkling, 117k + 245k)")
    ax[1].set_ylabel("capped margin vs release K/V (nats)")
    ax[1].set_title("(b) Inkling K/V settings: spread, no shift")
    ax[1].legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig5-item-margins")


# ------------------------------------------------------------------ Fig 4
def fig4():
    panels = [("(a) Seeds 102-105 (none used for a choice)", [102, 103, 104, 105],
               [("M2, release K/V (ours)", "inkm", OK["blue"], ""), ("M2, 4-bit global cache", "q65", OK["sky"], "//"),
                ("official weights, same engine", "raw", OK["black"], ".."),
                ("Unsloth's setup (setup vs setup)", "ink", OK["orange"], "xx")]),
              ("(b) Seeds 108-111, one engine and launcher setting", [108, 109, 110, 111],
               [("M2 (ours)", "w2dm2", OK["blue"], ""), ("uniform map, same bytes", "w2du", OK["green"], "//"),
                ("Unsloth UD-Q2_K_XL, same engine", "w2da", OK["orange"], "xx")])]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [4, 3]})
    for ax, (title, seeds, arms) in zip(axs, panels):
        w = 0.8 / len(arms)
        for i, (n, key, c, h) in enumerate(arms):
            sc = [aw_score(f"{key}-s{s}") for s in seeds]
            vals = [v for v, _ in sc]
            ax.bar(np.arange(len(seeds)) + (i - (len(arms) - 1) / 2) * w, vals, w, color=c if not h else "white", edgecolor=c,
                   hatch=h, lw=1.2, label=f"{n}: {sum(vals)} / {sum(k for _, k in sc)}")
        ax.set_xticks(np.arange(len(seeds))); ax.set_xticklabels([str(s) for s in seeds])
        ax.set_ylim(125, 158); ax.set_xlabel("seed"); ax.set_title(title, fontsize=8.5)
        ax.legend(frameon=False, ncol=1, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    axs[0].set_ylabel("AppWorld test_normal\ntask-runs (of 168)")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig7-inkling-appworld")


# ------------------------------------------------------------------ Fig 5
def fig5():
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.2); ax.axis("off")
    steps = [("1. Diagnose", "KL and task probes\n(recall at depth, long\nchat, agent runs)"),
             ("2. Locate", "flip one group;\nlist the items\nthat go below 0"),
             ("3. Repair locally", "move bits / pick a\ncache type at equal\nbytes; measure it"),
             ("4. Gate", "pass lines written\nbefore the run;\na miss stays a miss"),
             ("5. Confirm", "fresh seeds and\nheld-out items;\nclaims follow these")]
    xs = np.linspace(1.0, 9.0, len(steps))
    for i, (t, d) in enumerate(steps):
        ax.add_patch(plt.Rectangle((xs[i] - 0.88, 0.25), 1.76, 1.7, fc="white", ec=OK["blue"] if i < 4 else OK["verm"],
                                   lw=1.4))
        ax.text(xs[i], 1.68, t, ha="center", va="center", fontsize=7.5, weight="bold")
        ax.text(xs[i], 0.95, d, ha="center", va="center", fontsize=6.3)
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.9, 1.1), xytext=(xs[i] + 0.9, 1.1),
                        arrowprops=dict(arrowstyle="->", color=OK["black"], lw=1.2))
    ax.annotate("", xy=(xs[0], 0.22), xytext=(xs[3], 0.22),
                arrowprops=dict(arrowstyle="->", color=OK["grey"], lw=1, ls="--", connectionstyle="arc3,rad=-0.25"))
    ax.text((xs[0] + xs[3]) / 2, 0.0, "a failed gate sends the candidate back", ha="center", fontsize=7, color=OK["grey"])
    save(fig, "fig1-workflow")


# ------------------------------------------------------------------ Fig 6 (B1)
def fig6():
    import re
    res = open(os.path.join(ROOT, "results/b1-result.txt"), encoding="utf-8").read()
    num = r"([+-]\d+\.\d+)"
    pat = re.compile(r"^  (\w+): L " + num + r" \[" + num + r", " + num + r"\].*?; S " + num + r" \[" + num + r", " + num +
                     r"\].*?; C " + num + r" \[" + num + r", " + num + r"\]", re.M)
    rows = {m.group(1): [float(x) for x in m.groups()[1:]] for m in pat.finditer(res)}
    fl = re.search(r"F_L = max\(run [0-9.]+, \|L\(V2body\)\| [0-9.]+\) = ([0-9.]+); F_C = .*? = ([0-9.]+)", res)
    FL, FC = float(fl.group(1)), float(fl.group(2))
    order = [("V3s2", "KL-\ngroup"), ("V5a", "KL-\ntable A"), ("V5c", "KL-\ntable B"), ("V5bc", "KL-only\nwinner"),
             ("V5tc", "KL-\ntools"), ("U1", "uni-\nform"), ("H1", "HAWQ-\nstyle*"), ("V2body", "author's\nfits*")]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1.9, 1]})
    for j, (key, lab, c, m, dx) in enumerate((("L", "L: long items (117k + 245k)", OK["verm"], "o", -0.22),
                                             ("S", "S: shallow twins (~3.9k)", OK["blue"], "s", 0.0),
                                             ("C", "C = L - S (depth contrast)", OK["black"], "D", 0.22))):
        for i, (k, _) in enumerate(order):
            v = rows[k][3 * j:3 * j + 3]
            ax[0].errorbar(i + dx, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt=m, ms=4.5, color=c,
                           mfc=c if key != "S" else "white", mew=1.2, capsize=2, lw=1,
                           label=lab if i == 0 else None)
    ax[0].axhline(-FL, color=OK["verm"], lw=0.9, ls="--")
    ax[0].axhline(-FC, color=OK["black"], lw=0.9, ls=":")
    ax[0].axhline(0, color=OK["grey"], lw=0.6)
    ax[0].text(2.5, -FL + 0.02, f"-F_L = -{FL:.3f}", ha="center", va="bottom", fontsize=6.5, color=OK["verm"])
    ax[0].text(2.5, -FC - 0.02, f"-F_C = -{FC:.3f}", ha="center", va="top", fontsize=6.5)
    ax[0].axvspan(5.5, 7.5, color="#EEEEEE", zorder=0)
    ax[0].set_xticks(range(len(order))); ax[0].set_xticklabels([b for _, b in order], fontsize=6.8)
    ax[0].set_xlabel("* not counted for the line (higher KL; refit arm)", fontsize=6.8)
    ax[0].set_xlim(-0.5, 7.5)
    ax[0].set_ylabel("mean capped margin vs shipped (nats)\n99.72 % bootstrap over documents")
    ax[0].set_title("(a) Held-out recall population (B1): six lower-KL maps")
    ax[0].legend(frameon=False, fontsize=6.8, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.27),
                 columnspacing=0.8, handletextpad=0.3)
    # (b) KL-table B (V5c) by depth and family
    box = os.path.join(ROOT, "data/b1/out")
    acc = {}
    for h in range(20):
        S = {(r["depth"], tuple(r["of"]) if r["of"] else None, r["family"], r["slot"]): r
             for r in jl(os.path.join(box, f"shipped-doc{h:02d}.jsonl")) if "min" in r}
        X = {(r["depth"], tuple(r["of"]) if r["of"] else None, r["family"], r["slot"]): r
             for r in jl(os.path.join(box, f"V5c-doc{h:02d}.jsonl")) if "min" in r}
        assert set(S) == set(X)
        for k in S:
            g = (k[0] if k[0] != "twin" else "twin", k[2])
            acc.setdefault(g, []).append(min(X[k]["min"], CAP) - min(S[k]["min"], CAP))
    groups = ["64000", "117000", "245000", "twin"]
    names = ["64k", "117k", "245k", "shallow\ntwins"]
    w = 0.36
    for j, (fam, c, hatch) in enumerate((("needle", OK["verm"], ""), ("copy", OK["blue"], "//"))):
        vals = [float(np.mean(acc[(g, fam)])) for g in groups]
        ax[1].bar(np.arange(4) + (j - 0.5) * w, vals, w, color=c if not hatch else "white", edgecolor=c, hatch=hatch, lw=1.2,
                  label="needles" if fam == "needle" else "copies")
        for i, v in enumerate(vals if fam == "needle" else []):
            ax[1].annotate(f"{v:+.2f}", (i + (j - 0.5) * w, v), xytext=(0, -9 if v < 0 else 2), textcoords="offset points",
                           ha="center", fontsize=6.3)
    ax[1].axhline(0, color=OK["grey"], lw=0.6)
    ax[1].set_xticks(np.arange(4)); ax[1].set_xticklabels(names, fontsize=7)
    ax[1].set_ylim(-3.9, 0.6)
    ax[1].set_ylabel("mean capped margin vs shipped (nats)")
    ax[1].set_title("(b) KL-table B by depth and family")
    ax[1].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig4-b1-recall-population")


# ------------------------------------------------------------------ BF16 per seed (test_challenge)
def fig_bf16():
    seeds = [106, 107, 108, 109]
    ship = [aw_score(("t3-pv3ac" if s < 108 else "w2-pv3ac") + f"-s{s}")[0] for s in seeds]
    bf = [aw_score(f"w2-pbf16-s{s}")[0] for s in seeds]
    ws = [aw_score(f"w2-pv3acB-s{s}")[0] for s in (106, 107)]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.25, 1]})
    x, w = np.arange(4), 0.27
    ax[0].bar(x - w, ship, w, color=OK["black"], edgecolor=OK["black"], lw=1.2, label="shipped 2.47-bit file, 12 GB setup")
    ax[0].bar(x, bf, w, color="white", edgecolor=OK["blue"], hatch="//", lw=1.2, label="BF16")
    ax[0].bar(x[:2] + w, ws, w, color="white", edgecolor=OK["orange"], hatch="xx", lw=1.2, label="shipped file in BF16's setup")
    for xs, vs in ((x - w, ship), (x, bf), (x[:2] + w, ws)):
        for a_, v in zip(xs, vs):
            ax[0].annotate(str(v), (a_, v), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=6.5)
    ax[0].set_xticks(x); ax[0].set_xticklabels([str(s) for s in seeds]); ax[0].set_xlabel("seed")
    ax[0].set_ylim(270, 362); ax[0].set_ylabel("AppWorld test_challenge\ntask-runs (of 417)")
    ax[0].set_title("(a) Score per seed")
    ax[0].legend(frameon=False, fontsize=7, loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.2))
    rows = [("shipped, 12 GB setup\n4 seeds, of 1,668", "shipped vs BF16 (W2-A + W2-A2)", OK["black"], "o"),
            ("shipped in BF16's setup\n2 seeds, of 834", "W2-S vs BF16", OK["orange"], "s"),
            ("shipped, two-stage head off\n2 seeds, of 834", "noH2 vs BF16", OK["green"], "D")]
    for i, (n, key, c, m) in enumerate(rows):
        r = seed_level(key)
        tot, (lo, hi) = r["total"], r["ci"]
        ax[1].errorbar(tot, len(rows) - 1 - i, xerr=[[tot - lo], [hi - tot]], fmt=m, ms=7, color=c, mfc=c, capsize=4)
        ax[1].annotate(f"{tot:+d}  [{lo:+.0f}, {hi:+.0f}]\nseed-level p = {pfmt(r['p_design'])}", (tot, len(rows) - 1 - i),
                       xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7)
    ax[1].axvline(0, color=OK["grey"], lw=0.8, ls=":")
    ax[1].set_yticks(range(len(rows))); ax[1].set_yticklabels([n for n, *_ in rows][::-1], fontsize=7)
    ax[1].set_ylim(-0.6, len(rows) - 0.3); ax[1].set_xlim(-20, 200)
    ax[1].set_xlabel("difference to BF16 (task-runs)\ncluster-robust 95 % interval")
    ax[1].set_title("(b) Lead over BF16")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig2-bf16-per-seed")


if __name__ == "__main__":
    fig5(); fig_bf16(); fig1(); fig6(); fig3(); fig2(); fig4()
    print("figures written to", os.path.abspath(OUT))
