"""Figures of the Kintsugi-method paper, from data already in the record only (no new runs). Deterministic.
    python -X utf8 scripts/make_figs.py      -> paper/figs/fig{1..5}-<name>.{png,pdf} (numbered in order of appearance)

Sources by output file (the paper numbers figures by the file prefix; every number is also in evidence/EVIDENCE.md):
- fig2-kl-vs-task: KL change and recall-gate results per map (E1.3, E1.6-E1.9, E1.28, E1.29); AppWorld paired differences with
  bootstrap intervals (data/ = scripts/gg_stats.py; E3.17).
- fig5-seed-pairs: AppWorld counts per seed pair (E3.1-E3.8, E3.15, E3.16) and the shipped - BF16 gap with intervals (E3.17).
- fig4-item-margins: per-item capped margins, data/ (G-B2, E1.33) and data/ (G-C2,
  E2.23), paired by request order as the two scorers do.
- fig6-inkling-appworld: Inkling AppWorld per seed from data/ (E4.8, E4.12, E3.12,
  E3.13).
- fig1-workflow: the workflow (no data).
- fig3-b1-recall-population: data/ (scripts/b1/b1_score.py) and data/ (KL-table B = V5c by depth; E1.34).
Style: light theme, Okabe-Ito colours (colour-blind safe) plus distinct markers / hatches (readable in grayscale).
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
    ax[0].set_title("(a) Recall gate: lower KL did not mean full recall")
    ax[0].legend(loc="upper center", frameon=False, fontsize=7, ncol=2, bbox_to_anchor=(0.5, -0.3),
                 columnspacing=0.8, handletextpad=0.3)
    # (b) AppWorld paired difference vs shipped: selection seeds vs fresh seeds 104 + 105
    rows = [("KL-only winner", -7.0, (-17, -30, -3), (4, -10, 18)), ("KL-tools", -7.9, (-1, -14, 12), (7, -5, 20))]
    for i, (n, x, s, f) in enumerate(rows):
        ax[1].errorbar(i - 0.12, s[0], yerr=[[s[0] - s[1]], [s[2] - s[0]]], fmt="o", ms=7, color=OK["verm"], mfc=OK["verm"],
                       capsize=3, label="selection seeds 100 + 101" if i == 0 else None)
        ax[1].errorbar(i + 0.12, f[0], yerr=[[f[0] - f[1]], [f[2] - f[0]]], fmt="s", ms=7, color=OK["blue"], mfc="white",
                       mew=1.6, capsize=3, label="fresh seeds 104 + 105" if i == 0 else None)
    ax[1].axhline(0, color=OK["grey"], lw=0.8, ls=":")
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["KL-only winner\n(KL -7.0 %)", "KL-tools\n(KL -7.9 %)"])
    ax[1].set_xlim(-0.6, 1.6)
    ax[1].set_ylabel("AppWorld task-runs vs shipped\n(of 336; 95 % paired bootstrap)")
    ax[1].set_title("(b) Standard split: selection vs fresh seeds")
    ax[1].legend(loc="upper center", frameon=False, ncol=1, bbox_to_anchor=(0.5, -0.3))
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig2-kl-vs-task")


# ------------------------------------------------------------------ Fig 2
def fig2():
    pairs = ["100 + 101\n(selection)", "102 + 103\n(fresh)", "104 + 105\n(fresh)"]
    arms = [("shipped (selected)", [302, 294, 292], OK["black"], "*", "-"),
            ("BF16 (never selected)", [281, 293, 294], OK["blue"], "s", "-"),
            ("KL-tools", [301, np.nan, 299], OK["orange"], "^", "--"),
            ("author's fits", [288, np.nan, 289], OK["green"], "v", "--"),
            ("KL-only winner", [285, np.nan, 296], OK["verm"], "D", "--")]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    x = np.arange(3)
    for n, v, c, m, ls in arms:
        v = np.array(v, dtype=float)
        ok = ~np.isnan(v)
        ax[0].plot(x[ok], v[ok], ls=ls, color=c, marker=m, ms=7 if m != "*" else 11, mfc=c if m in "*s" else "white",
                   mew=1.5, label=n)
    ax[0].set_xticks(x); ax[0].set_xticklabels(pairs)
    ax[0].set_ylabel("AppWorld test_normal task-runs (of 336)")
    ax[0].set_title("(a) Score per seed pair")
    ax[0].legend(frameon=False, fontsize=7, loc="lower right", ncol=1)
    g = [(21, 8, 34, 0.0025), (1, -14, 17, 1.0), (-2, -16, 11, 0.89)]
    for i, (p, lo, hi, pv) in enumerate(g):
        ax[1].errorbar(i, p, yerr=[[p - lo], [hi - p]], fmt="o", ms=8, color=OK["black"] if i == 0 else OK["blue"],
                       mfc=OK["black"] if i == 0 else "white", mew=1.6, capsize=4)
        ax[1].annotate(f"{p:+d}\np = {pv:g}", (i, hi), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7.5)
    ax[1].axhline(0, color=OK["grey"], lw=0.8, ls=":")
    ax[1].set_xticks(x); ax[1].set_xticklabels(pairs); ax[1].set_xlim(-0.5, 2.5); ax[1].set_ylim(-22, 46)
    ax[1].set_ylabel("shipped - BF16 (task-runs)\n95 % paired bootstrap")
    ax[1].set_title("(b) The +21 lead on the selection seeds")
    fig.tight_layout()
    save(fig, "fig5-seed-pairs")


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
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for t, c, m, n in (("U1", OK["orange"], "o", "uniform baseline"), ("H1", OK["blue"], "s", "HAWQ-style baseline")):
        v = np.sort(gb[t])
        ax[0].plot(np.arange(len(v)), v, ls="-", lw=1, color=c, marker=m, ms=3.5, mfc="white" if m == "o" else c,
                   label=f"{n}: {int(np.sum(v != 0))} of 44 items differ")
    ax[0].axhline(0, color=OK["grey"], lw=0.8, ls=":")
    ax[0].set_xlabel("items, sorted by difference (Qwen, 117k)")
    ax[0].set_ylabel("capped item margin vs shipped (nats)")
    ax[0].set_title("(a) Qwen baselines: the change sits in few items")
    ax[0].legend(frameon=False, fontsize=7, loc="upper left")
    for arm, c, m, n in (("b", OK["verm"], "^", "blanket q4_0"), ("c", OK["green"], "v", "blanket q8_0"),
                         ("d", OK["purple"], "D", "q4_0 global + f16 SWA")):
        v = np.sort(gc[arm])
        ax[1].plot(np.arange(len(v)), v, ls="-", lw=1, color=c, marker=m, ms=3, mfc="white", mew=1,
                   label=f"{n}: mean {v.mean():+.3f}")
    ax[1].axhline(0, color=OK["grey"], lw=0.8, ls=":")
    ax[1].set_xlabel("item-depth units, sorted (Inkling, 117k + 245k)")
    ax[1].set_ylabel("capped item margin vs release K/V (nats)")
    ax[1].set_title("(b) Inkling K/V settings: spread, no shift")
    ax[1].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig4-item-margins")


# ------------------------------------------------------------------ Fig 4
def fig4():
    aw = os.path.join(ROOT, "data/appworld")
    arms = [("M2, release K/V (ours)", "inkm", OK["blue"], ""), ("M2, 4-bit global cache", "q65", OK["sky"], "//"),
            ("official weights (NVFP4 / MXFP8), same engine", "raw", OK["black"], ".."),
            ("Unsloth's setup (setup vs setup)", "ink", OK["orange"], "xx")]
    seeds = [102, 103, 104, 105]
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 2.7))
    w = 0.2
    for i, (n, key, c, h) in enumerate(arms):
        vals, tots = [], []
        for s in seeds:
            with open(os.path.join(aw, f"{key}-s{s}.csv"), encoding="utf-8") as f:
                rows = [x.strip().split(",") for x in f.readlines()[1:] if x.strip()]
            vals.append(sum(int(r[1]) for r in rows)); tots.append(len(rows))
        tot = sum(vals)
        ax.bar(np.arange(4) + (i - 1.5) * w, vals, w, color=c if not h else "white", edgecolor=c, hatch=h, lw=1.2,
               label=f"{n}: {tot} / {sum(tots)}")
    ax.set_xticks(np.arange(4)); ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_ylim(125, 158); ax.set_ylabel("AppWorld task-runs (of 168)")
    ax.set_title("Inkling 2 Mini, AppWorld test_normal, seeds 102-105 (none used for a choice)")
    ax.legend(frameon=False, ncol=2, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "fig6-inkling-appworld")


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
    save(fig, "fig3-b1-recall-population")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6()
    print("figures written to", os.path.abspath(OUT))
