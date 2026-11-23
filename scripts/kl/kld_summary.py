"""Summaries of escha_kld reports on the quant-challenge public set (gen rows only, like the leaderboard).

    python -I kld_summary.py <name>=<report.json[,report2.json...]> [<name>=...]

Every name can have more than one report (parts with different documents); rows are merged. Prints, per name: mean KL,
median, p99, top-1 agreement and PPL ratio over the gen rows of the documents that ALL names cover; mean KL by corpus
category; and for each pair (first name = base) the paired difference with a bootstrap interval over documents."""
import json, sys, os, collections
import numpy as np

GD = os.environ.get("QC_ROOT", "quant-challenge")   # the challenge folder (golden/public, corpus/public.jsonl)
man = json.load(open(os.path.join(GD, "golden/public/kld.json")))
gen = np.fromfile(os.path.join(GD, "golden/public/gen.u8"), np.uint8).astype(bool)
cat = [json.loads(l)["category"] for l in open(os.path.join(GD, "corpus/public.jsonl"), encoding="utf-8")]
assert len(cat) == len(man["docs"])
R = man["rows"]
names, data = [], {}
for arg in sys.argv[1:]:
    name, paths = arg.split("=", 1)
    rows = np.full((R, 4), np.nan, np.float32)
    for p in paths.split(","):
        r = np.fromfile(p + ".rows", np.float32).reshape(-1, 4)
        m = ~np.isnan(r[:, 0])
        rows[m] = r[m]
    names.append(name); data[name] = rows
ok = gen.copy()
for n in names:
    ok &= ~np.isnan(data[n][:, 0])
docs = []
for i, d in enumerate(man["docs"]):
    s = slice(d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"])
    g = gen[s]
    if g.any() and ok[s][g].all():
        docs.append(i)
sel = np.zeros(R, bool)
for i in docs:
    d = man["docs"][i]
    sel[d["row0"]:d["row0"] + d["tokens"] - 1 - d["score_from"]] = True
sel &= gen
print(f"documents covered by every report: {len(docs)} of {len(man['docs'])}; gen rows {int(sel.sum()):,} of {int(gen.sum()):,}")
print(f"{'name':28s} {'mean KL':>9s} {'median':>9s} {'p99':>8s} {'top-1':>8s} {'PPL ratio':>9s}")
for n in names:
    r = data[n][sel].astype(np.float64)
    print(f"{n:28s} {r[:, 0].mean():9.5f} {np.median(r[:, 0]):9.6f} {np.quantile(r[:, 0], .99):8.4f} {100 * r[:, 3].mean():7.2f}% "
          f"{np.exp((r[:, 1].sum() - r[:, 2].sum()) / len(r)):9.4f}")
bycat = collections.defaultdict(list)
for i in docs:
    bycat[cat[i]].append(i)


def doc_sums(n, ds):
    out = []
    for i in ds:
        d = man["docs"][i]; s = slice(d["row0"], d["row0"] + d["tokens"] - 1 - d["score_from"])
        g = gen[s]
        out.append((float(data[n][s][g, 0].astype(np.float64).sum()), int(g.sum())))
    return np.array(out)


print("\nmean KL by category (gen rows)")
print(f"{'category':14s} {'docs':>5s} {'rows':>9s} " + " ".join(f"{n[:18]:>18s}" for n in names))
for c in sorted(bycat, key=lambda c: -sum(doc_sums(names[0], bycat[c])[:, 1])):
    ds = bycat[c]
    vals = [doc_sums(n, ds) for n in names]
    print(f"{c:14s} {len(ds):5d} {int(vals[0][:, 1].sum()):9d} " + " ".join(f"{v[:, 0].sum() / v[:, 1].sum():18.5f}" for v in vals))
rng = np.random.default_rng(1)
base = doc_sums(names[0], docs)
for n in names[1:]:
    v = doc_sums(n, docs)
    ratio = v[:, 0].sum() / base[:, 0].sum()
    bs = []
    for _ in range(2000):
        k = rng.integers(0, len(docs), len(docs))
        bs.append(v[k, 0].sum() / base[k, 0].sum())
    lo, hi = np.quantile(bs, [.025, .975])
    better = int((v[:, 0] < base[:, 0]).sum())
    print(f"\n{n} against {names[0]}: mean KL {100 * (ratio - 1):+.1f} % [{100 * (lo - 1):+.1f}, {100 * (hi - 1):+.1f}] "
          f"(bootstrap over documents); lower KL on {better} of {len(docs)} documents")
