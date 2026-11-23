"""KL comparisons of the paper from the per-document sums in data/kl/ (no GPU, no golden set needed).

    python -X utf8 scripts/kl/kl_docs.py <p1|p2> <base> <map> [<map> ...]      e.g.  p1 V3a V3s2

data/kl/<half>-<name>.csv holds, per document of the golden-set half, the number of scored (gen) rows and the sums of
the per-row KL and top-1 agreement, exported from the scorer's row files. This script applies the arithmetic of
scripts/kl/kld_summary.py to them: only documents covered by every named file count; mean KL = sum / rows; the
paired interval is a bootstrap over documents of the ratio of KL sums (numpy default_rng(1), 2,000 resamples, the
maps in the order given, against the first name). With the same names in the same order the numbers equal
kld_summary.py's on the row files.
"""
import csv
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")


def load(half, name):
    with open(os.path.join(ROOT, "data", "kl", f"{half}-{name}.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {int(r["doc"]): (r["category"], int(r["gen_rows"]), float(r["kl_sum"]), float(r["top1_sum"])) for r in rows}


def main():
    half, names = sys.argv[1], sys.argv[2:]
    D = {n: load(half, n) for n in names}
    docs = sorted(set.intersection(*[set(v) for v in D.values()]))
    print(f"{half}: documents covered by every file: {len(docs)}; gen rows {sum(D[names[0]][i][1] for i in docs):,}")
    print(f"{'name':24s} {'mean KL':>9s} {'top-1':>8s}")
    for n in names:
        rows = sum(D[n][i][1] for i in docs)
        print(f"{n:24s} {sum(D[n][i][2] for i in docs) / rows:9.5f} {100 * sum(D[n][i][3] for i in docs) / rows:7.2f}%")
    rng = np.random.default_rng(1)
    base = np.array([[D[names[0]][i][2], D[names[0]][i][1]] for i in docs])
    for n in names[1:]:
        v = np.array([[D[n][i][2], D[n][i][1]] for i in docs])
        ratio = v[:, 0].sum() / base[:, 0].sum()
        bs = []
        for _ in range(2000):
            k = rng.integers(0, len(docs), len(docs))
            bs.append(v[k, 0].sum() / base[k, 0].sum())
        lo, hi = np.quantile(bs, [.025, .975])
        better = int((v[:, 0] < base[:, 0]).sum())
        print(f"{n} against {names[0]}: mean KL {100 * (ratio - 1):+.1f} % [{100 * (lo - 1):+.1f}, {100 * (hi - 1):+.1f}] "
              f"(bootstrap over documents); lower KL on {better} of {len(docs)} documents")


if __name__ == "__main__":
    main()
