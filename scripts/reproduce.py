"""Recompute the statistics and the figures of the paper from data/ (CPU only, a few minutes).

    python -m venv .venv && .venv/bin/pip install -r scripts/requirements.txt      (Windows: .venv\\Scripts\\pip)
    .venv/bin/python -X utf8 scripts/reproduce.py

Writes results/*.txt and paper/figs/*, then checks that every figure is byte-identical to the one in the repository
(the hashes are taken before the figures are redrawn). Exit code 0 only if every step ran and every figure matched.
Numbers that this repository gives only as lab result tables are in data/scores/ (see README.md).
"""
import hashlib
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PY = sys.executable
FIGS = os.path.join(ROOT, "paper", "figs")
STEPS = [
    ("Seed clustering and seed-level tests, test_challenge and Inkling second batch (E5.1, E5.2, E5.4-E5.6, E5.12-E5.14, E5.17, E5.22, E5.23)",
     ["scripts/gg_seed.py"]),
    ("Seed-level tests of the test_normal comparisons (E5.28)", ["scripts/gg_seed_tn.py"]),
    ("Temperature 1.0 minus temperature 0 lead; greedy split of the T 1.0 runs (E6.1, E6.2)", ["scripts/v12_t0.py"]),
    ("AppWorld counts and task-level statistics of v1.0.x (E1.31, E1.32, E1.36, E3.x, E4.8, E4.12)", ["scripts/gg_stats.py"]),
    ("W2-C flip maps on the held-out recall population (E5.15)", ["scripts/w2c_score.py", "--dir", "data/w2c/out"]),
    ("W2-D graded recall of the Inkling baselines (E5.21)", ["scripts/w2d_margin.py"]),
    ("Free-generation check: thinking length per item (E5.26)", ["scripts/freegen_summary.py"]),
    ("G-C2 graded K/V margins (E2.23)", ["scripts/gc2_score.py"]),
    ("G-B2 graded margins of the baselines (E1.33)", ["scripts/gb2_score.py"]),
    ("B1 recall population test (E1.34)", ["scripts/b1/b1_score.py", "--dir", "data/b1/out"]),
    ("B1c same-box confirmation (E1.35)", ["scripts/b1/b1c_score.py", "--dir", "data/b1c/out"]),
]
KL = [("p1", ["V3a", "V3b"]), ("p1", ["V3a", "V3c"]), ("p1", ["V3a", "V3h1"]), ("p1", ["V3a", "V3s2"]),
      ("p2", ["V3a", "V3s2"]), ("p1", ["V3ac-h32", "V5bc-h32"]), ("p1", ["V3ac-h32", "V5t"]),
      ("p1", ["box-V3ac-h32", "box-U1c-h32", "box-H1c-h32"]), ("p2", ["box-V3ac-h32", "box-U1c-h32", "box-H1c-h32"])]


def run(args, out=None):
    r = subprocess.run([PY, "-X", "utf8"] + args, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(f"FAILED: {' '.join(args)}")
    if out:
        with open(os.path.join(ROOT, "results", out), "w", encoding="utf-8", newline="\n") as f:
            f.write(r.stdout)
    return r.stdout


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    for what, args in STEPS:
        run(args)
        print(f"ok   {what}")
    kl = []
    for half, names in KL:
        kl.append(f"$ kl_docs.py {half} {' '.join(names)}\n" + run(["scripts/kl/kl_docs.py", half] + names))
    with open(os.path.join(ROOT, "results", "kl.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(kl))
    print("ok   KL comparisons from per-document sums (E1.1-E1.3, E1.8, E1.9, E1.28-E1.30) -> results/kl.txt")
    before = {f: sha(os.path.join(FIGS, f)) for f in sorted(os.listdir(FIGS))}
    run(["scripts/make_figs.py"])
    after = {f: sha(os.path.join(FIGS, f)) for f in sorted(os.listdir(FIGS))}
    bad = [f for f in before if before[f] != after.get(f)]
    print(f"{'ok  ' if not bad else 'FAIL'} figures: {len(before) - len(bad)} of {len(before)} byte-identical" + (f"; differ: {bad}" if bad else ""))
    if bad:
        raise SystemExit(1)
    print("all steps reproduced; results in results/")


if __name__ == "__main__":
    main()
