"""Free-generation check (paper Section 6, EVIDENCE E5.26): thinking length per item on the 24 needles at 245k.

    python -X utf8 scripts/freegen_summary.py      -> results/freegen.txt

Reads data/freegen/<run>.jsonl (the records of scripts/free_needles.py). Prints, per run: items, correct answers,
replies ended by the model, the thinking length (characters) as minimum / median / maximum, and every item over
1,000 characters. The 1,000-character value is the line that was written before the run; it is not a validated
threshold (see the paper). Read the lengths against the run "bf16" (BF16, f16 K/V, no drafting) and the run of the
shipped file, and read the text against the reference token by token (EVIDENCE E5.31): a length difference alone
does not show where a reply left the reference.
"""
import glob
import json
import os
import statistics

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
ORDER = ["bf16", "shipped", "kl-tools", "kl-tools-repeat", "kl-tools-mtp", "kl-tools-no-drafting", "kl-tools-vault9-first",
         "kl-tools-qkv-back", "kl-tools-ssmout-back"]


def main():
    out = ["Free-generation check: 24 needle questions at 245,000 tokens, default thinking, temperature 0, max 4,096 tokens.",
           "run | items | correct | ended by the model | thinking chars min / median / max | items over 1,000 chars (vault: chars)"]
    files = {os.path.basename(p)[:-6]: p for p in glob.glob(os.path.join(ROOT, "data", "freegen", "*.jsonl"))}
    names = [n for n in ORDER if n in files] + sorted(n for n in files if n not in ORDER)
    for n in names:
        with open(files[n], encoding="utf-8") as f:
            rs = [json.loads(x) for x in f if x.strip()]
        rc = [r["reasoning_chars"] for r in rs]
        over = ", ".join(f"{r['vault']}: {r['reasoning_chars']}" for r in rs if r["reasoning_chars"] > 1000) or "none"
        out.append(f"{n} | {len(rs)} | {sum(bool(r['ok']) for r in rs)} | {sum(r['finish_reason'] == 'stop' for r in rs)} | "
                   f"{min(rc)} / {statistics.median(rc):g} / {max(rc)} | {over}")
    out += ["", "Vault 9 asked alone on a fresh server (one record per map; '<group>-back' = KL-tools with that group at the shipped bits):",
            "map | correct | thinking chars"]
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "freegen", "vault9-alone", "*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            r = [json.loads(x) for x in f if x.strip()][0]
        out.append(f"{os.path.basename(p)[:-6]} | {int(bool(r['ok']))} | {r['reasoning_chars']}")
    t = "\n".join(out)
    print(t)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "freegen.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(t + "\n")


if __name__ == "__main__":
    main()
