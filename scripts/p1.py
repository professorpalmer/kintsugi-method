"""P1 of docs/TASK-DAMAGE-MAP.md: recall margins of ALL G2 items at 117k (24 needles + 20 copies), teacher-forced on
the bench's own prompts with tools/v3s/margin.py's method (one /completion per answer token on the cached prompt, top
20 log-probabilities; margin = logp(correct) - logp(best wrong)). The haystack of each family is the same for every
item (only the question at the end changes), so the server's prompt cache reads it once per family.
Score: S1 = sum over items of min(min margin, 5.0); n_neg = items with min margin < 0.
    python -I p1.py --base http://127.0.0.1:18090 --key-file K --out <dir> --tag T [--depth 117000] [--expect 117319,110992]
Writes <out>/p1.jsonl (one record per item, with all margins) and prints a summary line.
"""
import argparse, json, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import margin as M   # noqa: E402


def copy_targets(depth, n=20):
    """The 20 function names of the frozen copy bench, in bench order (the same selection as margin.frozen_copy)."""
    import ast
    import code_needle as CN
    text = open(CN.CORPUS, encoding="utf-8").read()
    parts = re.split(r"(?m)^# ===== (.+?) =====\n", text)[1:]
    files = list(zip(parts[0::2], parts[1::2]))

    def build(chars):
        out, k = [], 0
        for name, src in files:
            if k >= chars:
                break
            out.append((name, src)); k += len(src)
        return out
    chosen = build(int(depth * 3.6))
    ctx = "".join(f"# ===== {a} =====\n{s}" for a, s in chosen)
    chosen = build(int(depth * 3.6 * depth / max(1, CN.n_tokens(M.BASE, M.KEY, ctx))))
    ctx = "".join(f"# ===== {a} =====\n{s}" for a, s in chosen)
    cands, seen, offset = [], {}, 0
    for name, src in chosen:
        block = f"# ===== {name} =====\n{src}"
        try:
            tree = ast.parse(src)
        except SyntaxError:
            tree = None
        if tree:
            lines = src.splitlines()
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    seen[node.name] = seen.get(node.name, 0) + 1
                    if 4 <= node.end_lineno - node.lineno + 1 <= 20 and not node.decorator_list:
                        code = "\n".join(lines[node.lineno - 1:node.end_lineno])
                        cands.append({"fn": node.name, "pos": offset + src.find(code) + len(f"# ===== {name} =====\n")})
        offset += len(block)
    cands = [c for c in cands if seen[c["fn"]] == 1]
    return [min(cands, key=lambda c: abs(c["pos"] - len(ctx) * (k + 0.5) / n))["fn"] for k in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18090")
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--depth", type=int, default=117000)
    ap.add_argument("--expect", default="117319,110992")
    a = ap.parse_args()
    M.KEY = open(a.key_file).read().strip(); M.BASE = a.base
    os.makedirs(a.out, exist_ok=True)
    probe_text = "Template check %d: name one colour." % int(time.time())
    n_chat, n_hand = M.chat_prompt_n(probe_text), len(M.tok(M.templated(probe_text)))
    if n_chat not in (n_hand, n_hand + 1):
        raise SystemExit("template check failed: chat %s, by hand %s" % (n_chat, n_hand))
    want = [int(x) for x in a.expect.split(",")]
    items = [("needle", "vault %d" % v, lambda v=v: M.frozen_needle(a.depth, v)) for v in range(1, 25)]
    items += [("copy", fn, lambda fn=fn: M.frozen_copy(a.depth, fn)) for fn in copy_targets(a.depth)]
    S, neg, bad_prompt, t_all = 0.0, 0, 0, time.time()
    for fam, name, make in items:
        t0 = time.time()
        content, ans = make()
        r = M.margins(content, ans)
        # the bench's prompt counts are known for vault 8 and _check_tzname only (other items: None, not a flag)
        same = (r["prompt_tokens"] == want[0 if fam == "needle" else 1]) if name in ("vault 8", "_check_tzname") else None
        bad_prompt += (same is False)
        S += min(r["min"], 5.0); neg += r["min"] < 0
        rec = {"tag": a.tag, "family": fam, "item": name, "same_prompt_as_bench": same, "wall_s": round(time.time() - t0, 1)}
        rec.update(r)
        with open(os.path.join(a.out, "p1.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k != "margins"}), flush=True)
    summ = {"tag": a.tag, "summary": True, "S1": round(S, 3), "n_neg": int(neg), "items": len(items),
            "prompts_not_bench": bad_prompt, "wall_s": round(time.time() - t_all, 1)}
    with open(os.path.join(a.out, "p1.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(summ) + "\n")
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
