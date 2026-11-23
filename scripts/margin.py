"""Continuous recall score of retrieval items (coordinator's request, 2026-11-19 ~17:30): teacher-force the correct
answer and record, over the answer tokens, margin = logp(correct token) - logp(best wrong token):
min, sum, mean, and the first answer position where the margin is negative. Runs against a llama-server that is up.

Items:
  --needle V       frozen bench needle (scripts/bench/needle_depth.py: --n 24, seed 7), vault V
  --copy FN        frozen bench copy (code_needle.py: --n 20), function FN
  --wide           all 24 items of tools/v3s/wide_probe.py
Prompt = the Qwen chat template with thinking off, built by hand:
  <|im_start|>user\\n{content}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n
checked against the server once per run on a short uncached text: the hand-built token count must equal prompt_n of
a real chat request (or be one less, if the endpoint adds a BOS); otherwise the script stops. Answer forced: needle = the digits; copy =
"```python\\n" + code + "\\n```". One /completion request per answer token on the cached prompt (prompt given as token
ids), n_probs 20. If the correct token is not in the top 20, its logp is unknown: the margin is then at most
logp(20th) - logp(1st) and the record says "below_top": true (the bound is used in min and sum).

    python -I margin.py --base http://127.0.0.1:18090 --key-file K --out <dir> [--depth 117000] [--tag T] (--needle 8 | --copy _check_tzname | --wide)
"""
import argparse, ast, json, math, os, random, re, sys, time, urllib.request

BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench")
sys.path.insert(0, BENCH)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KEY = ""
BASE = ""


def call(path, body, timeout=3600):
    req = urllib.request.Request(BASE.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def tok(text):
    return call("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]


def chat_prompt_n(content):
    r = call("/v1/chat/completions", {"messages": [{"role": "user", "content": content}], "max_tokens": 1, "temperature": 0,
                                      "chat_template_kwargs": {"enable_thinking": False}})
    return r.get("timings", {}).get("prompt_n")


def templated(content):
    return "<|im_start|>user\n" + content + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def top_logprobs(r):
    """[(id, logprob)] of the first predicted position, from either answer format of llama-server."""
    cp = r["completion_probabilities"][0]
    if "top_logprobs" in cp:
        return [(t["id"], t["logprob"]) for t in cp["top_logprobs"]]
    return [(t["id"], math.log(max(t["prob"], 1e-30))) for t in cp["probs"]]


def margins(content, answer, check_n=None):
    ptoks = tok(templated(content))
    if check_n is not None and len(ptoks) != check_n:
        # the chat endpoint may count a BOS the tokenizer call does not add
        raise SystemExit("template check failed: %d tokens by hand, %d by the chat endpoint" % (len(ptoks), check_n))
    atoks = tok(answer)
    out = []
    for i, a in enumerate(atoks):
        r = call("/completion", {"prompt": ptoks + atoks[:i], "n_predict": 1, "n_probs": 20, "temperature": 0,
                                 "cache_prompt": True, "post_sampling_probs": False})
        tl = top_logprobs(r)
        lp = dict(tl)
        best_wrong = max(l for t, l in tl if t != a)
        if a in lp:
            out.append({"m": lp[a] - best_wrong, "below_top": False})
        else:
            out.append({"m": min(l for _, l in tl) - tl[0][1], "below_top": True})
    m = [x["m"] for x in out]
    neg = [i for i, x in enumerate(m) if x < 0]
    return {"prompt_tokens": len(ptoks), "answer_tokens": len(atoks), "min": min(m), "sum": sum(m), "mean": sum(m) / len(m),
            "first_neg": neg[0] if neg else None, "n_neg": len(neg), "below_top": sum(x["below_top"] for x in out),
            "margins": [round(x, 3) for x in m]}


def frozen_needle(depth, v, n=24):
    import needle_depth as ND
    rng = random.Random(7)
    codes = {k: f"{rng.randrange(1000, 10000)}" for k in range(1, n + 1)}
    n_sent = max(1, depth // 12)
    probe = " ".join(ND.sentence(i) for i in range(n_sent))
    n_sent = max(1, int(n_sent * depth / max(1, ND.n_tokens(BASE, KEY, probe))))
    slots = {int(n_sent * (k - 0.5) / n): k for k in range(1, n + 1)}      # vault k sits at (k - 0.5) / n, as in the bench
    parts = []
    for i in range(n_sent):
        if i in slots:
            parts.append(f"The access code for vault {slots[i]} is {codes[slots[i]]}.")
        parts.append(ND.sentence(i))
    doc = "Facility notes:\n" + " ".join(parts) + "\n\n"
    return doc + f"What is the access code for vault {v}? Answer with the 4 digits only.", codes[v]


def frozen_copy(depth, fn, n=20):
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
    chosen = build(int(depth * 3.6 * depth / max(1, CN.n_tokens(BASE, KEY, ctx))))
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
                        cands.append({"file": name, "fn": node.name, "code": code,
                                      "pos": offset + src.find(code) + len(f"# ===== {name} =====\n")})
        offset += len(block)
    cands = [c for c in cands if seen[c["fn"]] == 1]
    targets = [min(cands, key=lambda c: abs(c["pos"] - len(ctx) * (k + 0.5) / n)) for k in range(n)]
    t = [t for t in targets if t["fn"] == fn][0]
    q = (f"From the file `{t['file']}` above, output the exact source code of the function `{t['fn']}`, "
         "unchanged, in one python code block. Output nothing else.")
    return "Python source files:\n\n" + ctx + "\n\n" + q, "```python\n" + t["code"] + "\n```"


def main():
    global KEY, BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18090")
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--depth", type=int, default=117000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--needle", type=int, default=0)
    ap.add_argument("--copy", default="")
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--seed", type=int, default=2026, help="wide set: 2026 = W1, 2027 = W2")
    ap.add_argument("--expect", default="", help="needle,copy prompt token counts read by the bench (its 'evaluated'); a mismatch is flagged")
    a = ap.parse_args()
    KEY = open(a.key_file).read().strip(); BASE = a.base
    items = []
    if a.needle:
        c, ans = frozen_needle(a.depth, a.needle); items.append(("needle vault %d" % a.needle, c, ans))
    if a.copy:
        c, ans = frozen_copy(a.depth, a.copy); items.append(("copy " + a.copy, c, ans))
    if a.wide:
        import wide_probe as WP
        WP.SEED = a.seed
        doc, codes, insp = WP.needle_doc(BASE, KEY, a.depth)
        for k in range(1, 13):
            items.append(("wide needle inspector %d" % insp[k - 1],
                          doc + f"What is the badge number of inspector {insp[k - 1]}? Answer with the 5 digits only.", codes[k]))
        ctx, targets = WP.code_ctx(BASE, KEY, a.depth)
        for t in targets:
            q = (f"From the file `{t['file']}` above, output the exact source code of the function `{t['fn']}`, "
                 "unchanged, in one python code block. Output nothing else.")
            items.append(("wide copy " + t["fn"], "Python source files:\n\n" + ctx + "\n\n" + q, "```python\n" + t["code"] + "\n```"))
    # template check on a short uncached text: the hand-built prompt must have as many tokens as the chat endpoint reads
    probe_text = "Template check %d: name one colour." % int(time.time())
    n_chat = chat_prompt_n(probe_text)
    n_hand = len(tok(templated(probe_text)))
    print(json.dumps({"template_check": {"chat_prompt_n": n_chat, "by_hand": n_hand}}), flush=True)
    if n_chat not in (n_hand, n_hand + 1):
        raise SystemExit("template check failed")
    for name, content, ans in items:
        t0 = time.time()
        r = margins(content, ans)
        rec = {"tag": a.tag, "item": name, "template_check": [n_chat, n_hand], "wall_s": round(time.time() - t0, 1)}
        rec.update(r)
        if a.expect:
            want = dict(zip(("needle", "copy"), [int(x) if x else None for x in a.expect.split(",")]))
            w = want.get(name.split()[0])
            rec["same_prompt_as_bench"] = (w is None) or (w == r["prompt_tokens"])
        print(json.dumps({k: v for k, v in rec.items() if k != "margins"}), flush=True)
        if a.out:
            with open(os.path.join(a.out, "margins.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
