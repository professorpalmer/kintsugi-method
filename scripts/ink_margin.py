"""G-C2 (docs/paper/GAPS.md): graded recall for Inkling 2 Mini. The 44 items of the G3 benches (24 needles of
ink_needle_depth.py --n 24, 20 copies of ink_code_needle.py --n 20; the same documents, questions and order, rebuilt
here), teacher-forced on a running llama-server:
  prompt  = the server's own rendering (/apply-template) of the bench request: one user message, chat_template_kwargs
            {"reasoning_effort": "none"}, as the bench sends it; tokenized by the server (parse_special).
  forced  = the template's rendering of an assistant reply that is exactly the answer (the needle's 4 digits; the copy
            as "```python\\n" + code + "\\n```"), i.e. the text the template appends after the generation prompt.
            Only the ANSWER tokens are scored (the format tokens before the answer are forced, not scored).
  per answer token t: one /completion on the cached prompt + forced[:t] (token ids), n_probs 20, temperature 0;
            margin = logp(correct) - logp(best wrong); if the correct token is not in the top 20, its logp is unknown:
            margin = logp(20th) - logp(1st) (a bound, flagged), logp(correct) <= logp(20th) (a bound, flagged).
Per item: min margin, mean margin, n tokens < 0, sum of logp(correct) ("copy logprob"), flags. The forced text is the
same for every arm (it comes from the template and the answer only), so items pair across arms.
    python -I ink_margin.py --base http://127.0.0.1:18090 --out <file.jsonl> --tag T --depth D --family needle|copy [--items a:b]
"""
import argparse, ast, json, math, os, random, re, sys, time, urllib.request

BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench")
sys.path.insert(0, BENCH)
from quick_tps import n_tokens  # noqa: E402
import ink_needle_depth as ND  # noqa: E402
import ink_code_needle as CN  # noqa: E402

BASE, KEY = "", "dummy-not-a-key"
KW = {"reasoning_effort": "none"}


def call(path, body, timeout=3600):
    req = urllib.request.Request(BASE.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def tok(text):
    return call("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]


def render(messages, gen=True):
    return call("/apply-template", {"messages": messages, "chat_template_kwargs": KW, "add_generation_prompt": gen})["prompt"]


def needle_items(depth, n=24):
    rng = random.Random(7)
    codes = {v: f"{rng.randrange(1000, 10000)}" for v in range(1, n + 1)}
    n_sent = max(1, depth // 12)
    probe = " ".join(ND.sentence(i) for i in range(n_sent))
    n_sent = max(1, int(n_sent * depth / max(1, n_tokens(BASE, KEY, probe))))
    slots = {int(n_sent * (k + 0.5) / n): v for k, v in enumerate(range(1, n + 1))}
    parts = []
    for i in range(n_sent):
        if i in slots:
            v = slots[i]
            parts.append(f"The access code for vault {v} is {codes[v]}.")
        parts.append(ND.sentence(i))
    doc = "Facility notes:\n" + " ".join(parts) + "\n\n"
    return [(f"vault {v}", doc + f"What is the access code for vault {v}? Answer with the 4 digits only.", codes[v])
            for v in range(1, n + 1)]


def copy_items(depth, n=20):
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
    chosen = build(int(depth * 3.6 * depth / max(1, n_tokens(BASE, KEY, ctx))))
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
    out = []
    for t in targets:
        q = (f"From the file `{t['file']}` above, output the exact source code of the function `{t['fn']}`, "
             "unchanged, in one python code block. Output nothing else.")
        out.append((t["fn"], "Python source files:\n\n" + ctx + "\n\n" + q, "```python\n" + t["code"] + "\n```"))
    return out


def top(r):
    cp = r["completion_probabilities"][0]
    if "top_logprobs" in cp:
        return [(t["id"], t["logprob"]) for t in cp["top_logprobs"]]
    return [(t["id"], math.log(max(t["prob"], 1e-30))) for t in cp["probs"]]


def forced_tokens(content, answer):
    """prompt ids, format ids (forced, not scored), answer ids (scored)."""
    msgs = [{"role": "user", "content": content}]
    p_text = render(msgs, True)
    full = render(msgs + [{"role": "assistant", "content": answer}], False)
    if not full.startswith(p_text):
        # some templates render a past assistant turn differently from the generation prompt: use the common prefix
        k = os.path.commonprefix([p_text, full])
        raise SystemExit("rendering check failed: the reply rendering does not extend the generation prompt "
                         "(common prefix %d of %d chars; tail %r)" % (len(k), len(p_text), p_text[len(k):][:80]))
    rest = full[len(p_text):]
    i = rest.find(answer)
    if i < 0:
        raise SystemExit("rendering check failed: the answer is not in the rendered reply: %r" % rest[:200])
    fmt = rest[:i]
    P = tok(p_text)
    F = tok(fmt) if fmt else []
    FA = tok(fmt + answer)
    if FA[:len(F)] != F:
        raise SystemExit("tokenization check failed at the format / answer boundary: %r" % fmt)
    return P, F, FA[len(F):], fmt


def gbnf_literal(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t") + '"'


def tbytes(t):
    return bytes(t["bytes"]) if t.get("bytes") is not None else t["token"].encode("utf-8")


def score_gf(content, answer):
    """Grammar-forced decode (method gf, 11-22 15:2x): ONE /completion per item on the cached prompt + format, with a
    grammar that allows exactly the answer text; temperature 0, n_probs 20, post_sampling_probs false (the server's
    probabilities before the samplers and the grammar). Per generated position: correct = the best top-20 token whose
    bytes are a prefix of the remaining answer, wrong = the best top-20 token that is not; margin = correct - wrong.
    Tokenization-free: any split of the answer counts as correct. Bounds as in method pt, flagged."""
    P, F, A, fmt = forced_tokens(content, answer)
    g = "root ::= " + gbnf_literal(answer)
    r = call("/completion", {"prompt": P + F, "n_predict": len(A) + 16, "n_probs": 20, "temperature": 0, "cache_prompt": True,
                             "post_sampling_probs": False, "grammar": g})
    rem = answer.encode("utf-8")
    ms, lps, below, raw_seen, n = [], [], 0, 0, 0
    for cp in r["completion_probabilities"]:
        if not rem:
            break
        s = tbytes(cp)
        if not rem.startswith(s) or not s:
            raise SystemExit("grammar check failed: sampled %r is not a prefix of the remaining answer %r" % (s, rem[:40]))
        tl = [(tbytes(t), t["logprob"]) for t in cp["top_logprobs"]]
        val = [l for b, l in tl if b and rem.startswith(b)]
        inv = [l for b, l in tl if not (b and rem.startswith(b))]
        raw_seen += bool(inv)
        last = min(l for _, l in tl)
        if val and inv:
            ms.append(max(val) - max(inv))
        elif val:                                   # all 20 valid: the best wrong is at most the 20th
            ms.append(max(val) - last)
        else:                                       # no valid token in the top 20: a bound, flagged
            ms.append(last - max(inv)); below += 1
        lps.append(cp["logprob"] if "logprob" in cp else last)
        rem = rem[len(s):]; n += 1
    if rem:
        raise SystemExit("grammar check failed: generation ended %d bytes before the end of the answer" % len(rem))
    return {"prompt_tokens": len(P), "format": fmt, "format_tokens": len(F), "answer_tokens": n, "ref_tokens": len(A),
            "min": min(ms), "mean": sum(ms) / len(ms), "n_neg": sum(1 for m in ms if m < 0), "logp_sum": sum(lps),
            "below_top": below, "positions_with_wrong_in_top20": raw_seen, "method": "gf", "margins": [round(m, 3) for m in ms]}


def score(content, answer):
    P, F, A, fmt = forced_tokens(content, answer)
    ms, lps, below = [], [], 0
    for i, a in enumerate(A):
        r = call("/completion", {"prompt": P + F + A[:i], "n_predict": 1, "n_probs": 20, "temperature": 0,
                                 "cache_prompt": True, "post_sampling_probs": False})
        tl = top(r)
        lp = dict(tl)
        if a in lp:
            ms.append(lp[a] - max(l for t, l in tl if t != a)); lps.append(lp[a])
        else:
            ms.append(min(l for _, l in tl) - tl[0][1]); lps.append(min(l for _, l in tl)); below += 1
    return {"prompt_tokens": len(P), "format": fmt, "format_tokens": len(F), "answer_tokens": len(A),
            "min": min(ms), "mean": sum(ms) / len(ms), "n_neg": sum(1 for m in ms if m < 0), "logp_sum": sum(lps),
            "below_top": below, "margins": [round(m, 3) for m in ms]}


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18090")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--family", choices=("needle", "copy"), required=True)
    ap.add_argument("--items", default="", help="a:b slice of the family's items (ticket split)")
    ap.add_argument("--method", choices=("gf", "pt"), default="gf", help="gf = grammar-forced decode (default); pt = per token")
    a = ap.parse_args()
    BASE = a.base
    # template check: the server's rendering read by the chat endpoint has as many tokens as our tokenization
    probe = "Template check %d: name one colour." % int(time.time())
    n_chat = call("/v1/chat/completions", {"messages": [{"role": "user", "content": probe}], "max_tokens": 1,
                                           "temperature": 0, "chat_template_kwargs": KW}).get("timings", {}).get("prompt_n")
    n_hand = len(tok(render([{"role": "user", "content": probe}])))
    print(json.dumps({"template_check": {"chat_prompt_n": n_chat, "rendered": n_hand}}), flush=True)
    if n_chat not in (n_hand, n_hand + 1):
        raise SystemExit("template check failed")
    items = needle_items(a.depth) if a.family == "needle" else copy_items(a.depth)
    if a.items:
        s, e = (int(x) if x else None for x in a.items.split(":"))
        items = items[s:e]
    for name, content, ans in items:
        t0 = time.time()
        r = score_gf(content, ans) if a.method == "gf" else score(content, ans)
        rec = {"tag": a.tag, "depth": a.depth, "family": a.family, "item": name, "wall_s": round(time.time() - t0, 1)}
        rec.update(r)
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k != "margins"}), flush=True)
    print(json.dumps({"tag": a.tag, "done": len(items)}), flush=True)


if __name__ == "__main__":
    main()
