"""B1 probe (docs/paper/GAPS.md # B1): teacher-forced recall margins of one haystack document of the B1 population on
a running Kintsugi llama-server (Qwen3.8-27B). Standard library only (runs on the box with python3 -I).

    python3 -I -X utf8 b1_probe.py --base http://127.0.0.1:18090 --pop population.jsonl.gz --pop-sha <sha256> \
        --doc H --out <file.jsonl> --tag T [--method gf|pt] [--twins-only] [--limit N]

Order (the same for every map): depth 64,000 -> 117,000 -> 245,000 (nested haystacks, so the server's prompt cache
can reuse the shorter prefix), within a depth needles slot 0-3 then copies slot 0-3; then the shallow twins in
population order. Prompt = the server's own rendering (/apply-template) of one user message, thinking off
(chat_template_kwargs enable_thinking false), tokenized by the server.
Method gf (default; as tools/gc2/ink_margin.py score_gf): ONE /completion per item with a grammar that allows exactly
the answer, temperature 0, n_probs 20, post_sampling_probs false. Per generated position: correct = the best top-20
token whose bytes are a prefix of the remaining answer, wrong = the best top-20 token that is not; margin = correct -
wrong; no valid token in the top 20 -> a bound, flagged (below_top). Method pt (smoke check only): one /completion per
answer token (as tools/v3s/margin.py).
Per record: min / mean margin, tokens below 0, copy log-prob (sum of the sampled tokens' log-probs), flags, the
server's prompt_n (cache reuse check), wall time, all margins.
"""
import argparse
import gzip
import hashlib
import json
import math
import time
import urllib.request

BASE, KEY = "", "dummy-not-a-key"
KW = {"enable_thinking": False}
HEAD = "Python source files:\n\n"


def call(path, body, timeout=3600):
    req = urllib.request.Request(BASE.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def tok(text):
    return call("/tokenize", {"content": text, "add_special": False, "parse_special": True})["tokens"]


def render(content):
    return call("/apply-template", {"messages": [{"role": "user", "content": content}], "chat_template_kwargs": KW,
                                    "add_generation_prompt": True})["prompt"]


def gbnf_literal(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t") + '"'


def tbytes(t):
    return bytes(t["bytes"]) if t.get("bytes") is not None else t["token"].encode("utf-8")


def score_gf(P, answer):
    r = call("/completion", {"prompt": P, "n_predict": len(answer) + 16, "n_probs": 20, "temperature": 0,
                             "cache_prompt": True, "post_sampling_probs": False, "grammar": "root ::= " + gbnf_literal(answer)})
    rem = answer.encode("utf-8")
    ms, lps, below, raw_seen = [], [], 0, 0
    for cp in r["completion_probabilities"]:
        if not rem:
            break
        s = tbytes(cp)
        if not s or not rem.startswith(s):
            raise SystemExit("grammar check failed: sampled %r is not a prefix of the remaining answer %r" % (s, rem[:40]))
        tl = [(tbytes(t), t["logprob"]) for t in cp["top_logprobs"]]
        val = [l for b, l in tl if b and rem.startswith(b)]
        inv = [l for b, l in tl if not (b and rem.startswith(b))]
        raw_seen += bool(inv)
        last = min(l for _, l in tl)
        if val and inv:
            ms.append(max(val) - max(inv))
        elif val:
            ms.append(max(val) - last)
        else:
            ms.append(last - max(inv)); below += 1
        lps.append(cp.get("logprob", last))
        rem = rem[len(s):]
    if rem:
        raise SystemExit("grammar check failed: generation ended %d bytes before the end of the answer" % len(rem))
    return ms, lps, below, raw_seen, r.get("timings", {}).get("prompt_n")


def score_pt(P, answer):
    A = tok(answer)
    ms, lps, below, pn = [], [], 0, None
    for i, a in enumerate(A):
        r = call("/completion", {"prompt": P + A[:i], "n_predict": 1, "n_probs": 20, "temperature": 0, "cache_prompt": True,
                                 "post_sampling_probs": False})
        if pn is None:
            pn = r.get("timings", {}).get("prompt_n")
        cp = r["completion_probabilities"][0]
        tl = [(t["id"], t["logprob"]) for t in cp["top_logprobs"]] if "top_logprobs" in cp else \
             [(t["id"], math.log(max(t["prob"], 1e-30))) for t in cp["probs"]]
        lp = dict(tl)
        if a in lp:
            ms.append(lp[a] - max(l for t, l in tl if t != a)); lps.append(lp[a])
        else:
            ms.append(min(l for _, l in tl) - tl[0][1]); lps.append(min(l for _, l in tl)); below += 1
    return ms, lps, below, len(A), pn


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18090")
    ap.add_argument("--pop", required=True)
    ap.add_argument("--pop-sha", required=True)
    ap.add_argument("--doc", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--method", choices=("gf", "pt"), default="gf")
    ap.add_argument("--twins-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N twins only (smoke)")
    a = ap.parse_args()
    BASE = a.base
    sha = hashlib.sha256(open(a.pop, "rb").read()).hexdigest()
    if sha != a.pop_sha:
        raise SystemExit("population sha256 %s != %s" % (sha, a.pop_sha))
    doc = None
    with gzip.open(a.pop) as g:
        for line in g:
            r = json.loads(line)
            if r.get("doc") == a.doc:
                doc = r
                break
    if doc is None:
        raise SystemExit("document %d not in the population" % a.doc)
    probe = "Template check %d: name one colour." % int(time.time())
    n_chat = call("/v1/chat/completions", {"messages": [{"role": "user", "content": probe}], "max_tokens": 1, "temperature": 0,
                                           "chat_template_kwargs": KW}).get("timings", {}).get("prompt_n")
    rendered = render(probe)
    n_hand = len(tok(rendered))
    print(json.dumps({"template_check": {"chat_prompt_n": n_chat, "rendered": n_hand, "ends": rendered[-24:]}}), flush=True)
    if n_chat not in (n_hand, n_hand + 1):
        raise SystemExit("template check failed")
    jobs = []
    if not a.twins_only:
        for D in sorted(int(x) for x in doc["cuts"]):
            hay = doc["stream"][:doc["cuts"][str(D)]]
            for fam in ("needle", "copy"):
                for it in sorted((x for x in doc["items"] if x["depth"] == D and x["family"] == fam), key=lambda x: x["slot"]):
                    jobs.append((str(D), fam, it["slot"], it, hay, None))
    items = {(x["depth"], x["family"], x["slot"]): x for x in doc["items"]}
    tw = doc["twins"][:a.limit] if a.limit else doc["twins"]
    for t in tw:
        it = items[tuple(t["of"])]
        jobs.append(("twin", it["family"], it["slot"], it, t["haystack"], t["of"]))
    for D, fam, slot, it, hay, of in jobs:
        t0 = time.time()
        P = tok(render(HEAD + hay + "\n\n" + it["question"]))
        if a.method == "gf":
            ms, lps, below, raw_seen, pn = score_gf(P, it["answer"])
            extra = {"positions_with_wrong_in_top20": raw_seen}
        else:
            ms, lps, below, nA, pn = score_pt(P, it["answer"])
            extra = {"ref_tokens": nA}
        rec = {"tag": a.tag, "doc": a.doc, "depth": D, "of": of, "family": fam, "slot": slot, "item": it["name"],
               "method": a.method, "prompt_tokens": len(P), "prompt_n": pn, "answer_tokens": len(ms), "min": min(ms),
               "mean": sum(ms) / len(ms), "n_neg": sum(1 for m in ms if m < 0), "logp_sum": sum(lps), "below_top": below,
               "wall_s": round(time.time() - t0, 1)}
        rec.update(extra)
        rec["margins"] = [round(m, 3) for m in ms]
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps({k: v for k, v in rec.items() if k != "margins"}), flush=True)
    print(json.dumps({"tag": a.tag, "done": len(jobs)}), flush=True)


if __name__ == "__main__":
    main()
