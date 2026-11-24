"""Free-generation check (paper Section 6, EVIDENCE E5.26): the 24 needles of bench/needle_depth.py (same builder, rng
seed 7) with FREE generation: no chat_template_kwargs (the launcher's thinking default), temperature 0, max_tokens 4096.
Records per item: the answer, finish_reason, reasoning_content length in characters, usage, timings. The line written
before the first run was: every item reasoning <= 1,000 characters and finish_reason "stop", an answer given. That limit
is not a validated threshold (see the paper): compare the lengths with a reference file's run (freegen_summary.py).
    python -I free_needles.py --base http://127.0.0.1:18190 --depth 245000 --n 24 --out x.jsonl"""
import argparse, json, os, random, re, sys, time, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench"))
from quick_tps import n_tokens  # noqa: E402
from needle_depth import sentence  # noqa: E402  (the same builder pieces)


def post(base, body, timeout):
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer none"})
    t0 = time.time()
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read()), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:18190")
    ap.add_argument("--depth", type=int, default=245000)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", required=True)
    ap.add_argument("--order", default="", help="comma list of vaults in the order to ask (default 1..n)")
    ap.add_argument("--only", type=int, default=0, help="ask only this vault (same document)")
    a = ap.parse_args()
    key = "none"
    rng = random.Random(7)
    codes = {v: f"{rng.randrange(1000, 10000)}" for v in range(1, a.n + 1)}
    n_sent = max(1, a.depth // 12)
    probe = " ".join(sentence(i) for i in range(n_sent))
    n_sent = max(1, int(n_sent * a.depth / max(1, n_tokens(a.base, key, probe))))
    slots = {int(n_sent * (k + 0.5) / a.n): v for k, v in enumerate(range(1, a.n + 1))}
    parts = []
    for i in range(n_sent):
        if i in slots:
            v = slots[i]
            parts.append(f"The access code for vault {v} is {codes[v]}.")
        parts.append(sentence(i))
    doc = "Facility notes:\n" + " ".join(parts) + "\n\n"
    ok = over = length = noans = 0
    maxr = 0
    vaults = [int(x) for x in a.order.split(",")] if a.order else list(range(1, a.n + 1))
    if a.only:
        vaults = [a.only]
    for v in vaults:
        body = {"messages": [{"role": "user", "content": doc + f"What is the access code for vault {v}? Answer with the 4 digits only."}],
                "max_tokens": a.max_tokens, "temperature": 0}
        try:
            r, dt = post(a.base, body, a.timeout)
        except Exception as e:  # a timeout or HTTP error is a failed item
            rec = {"vault": v, "error": repr(e)[:300]}
            print(json.dumps(rec), flush=True)
            with open(a.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            noans += 1
            continue
        ch = r["choices"][0]
        msg = ch.get("message", {})
        ans = msg.get("content") or ""
        rc = msg.get("reasoning_content") or ""
        got = re.findall(r"\d{4}", ans)
        hit = bool(got) and got[0] == codes[v]
        ok += hit
        over += len(rc) > 1000
        length += ch.get("finish_reason") != "stop"
        noans += not got
        maxr = max(maxr, len(rc))
        rec = {"depth": a.depth, "vault": v, "want": codes[v], "answer": ans.strip()[:80], "ok": hit,
               "finish_reason": ch.get("finish_reason"), "reasoning_chars": len(rc), "reasoning_head": rc[:300],
               "reasoning_tail": rc[-300:], "think_tag_in_content": ("<think>" in ans) or ("</think>" in ans),
               "reasoning_content": rc, "content": ans,
               "usage": r.get("usage"), "timings": r.get("timings"), "wall_s": round(dt, 1)}
        print(json.dumps({k: rec[k] for k in ("vault", "want", "answer", "ok", "finish_reason", "reasoning_chars", "wall_s")}), flush=True)
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    line = over == 0 and length == 0 and noans == 0
    print(json.dumps({"depth": a.depth, "correct": ok, "of": len(vaults), "reasoning_over_1000": over, "not_stop": length,
                      "no_answer": noans, "max_reasoning_chars": maxr, "line": "PASS" if line else "FAIL"}), flush=True)


if __name__ == "__main__":
    main()
