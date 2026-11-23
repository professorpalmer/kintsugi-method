"""B1 recall population generator (docs/paper/GAPS.md # B1; design written before any run). Deterministic, CPU only.

    python -I -X utf8 tools/b1/b1_gen.py --lib <CPython 3.12 Lib dir> --tokenizer <Qwen3.8 tokenizer.json> \
        --out lab/b1/population.jsonl.gz

Population (fixed by SEED; every choice below is a rule, none is tuned on a model):
- Pool: the CPython standard-library .py files of --lib, minus (a) every file name of the G2 / G3 copy corpus
  (data/bench/pycode.raw: the copy items of every recall gate the selection used), (b) every file name of the W2
  wide-probe corpus (not public; its names that match a stdlib file are in data/b1/excluded_names.txt), (c) site-packages / __pycache__ folders (the
  stdlib test folders are in: the non-test files alone fill only 10 documents), (d) files that do not parse, are
  under 1,000 or over 120,000 chars, or have a line over 300 chars.
- 20 haystack documents: the pool is shuffled with Random(SEED) and dealt into documents in that order, each document
  taking whole files until it has >= 250,000 tokens; no file is in two documents.
- Depths D = 64,000, 117,000, 245,000 tokens of user content. The haystack at depth D is the
  longest line-boundary prefix of the document whose prompt (head + prefix + question) has <= D tokens, so the
  haystacks of one document are nested (one server can reuse the shorter one's cache for the longer one).
- Needles: for each depth D, 4 comment lines "# Note: the access code for locker <name> is <4 digits>." planted in the
  document at token positions 0.10 D, 0.30 D, 0.50 D, 0.70 D (at the next line boundary). Names and codes are drawn
  from Random(SEED + 1); every name is unique in the population. A haystack also holds the needles planted for
  shallower depths (other names, never asked at that depth).
- Copies: for each depth D, up to 4 top-level functions (4-20 lines, no decorator, a name defined once in its
  document) nearest to token positions 0.15 D, 0.35 D, 0.55 D, 0.75 D, inside [0.02 D, 0.92 D] of the haystack.
  No function (by name or by source text) is used twice anywhere in the population. A slot with no candidate is left
  empty and recorded. The generator stops unless >= 100 distinct functions and 20 documents result.
- Questions and answers as the G2 benches: needle "What is the access code for locker <name>? Answer with the 4
  digits only." -> the digits; copy "From the file `<path>` above, output the exact source code of the function
  `<fn>`, unchanged, in one python code block. Output nothing else." -> "```python\\n" + code + "\\n```".
- Shallow control (twins): every item at 117,000 and 245,000 is asked a second time on a short haystack: the item's
  file header line plus the document lines around the item, about 1,900 tokens before the item's line and up to
  3,900 tokens of content in all (the whole function included). Same question, same answer, same map: only the
  distance changes. (A first draft used a nested 4,000-token depth; there only 12 of 80 copy slots had a candidate,
  so the control was replaced by twins before any run.)
Output: gzip JSONL. Line 1 = meta (seed, tokenizer sha256, pool and exclusion counts, every file with its sha256 per
document). Then one line per document: the full document text, the char cut per depth, the measured prompt tokens per
depth, and its items. The sha256 of the output file is printed; the run spec pins it.
"""
import argparse
import ast
import gzip
import hashlib
import json
import os
import random
import re
import sys
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

from tokenizers import Tokenizer

SEED = 20261123
N_DOCS = 20
DOC_TOKENS = 250_000
DEPTHS = (64000, 117000, 245000)
TWIN_DEPTHS = (117000, 245000)
TWIN_BEFORE, TWIN_MAX = 1900, 3900
NEEDLE_R = (0.10, 0.30, 0.50, 0.70)
COPY_R = (0.15, 0.35, 0.55, 0.75)
HEAD = "Python source files:\n\n"
SKIP_DIRS = {"site-packages", "__pycache__"}
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
EXCLUDE = os.path.join(ROOT, "data", "b1", "excluded_names.txt")   # the names of (a) and (b) that match a stdlib file
COLORS = ["amber", "azure", "cedar", "cobalt", "coral", "ember", "flint", "garnet", "ivory", "jade", "maple", "ochre",
          "onyx", "pearl", "quartz", "russet", "sable", "slate", "teal", "umber"]


def needle_q(name):
    return f"What is the access code for locker {name}? Answer with the 4 digits only."


def copy_q(path, fn):
    return (f"From the file `{path}` above, output the exact source code of the function `{fn}`, unchanged, in one "
            "python code block. Output nothing else.")


def corpus_names(path):
    text = open(path, encoding="utf-8").read()
    return {os.path.basename(n) for n in re.findall(r"(?m)^# ===== (.+?) =====$", text)}


def pool(lib, excl):
    out, stats = [], {"seen": 0, "excluded_name": 0, "bad": 0}
    for d, ds, fs in os.walk(lib):
        ds[:] = sorted(x for x in ds if x not in SKIP_DIRS)
        for f in sorted(fs):
            if not f.endswith(".py"):
                continue
            stats["seen"] += 1
            rel = os.path.relpath(os.path.join(d, f), lib).replace("\\", "/")
            if f in excl:
                stats["excluded_name"] += 1
                continue
            try:
                src = open(os.path.join(d, f), encoding="utf-8").read()
                ast.parse(src)
            except Exception:
                stats["bad"] += 1
                continue
            if not (1000 <= len(src) <= 120000) or max(len(x) for x in src.splitlines()) > 300 or "\x00" in src:
                stats["bad"] += 1
                continue
            out.append((rel, src.replace("\r\n", "\n")))
    return out, stats


def functions(rel, src):
    lines = src.splitlines()
    out = []
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.decorator_list:
            n = node.end_lineno - node.lineno + 1
            if 4 <= n <= 20:
                out.append((node.name, node.lineno - 1, "\n".join(lines[node.lineno - 1:node.end_lineno])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tk = Tokenizer.from_file(a.tokenizer)
    ntok = lambda s: len(tk.encode(s, add_special_tokens=False).ids)
    excl = {x.strip() for x in open(EXCLUDE, encoding="utf-8") if x.strip()}
    files, stats = pool(a.lib, excl)
    rng = random.Random(SEED)
    rng.shuffle(files)
    nrng = random.Random(SEED + 1)
    used_names, used_fn, used_code = set(), set(), set()
    docs, k = [], 0
    for h in range(N_DOCS):
        blocks, tot = [], 0
        while tot < DOC_TOKENS:
            if k >= len(files):
                sys.exit(f"pool exhausted at document {h}: {len(files)} files")
            rel, src = files[k]; k += 1
            blk = f"# ===== {rel} =====\n{src}"
            if not blk.endswith("\n"):
                blk += "\n"
            blocks.append((rel, blk)); tot += ntok(blk)
        # lines with their token counts; function starts per line
        lines, owner = [], []
        fstart = {}
        for rel, blk in blocks:
            base = len(lines)
            ls = blk.split("\n")[:-1]
            lines += [x + "\n" for x in ls]; owner += [rel] * len(ls)
            for fn, li, code in functions(rel, blk.split("\n", 1)[1]):
                fstart.setdefault(fn, []).append((base + 1 + li, rel, code))
        ltok = [ntok(x) for x in lines]
        # needles: insert after the line where the running token count first reaches the target
        targets = sorted((int(r * D), D, j) for D in DEPTHS for j, r in enumerate(NEEDLE_R))
        needles = {}
        for pos, D, j in targets:
            while True:
                name = f"{nrng.choice(COLORS)}-{nrng.randrange(100, 1000)}"
                if name not in used_names:
                    break
            used_names.add(name)
            needles[(D, j)] = (pos, name, f"{nrng.randrange(1000, 10000)}")
        ins = {}
        run, ti = 0, 0
        for i, t in enumerate(ltok):
            run += t
            while ti < len(targets) and run >= targets[ti][0]:
                ins.setdefault(i, []).append(targets[ti]); ti += 1
        text_lines, line_tok_end, kind = [], [], []
        run = 0
        orig_to_new = {}
        for i, x in enumerate(lines):
            orig_to_new[i] = len(text_lines)
            text_lines.append(x); run += ltok[i]; line_tok_end.append(run); kind.append(("code", owner[i]))
            for pos, D, j in ins.get(i, []):
                _, name, code = needles[(D, j)]
                nl = f"# Note: the access code for locker {name} is {code}.\n"
                text_lines.append(nl); run += ntok(nl); line_tok_end.append(run); kind.append(("needle", (D, j)))
        stream = "".join(text_lines)
        char_end = []
        c = 0
        for x in text_lines:
            c += len(x); char_end.append(c)
        # a function is usable only if its name is defined once in the whole document
        cands = []
        for fn, occ in fstart.items():
            if len(occ) != 1:
                continue
            li, rel, code = occ[0]
            ni = orig_to_new[li]
            tok_start = line_tok_end[ni] - ltok[li]
            nl = code.count("\n") + 1
            if "".join(text_lines[ni:ni + nl]) != code + "\n":
                continue                   # a needle line was planted inside this function: not a copy target
            cands.append({"fn": fn, "file": rel, "code": code, "tok": tok_start, "line": ni})
        cands.sort(key=lambda x: x["tok"])
        items, cuts, ptoks = [], {}, {}
        longest_q = max([needle_q("x" * 12)] + [copy_q(c["file"], c["fn"]) for c in cands], key=len)
        budget_q = ntok("\n\n" + longest_q)
        for D in DEPTHS:
            lim = D - ntok(HEAD) - budget_q - 8
            n = max(i for i in range(len(text_lines)) if line_tok_end[i] <= lim) + 1
            # exact check with the real tokenizer on the whole prompt; shrink by lines if over
            while ntok(HEAD + stream[:char_end[n - 1]] + "\n\n" + longest_q) > D:
                n -= 1
            cuts[D] = char_end[n - 1]
            ptoks[D] = ntok(HEAD + stream[:cuts[D]])
            hay_tok = line_tok_end[n - 1]
            for j in range(len(NEEDLE_R)):
                pos, name, code = needles[(D, j)]
                ni = kind.index(("needle", (D, j)))
                assert ni < n, (h, D, j)
                items.append({"depth": D, "family": "needle", "slot": j, "name": name, "answer": code,
                              "question": needle_q(name), "tok": line_tok_end[ni], "rel_pos": round(line_tok_end[ni] / hay_tok, 4)})
            for j, r in enumerate(COPY_R):
                tgt = r * hay_tok
                ok = [c for c in cands if c["fn"] not in used_fn and c["code"] not in used_code
                      and 0.02 * hay_tok <= c["tok"] and c["line"] + c["code"].count("\n") + 1 < n
                      and c["tok"] <= 0.92 * hay_tok]
                if not ok:
                    items.append({"depth": D, "family": "copy", "slot": j, "empty": True})
                    continue
                c = min(ok, key=lambda x: (abs(x["tok"] - tgt), x["fn"]))
                used_fn.add(c["fn"]); used_code.add(c["code"])
                items.append({"depth": D, "family": "copy", "slot": j, "name": c["fn"], "file": c["file"],
                              "answer": "```python\n" + c["code"] + "\n```", "question": copy_q(c["file"], c["fn"]),
                              "tok": c["tok"], "rel_pos": round(c["tok"] / hay_tok, 4)})
        # shallow twins (see the docstring): the item's line li, the last line it needs, a window around it
        line_tok = [line_tok_end[0]] + [line_tok_end[i] - line_tok_end[i - 1] for i in range(1, len(line_tok_end))]
        hdr_line = {}
        for i, (kd, own) in enumerate(kind):
            if kd == "code" and own not in hdr_line:
                hdr_line[own] = i          # the first line of a file block is its "# ===== path =====" header
        twins = []
        for it in items:
            if it.get("empty") or it["depth"] not in TWIN_DEPTHS:
                continue
            if it["family"] == "needle":
                li = kind.index(("needle", (it["depth"], it["slot"])))
                last = li
            else:
                body = it["answer"][len("```python\n"):-len("\n```")]
                nlines = body.count("\n") + 1
                li = next(i for i in range(len(text_lines)) if kind[i] == ("code", it["file"])
                          and "".join(text_lines[i:i + nlines]) == body + "\n")
                last = li + nlines - 1
            a0, t = li, 0
            while a0 > 0 and t + line_tok[a0 - 1] <= TWIN_BEFORE:
                a0 -= 1; t += line_tok[a0]
            b, t = a0, 0
            while b < len(text_lines) and (b <= last or t + line_tok[b] <= TWIN_MAX):
                t += line_tok[b]; b += 1
            own = next(kind[i][1] for i in range(a0, b) if kind[i][0] == "code")
            hdr = text_lines[hdr_line[own]] if hdr_line[own] < a0 else ""
            hay = hdr + "".join(text_lines[a0:b])
            if it["family"] == "copy":
                assert body in hay and (it["file"] == own or f"# ===== {it['file']} =====" in hay), (h, it["name"])
            else:
                assert f"locker {it['name']} is {it['answer']}." in hay, (h, it["name"])
            twins.append({"of": [it["depth"], it["family"], it["slot"]], "haystack": hay,
                          "prompt_tokens": ntok(HEAD + hay), "rel_pos": round((line_tok_end[li] - line_tok_end[a0] + line_tok[a0]) / max(1, t), 4)})
        docs.append({"doc": h, "twins": twins, "files": [(rel, hashlib.sha256(blk.encode()).hexdigest()[:16]) for rel, blk in blocks],
                     "stream": stream, "cuts": cuts, "prompt_tokens": ptoks, "items": items})
        print(f"doc {h}: {len(blocks)} files, {line_tok_end[-1]} tokens; cuts {ptoks}; copies "
              f"{sum(1 for x in items if x['family'] == 'copy' and not x.get('empty'))}/12, twins {len(twins)}", flush=True)
    nfun = len(used_fn)
    empty = sum(1 for d in docs for x in d["items"] if x.get("empty"))
    meta = {"meta": True, "seed": SEED, "n_docs": N_DOCS, "depths": DEPTHS, "needle_r": NEEDLE_R, "copy_r": COPY_R,
            "tokenizer_sha256": hashlib.sha256(open(a.tokenizer, "rb").read()).hexdigest(), "python": sys.version,
            "pool": stats, "pool_files": len(files), "files_used": k, "excluded_names": 127,  # the full list of the run had 127 names; only those matching a stdlib file are published (same pool)
           
            "distinct_functions": nfun, "needles": len(used_names), "twins": sum(len(d["twins"]) for d in docs), "empty_copy_slots": empty, "head": HEAD}
    if nfun < 100 or len(docs) != N_DOCS:
        sys.exit(f"population check failed: {nfun} distinct functions, {len(docs)} documents")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as g:
        g.write((json.dumps(meta) + "\n").encode())
        for d in docs:
            g.write((json.dumps(d) + "\n").encode())
    sha = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print(json.dumps({k: v for k, v in meta.items() if k != "head"}))
    print(f"population {a.out} sha256 {sha}")


if __name__ == "__main__":
    main()
