"""Tensor-by-tensor byte diff of two GGUF files: names, types, shapes and data sha256. Use it to read the real
difference between two bit maps from the model files (EVIDENCE E5.12, E5.19); a map table can be wrong.
    python -X utf8 scripts/gguf_tensor_diff.py <a.gguf> <b.gguf>"""
import hashlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ggufio import GGUF  # noqa: E402


def spans(g, size):
    order = sorted(g.tensors, key=lambda n: g.tensors[n][2])
    ends = [g.tensors[order[i + 1]][2] for i in range(len(order) - 1)] + [size - g.data_start]
    return {n: (g.tensors[n][2], e) for n, e in zip(order, ends)}


def h(path, g, a, b):
    s = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(g.data_start + a); left = b - a
        while left > 0:
            c = f.read(min(left, 1 << 24)); s.update(c); left -= len(c)
    return s.hexdigest()


A, B = sys.argv[1], sys.argv[2]
ga, gb = GGUF(A), GGUF(B)
sa, sb = spans(ga, os.path.getsize(A)), spans(gb, os.path.getsize(B))
assert set(ga.tensors) == set(gb.tensors), "tensor sets differ"
diff = []
for n in sorted(ga.tensors):
    da, ta, _ = ga.tensors[n]; db, tb, _ = gb.tensors[n]
    if da != db or ta != tb or h(A, ga, *sa[n]) != h(B, gb, *sb[n]):
        diff.append(n)
proj = sorted({n.rsplit(".", 1)[0] for n in diff})
print(f"tensor diff {os.path.basename(A)} vs {os.path.basename(B)}: {len(ga.tensors)} tensors, {len(diff)} differ: {diff}")
print(f"projections that differ ({len(proj)}): {proj}")
mk = sorted(k for k in set(ga.kv) | set(gb.kv) if str(ga.kv.get(k)) != str(gb.kv.get(k)))
print(f"metadata keys that differ: {mk}")
