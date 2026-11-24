"""Minimal GGUF reader: metadata + tensor data by name (numpy memmap). No gguf-py, so unknown archs load."""
import struct, numpy as np
T = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
NP = {0: np.float32, 1: np.float16, 24: np.int8, 25: np.int16, 26: np.int32, 27: np.int64, 28: np.float64}


class GGUF:
    def __init__(self, path):
        self.path = path
        f = open(path, "rb")
        assert f.read(4) == b"GGUF"
        rd = lambda fmt: struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]
        rstr = lambda: f.read(rd("Q")).decode("utf-8", "replace")

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                et, n = rd("I"), rd("Q")
                self._sub = et
                if et in T:
                    return np.frombuffer(f.read(struct.calcsize("<" + T[et]) * n), dtype="<" + T[et].replace("?", "b"))
                return [rval(et) for _ in range(n)]
            return rd(T[t])
        self.version, nt, nkv = rd("I"), rd("Q"), rd("Q")
        self.kv, self.kv_type, self.kv_sub = {}, {}, {}
        for _ in range(nkv):
            k = rstr(); t = rd("I"); self.kv_type[k] = t; self.kv[k] = rval(t)
            if t == 9:
                self.kv_sub[k] = self._sub
        self.tensors = {}
        for _ in range(nt):
            name = rstr(); nd = rd("I"); dims = [rd("Q") for _ in range(nd)]; ty = rd("I"); off = rd("Q")
            self.tensors[name] = (dims, ty, off)
        align = int(self.kv.get("general.alignment", 32))
        self.data_start = (f.tell() + align - 1) // align * align
        f.close()

    def get(self, name):
        dims, ty, off = self.tensors[name]
        n = int(np.prod(dims))
        a = np.memmap(self.path, dtype=NP[ty], mode="r", offset=self.data_start + off, shape=(n,))
        return a.reshape(dims[::-1])  # numpy order
