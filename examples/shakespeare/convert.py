# Unpickle a torch zip checkpoint without torch; write weights.f32 + index.txt
# (name offset_floats dims...), row-major contiguous float32. Both land under
# their final names only once complete, so an interrupted run leaves nothing
# that fetch.sh would mistake for a finished conversion.
import pickle, zipfile, collections, os

zf = zipfile.ZipFile("pytorch_model.bin")
root = [n for n in zf.namelist() if n.endswith("data.pkl")][0].rsplit("/", 1)[0]

class Storage:
    def __init__(self, key, dtype): self.key, self.dtype = key, dtype

class Unp(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod == "torch._utils" and name == "_rebuild_tensor_v2":
            return lambda st, off, size, stride, *a: ("T", st, off, tuple(size), tuple(stride))
        if mod == "torch" and name.endswith("Storage"): return name
        if mod == "collections" and name == "OrderedDict": return collections.OrderedDict
        raise pickle.UnpicklingError(f"{mod}.{name}")
    def persistent_load(self, pid):
        _, stype, key, _loc, _n = pid
        return Storage(key, stype)

sd = Unp(zf.open(f"{root}/data.pkl")).load()
out = open("weights.f32.part", "wb"); idx = open("index.txt.part", "w"); pos = 0
for name, (_, st, off, size, stride) in sd.items():
    assert st.dtype == "FloatStorage", (name, st.dtype)
    n = 1
    for s in size: n *= s
    exp, acc = [], 1
    for s in reversed(size): exp.insert(0, acc); acc *= s
    assert list(stride) == exp or n == 1, (name, size, stride)
    raw = zf.read(f"{root}/data/{st.key}")[off * 4:(off + n) * 4]
    out.write(raw); idx.write(f"{name} {pos} {' '.join(map(str, size))}\n"); pos += n
out.close(); idx.close()
os.replace("weights.f32.part", "weights.f32"); os.replace("index.txt.part", "index.txt")
print(f"converted: {pos} floats")
