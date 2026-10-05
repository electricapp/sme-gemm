# Unpickle a torch zip checkpoint without torch and write its tensors to
# model.safetensors (float32, row-major). The file lands under its final name
# only once complete, so an interrupted run leaves nothing that fetch.sh would
# mistake for a finished conversion.
import pickle, zipfile, collections, json, os, struct

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
header, parts, pos = {}, [], 0
for name, (_, st, off, size, stride) in sd.items():
    assert st.dtype == "FloatStorage", (name, st.dtype)
    n = 1
    for s in size: n *= s
    exp, acc = [], 1
    for s in reversed(size): exp.insert(0, acc); acc *= s
    assert list(stride) == exp or n == 1, (name, size, stride)
    header[name] = {"dtype": "F32", "shape": list(size), "data_offsets": [pos, pos + 4 * n]}
    parts.append((st.key, off, n)); pos += 4 * n
head = json.dumps(header).encode()
head += b" " * (-len(head) % 8)
with open("model.safetensors.part", "wb") as out:
    out.write(struct.pack("<Q", len(head))); out.write(head)
    for key, off, n in parts:
        out.write(zf.read(f"{root}/data/{key}")[off * 4:(off + n) * 4])
os.replace("model.safetensors.part", "model.safetensors")
print(f"converted: {pos // 4} floats")
