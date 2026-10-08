//! A GGUF v3 reader: metadata, tensor directory, and the bytes of each tensor.

use std::collections::HashMap;
use std::io;

/// A metadata value; arrays keep their elements.
#[derive(Debug)]
pub(crate) enum Value {
    Int(i64),
    Float(f64),
    Bool(#[allow(dead_code)] bool),
    Str(String),
    Array(Vec<Self>),
}

/// One tensor's shape (innermost first, as GGUF stores it), ggml type and
/// byte range in the file.
#[derive(Debug)]
pub(crate) struct Tensor {
    pub(crate) dims: Vec<usize>,
    pub(crate) ty: u32,
    pub(crate) range: std::ops::Range<usize>,
}

/// ggml type ids this example reads.
pub(crate) const F32: u32 = 0;
pub(crate) const Q1_0: u32 = 41;

/// Bytes per element run of a type: (block elements, block bytes).
const fn block_of(ty: u32) -> Option<(usize, usize)> {
    match ty {
        F32 => Some((1, 4)),
        1 => Some((1, 2)),       // F16
        Q1_0 => Some((128, 18)), // f16 scale + 16 bytes of signs
        _ => None,
    }
}

pub(crate) struct Gguf {
    data: Vec<u8>,
    pub(crate) meta: HashMap<String, Value>,
    pub(crate) tensors: HashMap<String, Tensor>,
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

/// `b` (exactly `N` bytes, as `Reader::take` returns) as an array.
fn le<const N: usize>(b: &[u8]) -> [u8; N] {
    let mut a = [0; N];
    a.copy_from_slice(b);
    a
}

fn bad(what: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what.into())
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> io::Result<&[u8]> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.b.len());
        let end = end.ok_or_else(|| bad("truncated GGUF header"))?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(le(self.take(4)?)))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(le(self.take(8)?)))
    }
    fn usize(&mut self) -> io::Result<usize> {
        usize::try_from(self.u64()?).map_err(|_| bad("size overflows usize"))
    }
    fn string(&mut self) -> io::Result<String> {
        let n = self.usize()?;
        Ok(String::from_utf8_lossy(self.take(n)?).into_owned())
    }
    fn value(&mut self, ty: u32) -> io::Result<Value> {
        let v = match ty {
            0 => Value::Int(i64::from(self.take(1)?[0])),
            1 => Value::Int(i64::from(self.take(1)?[0] as i8)),
            2 => Value::Int(i64::from(u16::from_le_bytes(le(self.take(2)?)))),
            3 => Value::Int(i64::from(i16::from_le_bytes(le(self.take(2)?)))),
            4 => Value::Int(i64::from(self.u32()?)),
            5 => Value::Int(i64::from(self.u32()? as i32)),
            6 => Value::Float(f64::from(f32::from_bits(self.u32()?))),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                let (ety, n) = (self.u32()?, self.usize()?);
                let mut items = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    items.push(self.value(ety)?);
                }
                Value::Array(items)
            }
            10 | 11 => Value::Int(self.u64()? as i64),
            12 => Value::Float(f64::from_bits(self.u64()?)),
            _ => return Err(bad(format!("unknown GGUF value type {ty}"))),
        };
        Ok(v)
    }
}

impl Gguf {
    pub(crate) fn load(path: &std::path::Path) -> io::Result<Self> {
        let data = std::fs::read(path)?;
        let mut r = Reader { b: &data, at: 0 };
        if r.take(4)? != b"GGUF" {
            return Err(bad("not a GGUF file"));
        }
        let version = r.u32()?;
        if version != 3 {
            return Err(bad(format!("GGUF version {version}, expected 3")));
        }
        let (n_tensors, n_kv) = (r.usize()?, r.usize()?);
        let mut meta = HashMap::new();
        for _ in 0..n_kv {
            let key = r.string()?;
            let ty = r.u32()?;
            meta.insert(key, r.value(ty)?);
        }
        let mut infos = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = r.string()?;
            let n_dims = r.u32()? as usize;
            let dims = (0..n_dims)
                .map(|_| r.usize())
                .collect::<io::Result<Vec<_>>>()?;
            let (ty, offset) = (r.u32()?, r.usize()?);
            infos.push((name, dims, ty, offset));
        }
        let align = match meta.get("general.alignment") {
            Some(Value::Int(a)) => usize::try_from(*a).map_err(|_| bad("bad alignment"))?,
            _ => 32,
        };
        let start = r.at.next_multiple_of(align);
        let mut tensors = HashMap::new();
        for (name, dims, ty, offset) in infos {
            let (per, bytes) =
                block_of(ty).ok_or_else(|| bad(format!("{name}: unsupported ggml type {ty}")))?;
            let elems: usize = dims.iter().product();
            if !elems.is_multiple_of(per) {
                return Err(bad(format!("{name}: {elems} values is not whole blocks")));
            }
            let begin = start + offset;
            let end = begin + elems / per * bytes;
            if end > data.len() {
                return Err(bad(format!("{name}: data past the end of the file")));
            }
            tensors.insert(
                name,
                Tensor {
                    dims,
                    ty,
                    range: begin..end,
                },
            );
        }
        Ok(Self {
            data,
            meta,
            tensors,
        })
    }

    pub(crate) fn tensor(&self, name: &str) -> io::Result<(&Tensor, &[u8])> {
        let t = self
            .tensors
            .get(name)
            .ok_or_else(|| bad(format!("missing tensor {name}")))?;
        Ok((t, &self.data[t.range.clone()]))
    }

    pub(crate) fn int(&self, key: &str) -> io::Result<usize> {
        match self.meta.get(key) {
            Some(Value::Int(v)) => usize::try_from(*v).map_err(|_| bad(format!("{key} < 0"))),
            _ => Err(bad(format!("missing integer {key}"))),
        }
    }

    pub(crate) fn float(&self, key: &str) -> io::Result<f64> {
        match self.meta.get(key) {
            Some(Value::Float(v)) => Ok(*v),
            _ => Err(bad(format!("missing float {key}"))),
        }
    }

    pub(crate) fn str(&self, key: &str) -> Option<&str> {
        match self.meta.get(key) {
            Some(Value::Str(s)) => Some(s),
            _ => None,
        }
    }

    /// The tokenizer's token strings, by id.
    pub(crate) fn tokens(&self) -> io::Result<Vec<&str>> {
        match self.meta.get("tokenizer.ggml.tokens") {
            Some(Value::Array(items)) => items
                .iter()
                .map(|v| match v {
                    Value::Str(s) => Ok(s.as_str()),
                    _ => Err(bad("non-string token")),
                })
                .collect(),
            _ => Err(bad("missing tokenizer.ggml.tokens")),
        }
    }

    /// The BPE merges, as "left right" strings in rank order.
    pub(crate) fn merges(&self) -> io::Result<Vec<&str>> {
        match self.meta.get("tokenizer.ggml.merges") {
            Some(Value::Array(items)) => items
                .iter()
                .map(|v| match v {
                    Value::Str(s) => Ok(s.as_str()),
                    _ => Err(bad("non-string merge")),
                })
                .collect(),
            _ => Err(bad("missing tokenizer.ggml.merges")),
        }
    }
}
