//! Qwen's byte-level BPE tokenizer, from the GGUF's own vocabulary and merges.

use std::collections::HashMap;
use std::io;

use crate::gguf::{Gguf, Value};

pub(crate) struct Tokenizer {
    ids: HashMap<String, u32>,
    ranks: HashMap<String, usize>,
    /// Each token's bytes (empty for control tokens).
    bytes: Vec<Vec<u8>>,
    /// Control tokens by text, longest first, for splitting them out.
    specials: Vec<(String, u32)>,
    /// GPT-2's byte -> char map, and its inverse.
    byte_char: [char; 256],
    char_byte: HashMap<char, u8>,
}

/// GPT-2's reversible map from bytes to printable chars.
fn byte_chars() -> [char; 256] {
    let mut out = ['\0'; 256];
    let mut next = 256u32;
    for b in 0..=255u8 {
        let printable = (b'!'..=b'~').contains(&b) || (0xa1..=0xac).contains(&b) || b >= 0xae;
        out[usize::from(b)] = if printable {
            char::from(b)
        } else {
            let c = char::from_u32(next).unwrap_or('\0');
            next += 1;
            c
        };
    }
    out
}

impl Tokenizer {
    pub(crate) fn from_gguf(g: &Gguf) -> io::Result<Self> {
        let tokens = g.tokens()?;
        let types: Vec<i64> = match g.meta.get("tokenizer.ggml.token_type") {
            Some(Value::Array(t)) => t
                .iter()
                .map(|v| if let Value::Int(i) = v { *i } else { 1 })
                .collect(),
            _ => vec![1; tokens.len()],
        };
        let byte_char = byte_chars();
        let char_byte: HashMap<char, u8> = (0..=255u8)
            .map(|b| (byte_char[usize::from(b)], b))
            .collect();
        let mut ids = HashMap::with_capacity(tokens.len());
        let mut bytes = Vec::with_capacity(tokens.len());
        let mut specials = Vec::new();
        for (i, t) in tokens.iter().enumerate() {
            let id = u32::try_from(i).map_err(|_| io::Error::other("vocabulary too large"))?;
            ids.insert((*t).to_string(), id);
            // 3 = control, 4 = user-defined: matched as whole strings.
            if matches!(types.get(i), Some(3 | 4)) {
                specials.push(((*t).to_string(), id));
                bytes.push(if types[i] == 4 {
                    t.as_bytes().to_vec()
                } else {
                    Vec::new()
                });
            } else {
                bytes.push(
                    t.chars()
                        .filter_map(|c| char_byte.get(&c).copied())
                        .collect(),
                );
            }
        }
        specials.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        let ranks = g
            .merges()?
            .iter()
            .enumerate()
            .map(|(r, m)| ((*m).to_string(), r))
            .collect();
        Ok(Self {
            ids,
            ranks,
            bytes,
            specials,
            byte_char,
            char_byte,
        })
    }

    /// The token's bytes, empty for a control token.
    pub(crate) fn bytes(&self, id: u32) -> &[u8] {
        self.bytes.get(id as usize).map_or(&[], Vec::as_slice)
    }

    pub(crate) fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // The earliest control token, then the plain text before it.
            let next = self
                .specials
                .iter()
                .filter_map(|(s, id)| rest.find(s.as_str()).map(|at| (at, s.len(), *id)))
                .min_by_key(|&(at, len, _)| (at, std::cmp::Reverse(len)));
            let (plain, special) = match next {
                Some((at, len, id)) => (&rest[..at], Some((len, id, at))),
                None => (rest, None),
            };
            for piece in pretokenize(plain) {
                self.bpe(piece, &mut out);
            }
            match special {
                Some((len, id, at)) => {
                    out.push(id);
                    rest = &rest[at + len..];
                }
                None => break,
            }
        }
        out
    }

    fn bpe(&self, piece: &str, out: &mut Vec<u32>) {
        let mut syms: Vec<String> = piece
            .bytes()
            .map(|b| self.byte_char[usize::from(b)].to_string())
            .collect();
        loop {
            let best = (0..syms.len().saturating_sub(1))
                .filter_map(|i| {
                    self.ranks
                        .get(&format!("{} {}", syms[i], syms[i + 1]))
                        .map(|&r| (r, i))
                })
                .min();
            let Some((_, i)) = best else { break };
            let right = syms.remove(i + 1);
            syms[i].push_str(&right);
        }
        for s in syms {
            match self.ids.get(&s) {
                Some(&id) => out.push(id),
                // Every byte is a token, so this only happens for broken vocabularies.
                None => out.extend(s.chars().filter_map(|c| {
                    self.char_byte
                        .get(&c)
                        .and_then(|b| self.ids.get(&self.byte_char[usize::from(*b)].to_string()))
                        .copied()
                })),
            }
        }
    }
}

/// Splits text as Qwen2's pre-tokenizer pattern does:
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
///  ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
fn pretokenize(text: &str) -> Vec<&str> {
    let cs: Vec<(usize, char)> = text.char_indices().collect();
    let at = |i: usize| cs.get(i).map(|&(_, c)| c);
    let byte = |i: usize| cs.get(i).map_or(text.len(), |&(b, _)| b);
    let is_l = |c: Option<char>| c.is_some_and(char::is_alphabetic);
    let is_n = |c: Option<char>| c.is_some_and(char::is_numeric);
    let is_s = |c: Option<char>| c.is_some_and(char::is_whitespace);
    let is_nl = |c: Option<char>| matches!(c, Some('\r' | '\n'));
    let mut out = Vec::new();
    let mut i = 0;
    while i < cs.len() {
        let c = at(i);
        let end = 'm: {
            // Contractions.
            if c == Some('\'') {
                let lower = |j| at(j).map(|c: char| c.to_ascii_lowercase());
                if matches!(lower(i + 1), Some('s' | 't' | 'm' | 'd')) {
                    break 'm i + 2;
                }
                let two = (lower(i + 1), lower(i + 2));
                if matches!(two, (Some('r' | 'v'), Some('e')) | (Some('l'), Some('l'))) {
                    break 'm i + 3;
                }
            }
            // An optional non-letter-digit-newline, then letters.
            let start = if is_l(c) {
                Some(i)
            } else if !is_nl(c) && !is_n(c) && is_l(at(i + 1)) {
                Some(i + 1)
            } else {
                None
            };
            if let Some(mut j) = start {
                while is_l(at(j)) {
                    j += 1;
                }
                break 'm j;
            }
            if is_n(c) {
                break 'm i + 1;
            }
            // An optional space, punctuation, then newlines.
            let p = if c == Some(' ') { i + 1 } else { i };
            let punct = |c: Option<char>| c.is_some() && !is_s(c) && !is_l(c) && !is_n(c);
            if punct(at(p)) {
                let mut j = p;
                while punct(at(j)) {
                    j += 1;
                }
                while is_nl(at(j)) {
                    j += 1;
                }
                break 'm j;
            }
            // Whitespace: through its last newline; else all but the last
            // space before a non-space; else all of it.
            let mut j = i;
            while is_s(at(j)) {
                j += 1;
            }
            if let Some(last_nl) = (i..j).rev().find(|&x| is_nl(at(x))) {
                break 'm last_nl + 1;
            }
            if j < cs.len() && j - i > 1 {
                break 'm j - 1;
            }
            j.max(i + 1)
        };
        out.push(&text[byte(i)..byte(end)]);
        i = end;
    }
    out
}
