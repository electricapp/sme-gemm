//! `shakespeare`: continue text in the style of Shakespeare with nanoGPT
//! shakespeare-char running 4-bit on Apple's SME unit through sme-gemm.

mod model;
mod reference;

use std::io::{BufRead, IsTerminal, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

use model::{CTX, F32Weights, Model, Rng, Sampler, Session, decode, encode, sample};

const HELP: &str = "\
shakespeare -- continue text in the style of Shakespeare

USAGE:
    shakespeare [OPTIONS] [PROMPT...]

With no PROMPT it reads one from stdin when piped, or starts an interactive
prompt. The model knows 65 characters (letters, the digit 3, space, newline and
!$&',-.:;?); typographic quotes and dashes are mapped, anything else dropped.

OPTIONS:
    -n, --chars N         characters to generate [default: 500]
    -t, --temperature T   randomness; 0 always takes the likeliest [default: 0.8]
    -k, --top-k K         sample only among the K likeliest characters [default: all]
    -s, --seed S          random seed [default: from the clock]
        --q4-block B      quantization block 32, 64 or 128: larger is faster and
                          slightly less accurate [default: 32]
        --data DIR        checkpoint directory [default: $SHAKESPEARE_DATA, else
                          the data/ directory next to this crate]
    -q, --quiet           no timing line on stderr
        --bench           accuracy against f32 and throughput, then exit
    -h, --help            this text

EXAMPLES:
    shakespeare \"ROMEO:\"
    shakespeare -n 2000 -t 0.6 \"To be, or not to be\"
    echo \"KING HENRY:\" | shakespeare -q
";

/// Characters the window keeps when generation runs past the 256 context. The
/// model has absolute positions, so the kept tail is re-run as a prompt: half
/// the context costs ~2 ms every 128 characters (~16 us each).
const KEEP: usize = CTX / 2;

struct Opts {
    chars: usize,
    sampler: Sampler,
    q4_block: usize,
    data: PathBuf,
    quiet: bool,
    bench: bool,
    prompt: Option<String>,
}

fn parse() -> Result<Opts, String> {
    let mut o = Opts {
        chars: 500,
        sampler: Sampler {
            temperature: 0.8,
            top_k: 0,
            seed: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(1, |d| d.as_nanos() as u64),
        },
        q4_block: 32,
        data: std::env::var_os("SHAKESPEARE_DATA").map_or_else(
            || PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data"),
            PathBuf::from,
        ),
        quiet: false,
        bench: false,
        prompt: None,
    };
    let mut words = vec![];
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |name: &str| args.next().ok_or_else(|| format!("{name} needs a value"));
        match a.as_str() {
            "-h" | "--help" => {
                print!("{HELP}");
                std::process::exit(0);
            }
            "-n" | "--chars" => o.chars = val(&a)?.parse().map_err(|_| "--chars takes a count")?,
            "-t" | "--temperature" => {
                o.sampler.temperature = val(&a)?
                    .parse()
                    .map_err(|_| "--temperature takes a number")?;
            }
            "-k" | "--top-k" => {
                o.sampler.top_k = val(&a)?.parse().map_err(|_| "--top-k takes a count")?
            }
            "-s" | "--seed" => {
                o.sampler.seed = val(&a)?.parse().map_err(|_| "--seed takes an integer")?
            }
            "--q4-block" => {
                o.q4_block = val(&a)?
                    .parse()
                    .map_err(|_| "--q4-block takes 32, 64 or 128")?;
                if ![32, 64, 128].contains(&o.q4_block) {
                    return Err("--q4-block takes 32, 64 or 128".into());
                }
            }
            "--data" => o.data = PathBuf::from(val(&a)?),
            "-q" | "--quiet" => o.quiet = true,
            "--bench" => o.bench = true,
            "--" => words.extend(args.by_ref()),
            s if s.starts_with('-') && s.len() > 1 => {
                return Err(format!("unknown option {s} (try --help)"));
            }
            _ => words.push(a),
        }
    }
    if !words.is_empty() {
        o.prompt = Some(words.join(" "));
    }
    Ok(o)
}

/// The prompt as tokens: typographic quotes and dashes mapped onto the
/// vocabulary, anything else dropped (and counted).
fn tokenize(text: &str) -> (Vec<usize>, usize) {
    let mut dropped = 0;
    let toks = text
        .chars()
        .filter_map(|c| {
            let c = match c {
                '\t' => ' ',
                '\r' => return None,
                '"' | '\u{2018}' | '\u{2019}' | '\u{201c}' | '\u{201d}' | '`' => '\'',
                '\u{2013}' | '\u{2014}' => '-',
                _ => c,
            };
            let t = encode(c);
            dropped += usize::from(t.is_none());
            t
        })
        .collect();
    (toks, dropped)
}

struct Stats {
    prompt_chars: usize,
    prompt_s: f64,
    gen_s: f64,
}

/// Continues `prompt` by `n` characters. Past the 256-character context the
/// window restarts on its last KEEP characters as one batched prompt.
fn generate(
    sess: &mut Session<'_>,
    prompt: &[usize],
    n: usize,
    s: &Sampler,
    rng: &mut Rng,
) -> (String, Stats) {
    let mut hist: Vec<usize> = if prompt.is_empty() {
        vec![0]
    } else {
        prompt.to_vec()
    };
    sess.reset();
    let t0 = Instant::now();
    let start = hist.len().saturating_sub(CTX);
    sess.forward(&hist[start..]);
    let st = Stats {
        prompt_chars: hist.len() - start,
        prompt_s: t0.elapsed().as_secs_f64(),
        gen_s: 0.0,
    };
    let t1 = Instant::now();
    let mut out = String::with_capacity(n);
    for i in 0..n {
        let tok = sample(&sess.logits, s, rng);
        out.push(decode(tok));
        hist.push(tok);
        if i + 1 == n {
            break;
        }
        if sess.pos() == CTX {
            sess.reset();
            sess.forward(&hist[hist.len() - KEEP..]);
        } else {
            sess.forward(&[tok]);
        }
    }
    (
        out,
        Stats {
            gen_s: t1.elapsed().as_secs_f64(),
            ..st
        },
    )
}

fn run() -> Result<(), String> {
    let o = parse()?;
    let t_start = Instant::now();
    let w = F32Weights::load(&o.data).map_err(|e| {
        format!(
            "{e}\nfetch the checkpoint first: {}/fetch.sh",
            env!("CARGO_MANIFEST_DIR")
        )
    })?;
    let t_load = t_start.elapsed();
    let m = Model::quantize(&w, o.q4_block);
    if std::env::var_os("SHAKESPEARE_TIMING").is_some() {
        eprintln!(
            "[load {:.1} ms, quantize {:.1} ms]",
            t_load.as_secs_f64() * 1e3,
            (t_start.elapsed() - t_load).as_secs_f64() * 1e3
        );
    }
    if o.bench {
        let text = std::fs::read_to_string(o.data.join("input.txt"))
            .map_err(|e| format!("input.txt: {e}"))?;
        reference::bench(&w, &m, &text);
        return Ok(());
    }
    drop(w);
    let mut sess = Session::new(&m);
    let mut rng = Rng::new(o.sampler.seed);
    let mut continue_text = |text: &str| {
        let (toks, dropped) = tokenize(text);
        if dropped > 0 && !o.quiet {
            eprintln!("[dropped {dropped} characters the model does not know]");
        }
        let (cont, st) = generate(&mut sess, &toks, o.chars, &o.sampler, &mut rng);
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{text}{cont}");
        let _ = out.flush();
        if !o.quiet {
            eprintln!(
                "[prompt: {} chars in {:.2} ms | generated {} chars in {:.1} ms = {:.0} chars/s]",
                st.prompt_chars,
                st.prompt_s * 1e3,
                o.chars,
                st.gen_s * 1e3,
                o.chars as f64 / st.gen_s
            );
        }
    };
    let stdin = std::io::stdin();
    match &o.prompt {
        Some(p) => continue_text(p),
        None if !stdin.is_terminal() => {
            let mut text = String::new();
            stdin
                .lock()
                .read_to_string(&mut text)
                .map_err(|e| e.to_string())?;
            continue_text(text.trim_end_matches('\n'));
        }
        None => {
            eprintln!("Type a line for Shakespeare to continue (an empty line or Ctrl-D quits).");
            loop {
                eprint!("> ");
                let mut line = String::new();
                if stdin
                    .lock()
                    .read_line(&mut line)
                    .map_err(|e| e.to_string())?
                    == 0
                    || line.trim().is_empty()
                {
                    break;
                }
                continue_text(line.trim_end_matches('\n'));
                println!();
            }
        }
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("shakespeare: {e}");
        std::process::exit(2);
    }
}
