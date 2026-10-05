#!/bin/sh
# One-time setup for examples/shakespeare: downloads the nanoGPT
# shakespeare-char checkpoint, converts it to model.safetensors (no torch
# needed), and fetches the text --bench scores against, all into data/ next to
# this script. Re-running skips whatever is already there.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
crate=$(cd "$here/../.." && pwd)
data=$here/data
for tool in curl python3; do
    command -v "$tool" >/dev/null || { echo "fetch.sh: $tool is required" >&2; exit 1; }
done
mkdir -p "$data"

if [ -s "$data/model.safetensors" ]; then
    echo "checkpoint: already in $data"
else
    echo "checkpoint: downloading (~43 MB)"
    curl -fL --progress-bar -o "$data/pytorch_model.bin" \
        https://huggingface.co/n8cha/nanoGPT-shakespeare-char/resolve/main/pytorch_model.bin
    (cd "$data" && python3 "$here/convert.py")
    rm "$data/pytorch_model.bin"
fi

if [ -s "$data/input.txt" ]; then
    echo "text: already in $data"
else
    echo "text: downloading (~1 MB)"
    curl -fL --progress-bar -o "$data/input.txt.part" \
        https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    mv "$data/input.txt.part" "$data/input.txt"
fi

cat <<EOF

Ready. Run it from the crate:
    cd $crate && cargo run --release --example shakespeare -- "ROMEO:"
or install it as a command:
    cargo install --path $crate --example shakespeare
    shakespeare "ROMEO:"
EOF
