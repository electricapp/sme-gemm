#!/bin/sh
# Downloads PrismML's 1-bit Bonsai 1.7B GGUF (Apache-2.0, 248 MB) into data/.
set -eu
dir="$(dirname "$0")/data"
mkdir -p "$dir"
f="$dir/Bonsai-1.7B-Q1_0.gguf"
[ -f "$f" ] && exit 0
curl -fL --progress-bar -o "$f.part" \
    https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/main/Bonsai-1.7B-Q1_0.gguf
mv "$f.part" "$f"
