#!/bin/sh
# Download the nanoGPT shakespeare-char checkpoint and text into data/, then
# convert the checkpoint to raw f32 (weights.f32 + index.txt) without torch.
set -e
cd "$(dirname "$0")"
mkdir -p data
hf=https://huggingface.co/n8cha/nanoGPT-shakespeare-char/resolve/main
for f in pytorch_model.bin config.json model.py; do curl -sL -o "data/$f" "$hf/$f"; done
curl -sL -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
(cd data && python3 ../convert.py)
