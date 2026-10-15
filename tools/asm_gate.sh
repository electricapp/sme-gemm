#!/bin/sh
# The codegen check for refactors of csrc/: compile every kernel file with
# build.rs's flags, normalize the disassembly per function, and diff two
# builds. A function whose instructions are unchanged needs no benchmark; the
# ones that differ are what to benchmark.
#
#   tools/asm_gate.sh build <src-dir> <out-dir>    # compile and normalize
#   tools/asm_gate.sh diff  <base-out> <cand-out>  # per-function report
#
# Typical use: copy csrc/ at the base commit somewhere, build both, diff. A .c
# file builds as C and a .cc file as C++ (CXXSTD, default c++2c: C++26), so a base
# from before the move to C++ still compares.
# CC / CXX default to Apple's clang, which is what build.rs's `cc` and `c++`
# resolve to; another clang earlier on PATH would compare the wrong compiler.
set -eu

CC=${CC:-/usr/bin/clang}
CXX=${CXX:-/usr/bin/clang++}
FLAGS="-mcpu=apple-m4 -ffp-contract=fast -O3 -Wall -Wextra -Werror -Wshadow -Wpointer-arith
  -Wcast-qual -Wwrite-strings -Wvla -Wdouble-promotion -Wredundant-decls -Wundef -Wformat=2
  -Wcast-align -Wunreachable-code-aggressive -Wconditional-uninitialized"
C_ONLY="-Wstrict-prototypes -Wold-style-definition -Wnested-externs -Wbad-function-cast"

feat() {
  case "$1" in
    gemm_f16f16*) echo "-Xclang -target-feature -Xclang +sme-f16f16" ;;
    gemm_b16b16*) echo "-Xclang -target-feature -Xclang +sme-b16b16 -Xclang -target-feature -Xclang +sve-b16b16" ;;
    gemm_f64*)    echo "-Xclang -target-feature -Xclang +sme-f64f64" ;;
    gemm_i16i64*) echo "-Xclang -target-feature -Xclang +sme-i16i64" ;;
    gemm_q2*)     echo "-Xclang -target-feature -Xclang +sme2p1" ;;
  esac
}

norm() {  # one object -> "FN <plain name>" headers + normalized instructions
  # Function names: C "_run_streaming" and C++ "__ZL13run_streamingPvll..." both
  # become "run_streaming"; block (_block_invoke_N) and outlining (.cold.N, .N)
  # suffixes are kept.
  objdump -d --no-show-raw-insn "$1" | awk '
    /^[0-9a-f]+ <.*>:$/ { n = $0; sub(/^[0-9a-f]+ </, "", n); sub(/>:$/, "", n);
                          sfx = ""; if (match(n, /(_block_invoke(_[0-9]+)?)?(\.[A-Za-z0-9_.]+)?$/)) sfx = substr(n, RSTART);
                          sub(/^_+/, "", n);
                          if (match(n, /^ZL?[0-9]+/)) { len = substr(n, 1, RLENGTH); sub(/^ZL?/, "", len);
                                                        n = substr(n, RLENGTH + 1, len + 0) sfx }
                          print "FN " n; next }
    /^[[:space:]]+[0-9a-f]+:/ { sub(/^[[:space:]]+[0-9a-f]+:[[:space:]]*/, "");
                                gsub(/<[^>]*>/, ""); gsub(/0x[0-9a-f]+/, "X");
                                gsub(/[[:space:]]+/, " "); print }'
}

case "$1" in
build)
  src=$2 out=$3; mkdir -p "$out"
  for f in "$src"/*.c "$src"/*.cc; do
    [ -e "$f" ] || continue
    b=$(basename "${f%.*}")
    case "$f" in
      *.cc) cc="$CXX -std=${CXXSTD:-c++2c} -fno-exceptions -fno-rtti"; warn="" ;;
      *) cc="$CC"; warn=$C_ONLY ;;
    esac
    # shellcheck disable=SC2046,SC2086
    $cc $FLAGS $warn $(feat "$b") -I"$src" -c "$f" -o "$out/$b.o"
    norm "$out/$b.o" > "$out/$b.s"
    nm -gU "$out/$b.o" | awk '{print $3}' | sort > "$out/$b.syms"
  done ;;
diff)
  base=$2 cand=$3; rc=0
  for s in "$base"/*.s; do
    b=$(basename "$s" .s)
    if [ ! -e "$cand/$b.s" ]; then echo "MISSING    $b"; rc=1; continue; fi
    if ! cmp -s "$base/$b.syms" "$cand/$b.syms"; then echo "SYMBOLS DIFFER: $b"; rc=1; fi
    if cmp -s "$s" "$cand/$b.s"; then echo "identical  $b"; continue; fi
    rc=1; echo "DIFFERS    $b"
    # per-function instruction counts, then the diff with function context
    t=$(mktemp -d)
    awk '/^FN /{f=$2; next} {c[f]++} END{for (f in c) print f, c[f]}' "$s" | sort > "$t/a"
    awk '/^FN /{f=$2; next} {c[f]++} END{for (f in c) print f, c[f]}' "$cand/$b.s" | sort > "$t/b"
    join -a1 -a2 -e0 -o 0,1.2,2.2 "$t/a" "$t/b" | awk '$2 != $3 {print "    " $1 ": " $2 " -> " $3 " insns"}'
    rm -rf "$t"
    diff "$s" "$cand/$b.s" | head -40 | sed 's/^/    /' || true
  done
  exit $rc ;;
*)
  echo "usage: $0 build <src-dir> <out-dir> | diff <base-out> <cand-out>" >&2
  exit 2 ;;
esac
