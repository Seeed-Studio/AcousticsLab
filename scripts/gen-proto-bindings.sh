#!/usr/bin/env bash
# Generate C++ (--cpp, default) or C / protobuf-c (--c) bindings for the
# `acoustics` wire-format protos in modules/proto/, letting non-Rust processes
# decode the [output.inference] UDS stream. Framing contract: docs/PROTO.md.
#
# Usage: scripts/gen-proto-bindings.sh [--cpp] [--c] [-o OUT_DIR]
# Requires protoc (brew install protobuf / apt install protobuf-compiler).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROTO_DIR="${REPO_ROOT}/modules/proto"
# Order mirrors build.rs: imported leaves before the envelope wrapper.
PROTOS=(audio_stream.proto inference_stream.proto envelope.proto)
OUT_DIR="${REPO_ROOT}/target/proto-bindings"
GEN_CPP=0
GEN_C=0

usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next} {exit}' "$0"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpp) GEN_CPP=1 ;;
        --c) GEN_C=1 ;;
        -o|--out) shift; [[ $# -gt 0 ]] || { echo "error: -o needs a directory" >&2; exit 2; }; OUT_DIR="$1" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done
[[ $GEN_CPP -eq 0 && $GEN_C -eq 0 ]] && GEN_CPP=1

command -v protoc >/dev/null 2>&1 || {
    echo "error: protoc not found (brew install protobuf / apt install protobuf-compiler)" >&2
    exit 1
}

# Absolute OUT_DIR: run_protoc cd's into PROTO_DIR, so a relative --*_out would
# resolve against the wrong directory.
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

# Run protoc from PROTO_DIR so the bare `import "x.proto"` lines resolve via -I .
run_protoc() { ( cd "$PROTO_DIR" && protoc -I . "$@" "${PROTOS[@]}" ); }

echo "protoc: $(protoc --version)  out: ${OUT_DIR}"

if [[ $GEN_CPP -eq 1 ]]; then
    mkdir -p "${OUT_DIR}/cpp"
    run_protoc --cpp_out="${OUT_DIR}/cpp"
    echo "  cpp -> ${OUT_DIR}/cpp (*.pb.h, *.pb.cc)"
fi

if [[ $GEN_C -eq 1 ]]; then
    command -v protoc-gen-c >/dev/null 2>&1 || {
        echo "error: protoc-gen-c not found (brew install protobuf-c / apt install protobuf-c-compiler)" >&2
        exit 1
    }
    mkdir -p "${OUT_DIR}/c"
    # NOTE: stock protoc-gen-c cannot generate these schemas -- they use proto3
    # `optional`, unsupported by protobuf-c (#476/#783); protoc errors out here.
    run_protoc --c_out="${OUT_DIR}/c"
    echo "  c -> ${OUT_DIR}/c (*.pb-c.h, *.pb-c.c)"
fi

echo "done."
