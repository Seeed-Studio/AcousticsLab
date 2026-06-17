#!/usr/bin/env bash
# `zig c++` cross wrapper; drops the cc-rs `--target=` zig rejects (see zig-cc-*).
set -euo pipefail
args=(); for a in "$@"; do [[ $a == --target=* ]] || args+=("$a"); done
exec zig c++ -target aarch64-linux-gnu "${args[@]}"
