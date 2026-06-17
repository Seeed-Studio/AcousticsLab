#!/usr/bin/env bash
# `zig cc` cross wrapper for CC_aarch64_unknown_linux_gnu. cc-rs 1.2 appends
# `--target=aarch64-unknown-linux-gnu`, which zig 0.16 rejects; drop it and use
# zig's `aarch64-linux-gnu`. Companion: zig-cxx-aarch64-linux-gnu.sh.
set -euo pipefail
args=(); for a in "$@"; do [[ $a == --target=* ]] || args+=("$a"); done
exec zig cc -target aarch64-linux-gnu "${args[@]}"
