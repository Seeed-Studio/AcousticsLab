#!/usr/bin/env bash
# `zig ranlib` (LLVM ranlib) wrapper for RANLIB= in the alsa-lib cross build;
# indexes the aarch64 ELF archive Apple's ranlib can't. See zig-ar.sh.
set -euo pipefail
exec zig ranlib "$@"
