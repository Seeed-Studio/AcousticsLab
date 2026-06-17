#!/usr/bin/env bash
# `zig ar` (LLVM ar) wrapper for AR= in the alsa-lib cross build. Apple's ar
# writes an empty (mach-o) archive from aarch64 ELF objects; zig's LLVM ar
# indexes them correctly. Companion: zig-ranlib-*.sh.
set -euo pipefail
exec zig ar "$@"
