#!/usr/bin/env bash
# Cross-build acousticslabd for aarch64-unknown-linux-gnu via cargo-zigbuild on a
# macOS host: zig cc/c++ wrappers + a locally cross-built static libasound.a (no
# sysroot, no runtime libasound.so.2) + opus statically bundled by opusic-sys,
# which CMake-builds its vendored libopus and never probes pkg-config -- so opus
# needs no setup here. Forwards args to cargo zigbuild (default: embedded release
# with alsa-real,rknpu).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STUB_DIR="${REPO_ROOT}/target/cross-stub/aarch64-linux-gnu"

[[ -e "${STUB_DIR}/libasound.a" ]] || {
    echo "static libasound.a not present; preparing..."
    "${REPO_ROOT}/scripts/prepare-aarch64-linux-gnu-alsa-static.sh"
}

# cc-rs -> zig wrappers (translate the Rust target triple zig rejects).
export CC_aarch64_unknown_linux_gnu="${REPO_ROOT}/scripts/zig-cc-aarch64-linux-gnu.sh"
export CXX_aarch64_unknown_linux_gnu="${REPO_ROOT}/scripts/zig-cxx-aarch64-linux-gnu.sh"

# Override alsa-sys's `links = "alsa"` for this target: force a STATIC link
# against the local libasound.a (alsa-sys hardcodes a dynamic pkg-config probe;
# rustc won't fall back from -lasound.so to .a). This also skips alsa-sys's
# build script entirely, so no pkg-config is needed. dl/m/pthread satisfy
# libasound's transitive symbols (plugin dlopen / math / threads).
ALSA_LINK_OVERRIDE=(
    --config "target.aarch64-unknown-linux-gnu.alsa.rustc-link-lib=[\"static=asound\",\"dl\",\"m\",\"pthread\"]"
    --config "target.aarch64-unknown-linux-gnu.alsa.rustc-link-search=[\"native=${STUB_DIR}\"]"
)

[[ $# -eq 0 ]] && set -- --profile release-embedded --features alsa-real,rknpu --bin acousticslabd
exec cargo zigbuild --target aarch64-unknown-linux-gnu "${ALSA_LINK_OVERRIDE[@]}" "$@"
