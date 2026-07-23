#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROFILE="${PROFILE:-release}"
OUT_DIR="${OUT_DIR:-target/packages}"
FORMATS="${FORMATS:-deb rpm}"

case "$PROFILE" in
    release) PROFILE_DIR=release ;;
    dev) PROFILE_DIR=debug ;;
    *) PROFILE_DIR="$PROFILE" ;;
esac

host_arch() {
    if command -v dpkg >/dev/null 2>&1; then
        dpkg --print-architecture
    else
        case "$(uname -m)" in
            aarch64 | arm64) echo arm64 ;;
            x86_64 | amd64) echo amd64 ;;
            *)
                echo "cannot map $(uname -m) to a package arch; set ARCH" >&2
                exit 1
                ;;
        esac
    fi
}

ARCH="${ARCH:-$(host_arch)}"
case "$ARCH" in
    arm64) RUST_TARGET=aarch64-unknown-linux-gnu GNU_ARCH=aarch64-linux-gnu ;;
    amd64) RUST_TARGET=x86_64-unknown-linux-gnu GNU_ARCH=x86_64-linux-gnu ;;
    *)
        echo "unsupported ARCH '$ARCH' (expected arm64 or amd64)" >&2
        exit 1
        ;;
esac
PKG_ARCH="$ARCH"
BIN_DIR="target/${RUST_TARGET}/${PROFILE_DIR}"

setup_cross_env() {
    [ "$ARCH" != "$(host_arch)" ] || return 0
    local upper
    upper="$(echo "$RUST_TARGET" | tr 'a-z-' 'A-Z_')"
    export "CARGO_TARGET_${upper}_LINKER=${GNU_ARCH}-gcc"
    export "CC_${RUST_TARGET//-/_}=${GNU_ARCH}-gcc"
    export PKG_CONFIG_ALLOW_CROSS=1
    export PKG_CONFIG_LIBDIR="/usr/lib/${GNU_ARCH}/pkgconfig"
    command -v "${GNU_ARCH}-gcc" >/dev/null 2>&1 || {
        echo "cross build needs ${GNU_ARCH}-gcc (apt install gcc-${GNU_ARCH//_/-})" >&2
        exit 1
    }
    [ -e "${PKG_CONFIG_LIBDIR}/alsa.pc" ] || {
        echo "cross build needs libasound2-dev:${ARCH}" >&2
        echo "  dpkg --add-architecture ${ARCH} && apt update && apt install libasound2-dev:${ARCH}" >&2
        exit 1
    }
    echo ">> cross-compiling for ${RUST_TARGET} via ${GNU_ARCH}-gcc"
}

detect_libc_floor() {
    local od=objdump
    command -v "${GNU_ARCH}-objdump" >/dev/null 2>&1 && od="${GNU_ARCH}-objdump"
    command -v "$od" >/dev/null 2>&1 || {
        echo "need $od (binutils) to derive the glibc floor" >&2
        exit 1
    }
    local v
    v="$("$od" -T "$@" 2>/dev/null |
        grep -oE 'GLIBC_[0-9]+\.[0-9]+(\.[0-9]+)?' | sed 's/GLIBC_//' |
        sort -uV | tail -1)"
    [ -n "$v" ] || {
        echo "could not read glibc symbol versions from: $*" >&2
        exit 1
    }
    echo "$v"
}

PKG_VERSION="${PKG_VERSION:-$(awk '
    /^\[workspace\.package\]/ { p = 1; next }
    /^\[/ { p = 0 }
    p && /^version[[:space:]]*=/ {
        gsub(/["[:space:]]/, ""); sub(/^version=/, ""); print; exit
    }' Cargo.toml)}"
[ -n "$PKG_VERSION" ] || {
    echo "cannot parse version from Cargo.toml" >&2
    exit 1
}

require_nfpm() {
    command -v nfpm >/dev/null 2>&1 || {
        echo "nfpm not found on PATH. Install it (https://nfpm.goreleaser.com/install/)," >&2
        echo "e.g. 'brew install nfpm', or fetch a release binary from GitHub." >&2
        exit 1
    }
}

render_nfpm() {
    sed \
        -e "s#\${PKG_ARCH}#${PKG_ARCH}#g" \
        -e "s#\${PKG_VERSION}#${PKG_VERSION}#g" \
        -e "s#\${LIBC_VERSION}#${LIBC_VERSION:-}#g" \
        -e "s#\${DAEMON_BIN}#${DAEMON_BIN:-}#g" \
        -e "s#\${WEBD_BIN}#${WEBD_BIN:-}#g" \
        -e "s#\${CLI_BIN}#${CLI_BIN:-}#g" \
        "$1" >"$2"
}

export REPO_ROOT ARCH RUST_TARGET GNU_ARCH PKG_ARCH PKG_VERSION
export PROFILE PROFILE_DIR BIN_DIR OUT_DIR FORMATS
