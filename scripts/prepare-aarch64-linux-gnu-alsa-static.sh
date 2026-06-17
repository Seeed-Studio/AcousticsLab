#!/usr/bin/env bash
# Cross-build a STATIC libasound.a for aarch64-linux-gnu so acousticsd links
# ALSA statically (no DT_NEEDED libasound.so.2, no symbol-version skew with the
# device). Downloads the pinned alsa-lib source (sha256-checked, cached), then
# configures + builds the library with `zig cc` + zig's LLVM ar/ranlib.
#
# Runtime note: static linking removes the libasound.so.2 dependency but NOT
# ALSA's runtime data deps -- the device still needs /usr/share/alsa/*.conf, and
# hw_spec must stay on hw:/plughw: (external dlopen plugins like pulse/jack carry
# their own libasound.so.2 NEEDED, which would load a second libasound).
set -euo pipefail

ALSA_LIB_VER="1.2.16"
ALSA_LIB_SHA256="122b1e3166d55fe19bcde656535d7a36f2ab10e66c72c6ad2f43f20ffded0a96"
ALSA_LIB_URL="https://www.alsa-project.org/files/pub/lib/alsa-lib-${ALSA_LIB_VER}.tar.bz2"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STUB_DIR="${REPO_ROOT}/target/cross-stub/aarch64-linux-gnu"
CACHE_DIR="${REPO_ROOT}/target/cross-stub/cache"

# Idempotent: stamped by alsa-lib version (a version bump re-triggers).
STAMP="${STUB_DIR}/.stamp-alsa-static-${ALSA_LIB_VER}"
if [[ -f "${STAMP}" && -e "${STUB_DIR}/libasound.a" ]]; then
    echo "static libasound.a already prepared for alsa-lib ${ALSA_LIB_VER} at ${STUB_DIR}"
    exit 0
fi

command -v zig >/dev/null 2>&1 || { echo "zig not on PATH (needed for the cross build); install zig 0.16+" >&2; exit 1; }
command -v curl >/dev/null 2>&1 || { echo "curl not on PATH (needed to fetch alsa-lib source)" >&2; exit 1; }

sha256_of() {
    if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
    elif command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
    else echo "neither shasum nor sha256sum found" >&2; exit 1; fi
}

mkdir -p "${CACHE_DIR}" "${STUB_DIR}"
TARBALL="${CACHE_DIR}/alsa-lib-${ALSA_LIB_VER}.tar.bz2"

# Download once (cached); re-fetch if the cached copy fails the checksum.
if [[ -f "${TARBALL}" && "$(sha256_of "${TARBALL}")" == "${ALSA_LIB_SHA256}" ]]; then
    echo "using cached ${TARBALL}"
else
    echo "fetching ${ALSA_LIB_URL}..."
    curl -fsSL --max-time 180 -o "${TARBALL}" "${ALSA_LIB_URL}"
    got="$(sha256_of "${TARBALL}")"
    [[ "${got}" == "${ALSA_LIB_SHA256}" ]] || {
        echo "sha256 mismatch for alsa-lib-${ALSA_LIB_VER}.tar.bz2" >&2
        echo "  expected ${ALSA_LIB_SHA256}" >&2
        echo "  got      ${got}" >&2
        rm -f "${TARBALL}"; exit 1
    }
fi

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
tar xjf "${TARBALL}" -C "${WORK}"
SRC="${WORK}/alsa-lib-${ALSA_LIB_VER}"
[[ -x "${SRC}/configure" ]] || { echo "alsa-lib ${ALSA_LIB_VER} tarball has no configure script" >&2; exit 1; }

# Cross config: zig as the aarch64 cc, zig's LLVM ar/ranlib (Apple's ar makes an
# empty ELF archive). --without-versioned -> plain unversioned symbols in the
# .a (no GNU version nodes -> no version-skew). --disable-shared so only the .a
# is produced; built-in PCM plugins (hw/plug/route/rate) cover hw:/plughw:.
BUILD_TRIPLE="$(sh "${SRC}/config.guess")"
(
    cd "${SRC}"
    # Explicit, deterministic flags (do NOT inherit the host's CPPFLAGS/LDFLAGS,
    # which a Homebrew-LLVM profile points at macOS include/lib dirs -- those
    # would shadow zig's aarch64 sysroot in this cross build). -ffunction/-fdata
    # -sections let the daemon's final --gc-sections drop unused libasound code
    # at function granularity (not just whole objects); -fPIC for the -pie link.
    ./configure \
        --host=aarch64-linux-gnu \
        --build="${BUILD_TRIPLE}" \
        CC="zig cc -target aarch64-linux-gnu" \
        CC_FOR_BUILD=cc \
        AR="${REPO_ROOT}/scripts/zig-ar.sh" \
        RANLIB="${REPO_ROOT}/scripts/zig-ranlib.sh" \
        CFLAGS="-O2 -fPIC -ffunction-sections -fdata-sections" \
        CPPFLAGS="" \
        LDFLAGS="" \
        --enable-static --disable-shared \
        --without-versioned --disable-python --without-debug \
        --disable-topology --disable-old-symbols >/dev/null
    make -C src -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)" >/dev/null
)

ARCHIVE="${SRC}/src/.libs/libasound.a"
[[ -s "${ARCHIVE}" ]] || { echo "build produced no libasound.a at ${ARCHIVE}" >&2; exit 1; }

# A trailing-byte-only (empty) archive means the host ar mis-handled the ELF
# objects; guard against silently shipping a no-op ALSA.
[[ "$(wc -c <"${ARCHIVE}")" -gt 100000 ]] || { echo "libasound.a is suspiciously small ($(wc -c <"${ARCHIVE}") bytes) -- archiver likely failed" >&2; exit 1; }

# A stale dynamic stub here would let the linker pick .so over .a; remove it.
rm -f "${STUB_DIR}/libasound.so" "${STUB_DIR}/libasound.so.2"
cp "${ARCHIVE}" "${STUB_DIR}/libasound.a"
touch "${STAMP}"
echo "static libasound.a ready: ${STUB_DIR}/libasound.a ($(wc -c <"${STUB_DIR}/libasound.a") bytes, alsa-lib ${ALSA_LIB_VER})"
