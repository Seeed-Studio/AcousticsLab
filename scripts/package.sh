#!/usr/bin/env bash

. "$(dirname "$0")/_pkg-common.sh"

FEATURES="${FEATURES:-alsa-real,rknpu}"
setup_cross_env

echo ">> building binaries (arch=$PKG_ARCH target=$RUST_TARGET profile=$PROFILE version=$PKG_VERSION)"
cargo build --profile "$PROFILE" --target "$RUST_TARGET" --features "$FEATURES" --bin acousticslabd
cargo build --profile "$PROFILE" --target "$RUST_TARGET" -p acousticslab-webd --bin acousticslab-webd
cargo build --profile "$PROFILE" --target "$RUST_TARGET" -p acousticslab-cli --bin acousticslab

DAEMON_BIN="${BIN_DIR}/acousticslabd"
WEBD_BIN="${BIN_DIR}/acousticslab-webd"
CLI_BIN="${BIN_DIR}/acousticslab"
for b in "$DAEMON_BIN" "$WEBD_BIN" "$CLI_BIN"; do
    [ -x "$b" ] || {
        echo "built binary missing: $b" >&2
        exit 1
    }
done

LIBC_VERSION="$(detect_libc_floor "$DAEMON_BIN" "$WEBD_BIN" "$CLI_BIN")"
echo ">> glibc floor: >= ${LIBC_VERSION}"

echo ">> building web SPA (VITE_BASE_PATH='${VITE_BASE_PATH:-}')"
(
    cd web
    if [ -f package-lock.json ]; then npm ci; else npm install; fi
    npm run build
)
[ -f web/build/index.html ] || {
    echo "web build missing: web/build/index.html" >&2
    exit 1
}

require_nfpm
mkdir -p "$OUT_DIR"
export DAEMON_BIN WEBD_BIN CLI_BIN LIBC_VERSION
render_nfpm packaging/nfpm.yaml "$OUT_DIR/nfpm.gen.yaml"
for fmt in $FORMATS; do
    echo ">> packaging $fmt -> $OUT_DIR"
    nfpm package -f "$OUT_DIR/nfpm.gen.yaml" -p "$fmt" -t "$OUT_DIR"
done

echo ">> done:"
ls -1sh "$OUT_DIR"/acousticslab[-_]*
