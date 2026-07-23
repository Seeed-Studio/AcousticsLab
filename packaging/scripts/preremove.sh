#!/bin/sh

set -e

case "${1:-}" in
    remove | 0) FINAL=1 ;;
    *) FINAL=0 ;;
esac

if [ -d /run/systemd/system ] && [ "$FINAL" = 1 ]; then
    systemctl disable --now acousticslab-webd.service acousticslabd.service || true
fi

exit 0
