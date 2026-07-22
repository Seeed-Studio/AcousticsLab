#!/bin/sh

set -e

case "${1:-}" in
    configure) [ -z "${2:-}" ] && FIRST_INSTALL=1 || FIRST_INSTALL=0 ;;
    1) FIRST_INSTALL=1 ;;
    2) FIRST_INSTALL=0 ;;
    *) FIRST_INSTALL=1 ;;
esac

NOLOGIN=/usr/sbin/nologin
[ -x "$NOLOGIN" ] || NOLOGIN=/sbin/nologin

getent group acousticslab >/dev/null 2>&1 || groupadd --system acousticslab
getent passwd acousticslab >/dev/null 2>&1 || useradd --system \
    --gid acousticslab --home-dir /var/lib/acousticslab --no-create-home \
    --shell "$NOLOGIN" --comment "AcousticsLab daemon" acousticslab

if getent group audio >/dev/null 2>&1; then
    usermod --append --groups audio acousticslab >/dev/null 2>&1 || true
fi

install -d -o acousticslab -g acousticslab -m 0750 /var/lib/acousticslab

if [ -d /run/systemd/system ]; then
    systemctl daemon-reload || true
    if [ "$FIRST_INSTALL" = 1 ]; then
        systemctl enable --now acousticslabd.service acousticslab-web.service || true
    else
        systemctl try-restart acousticslabd.service acousticslab-web.service || true
    fi
fi

exit 0
