#!/bin/sh

set -e

if [ -d /run/systemd/system ]; then
    systemctl daemon-reload || true
fi

if [ "${1:-}" = purge ]; then
    rm -rf /var/lib/acousticslab
    if getent passwd acousticslab >/dev/null 2>&1; then
        userdel acousticslab >/dev/null 2>&1 || true
    fi
    if getent group acousticslab >/dev/null 2>&1; then
        groupdel acousticslab >/dev/null 2>&1 || true
    fi
fi

exit 0
