#!/bin/sh
# infra/poller.sh — continuous SL + weather data collection
#
# Runs transit.pipeline every POLL_INTERVAL_SECONDS (default 900 = 15 min).
# Supervisord keeps this alive if it crashes.

POLL_INTERVAL="${POLL_INTERVAL_SECONDS:-900}"

echo "[poller] starting — interval=${POLL_INTERVAL}s"

while true; do
    echo "[poller] running pipeline at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    python -m transit.pipeline || echo "[poller] pipeline exited non-zero — will retry next cycle"
    echo "[poller] sleeping ${POLL_INTERVAL}s"
    sleep "$POLL_INTERVAL"
done
