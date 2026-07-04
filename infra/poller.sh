#!/bin/sh
# infra/poller.sh — continuous SL + weather data collection
#
# Daytime (05:00-00:00 Stockholm): POLL_INTERVAL_SECONDS (default 60s)
# Night   (00:00-05:00 Stockholm): NIGHT_INTERVAL_SECONDS (default 900s)
# Supervisord keeps this alive if it crashes.

POLL_INTERVAL="${POLL_INTERVAL_SECONDS:-60}"
NIGHT_INTERVAL="${NIGHT_INTERVAL_SECONDS:-1800}"

echo "[poller] starting — day=${POLL_INTERVAL}s  night=${NIGHT_INTERVAL}s"

while true; do
    # Stockholm is UTC+1 (winter) / UTC+2 (summer); use UTC hour + 2 as safe default
    HOUR_SE=$(TZ="Europe/Stockholm" date +%H)
    if [ "$HOUR_SE" -ge 0 ] && [ "$HOUR_SE" -lt 5 ]; then
        INTERVAL=$NIGHT_INTERVAL
    else
        INTERVAL=$POLL_INTERVAL
    fi

    echo "[poller] running pipeline at $(date -u +%Y-%m-%dT%H:%M:%SZ) (interval=${INTERVAL}s)"
    python -m transit.pipeline || echo "[poller] pipeline exited non-zero — will retry next cycle"
    sleep "$INTERVAL"
done
