"""Minimal Slack webhook alert sender.

If SLACK_WEBHOOK is not set, messages are printed to stdout (captured by Azure logs).
"""
from __future__ import annotations

import json
import os
import urllib.request


def send_slack_alert(message: str) -> bool:
    """Send a plain-text message to Slack via webhook.

    Returns True if sent, False if no webhook configured or request failed.
    Never raises — alert failure must not crash the main job.
    """
    webhook_url = os.getenv("SLACK_WEBHOOK")

    if not webhook_url:
        print(f"[slack_alert] No SLACK_WEBHOOK configured. Message:\n{message}")
        return False

    payload = json.dumps({"text": message}).encode("utf-8")

    try:
        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status != 200:
                print(f"[slack_alert] Unexpected response: {resp.status}")
                return False
        return True

    except Exception as exc:
        print(f"[slack_alert] Failed to send alert: {exc}")
        return False
