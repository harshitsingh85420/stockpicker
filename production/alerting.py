"""
alerting.py
===========
P50 — Telegram alerts + kill-switch activation.

SETUP:
    Set in .env:
        TELEGRAM_BOT_TOKEN=<your bot token>
        TELEGRAM_CHAT_ID=<your chat id>

    Get bot token: https://t.me/BotFather
    Get chat id: message your bot, then GET /getUpdates

KILL SWITCH:
    Create the file  stock_picker_data/KILL_SWITCH  (any content) to halt
    all new orders.  Delete the file to resume.  The alerting module will
    also create this file when it receives a CRITICAL alert.

Usage:
    from production.alerting import send_alert, AlertLevel
    send_alert("Circuit breaker HALTED", level=AlertLevel.CRITICAL)
"""

from __future__ import annotations

import enum
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_KILL_SWITCH_PATH = _ROOT / "stock_picker_data" / "KILL_SWITCH"


class AlertLevel(str, enum.Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Low-level Telegram send
# ---------------------------------------------------------------------------

def _telegram_send(message: str) -> bool:
    """
    Send message via Telegram Bot API.
    Returns True on success, False on any failure (never raises).
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    token   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID",   "")

    if not token or not chat_id:
        logger.debug("P50: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set — alert skipped.")
        return False

    try:
        import urllib.request, urllib.parse, json
        url     = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = json.dumps({"chat_id": chat_id, "text": message, "parse_mode": "HTML"}).encode()
        req     = urllib.request.Request(url, data=payload,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.load(resp)
            return bool(body.get("ok", False))
    except Exception as exc:
        logger.warning("P50: Telegram send failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_alert(
    message: str,
    level: AlertLevel | str = AlertLevel.INFO,
    activate_kill_switch_on_critical: bool = True,
) -> bool:
    """
    Send an alert via Telegram and optionally activate the kill switch.

    Parameters
    ----------
    message   : Human-readable alert text
    level     : AlertLevel.INFO | WARNING | CRITICAL (or string)
    activate_kill_switch_on_critical:
        When True (default), CRITICAL alerts also write the kill-switch file.

    Returns True if Telegram delivery succeeded, False otherwise.
    """
    level_str = level.value if isinstance(level, AlertLevel) else str(level).upper()
    emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(level_str, "📌")

    full_message = f"{emoji} <b>[STOCKPICKER {level_str}]</b>\n{message}"
    logger.log(
        logging.CRITICAL if level_str == "CRITICAL" else
        logging.WARNING  if level_str == "WARNING"  else logging.INFO,
        "P50 Alert [%s]: %s", level_str, message,
    )

    if level_str == "CRITICAL" and activate_kill_switch_on_critical:
        _activate_kill_switch(reason=message)

    return _telegram_send(full_message)


# ---------------------------------------------------------------------------
# Kill-switch helpers
# ---------------------------------------------------------------------------

def _activate_kill_switch(reason: str = "CRITICAL alert") -> None:
    """Write the kill-switch sentinel file."""
    try:
        _KILL_SWITCH_PATH.parent.mkdir(parents=True, exist_ok=True)
        _KILL_SWITCH_PATH.write_text(f"KILL_SWITCH activated: {reason}\n")
        logger.critical("P50: KILL_SWITCH file written -> %s", _KILL_SWITCH_PATH)
    except Exception as exc:
        logger.error("P50: Could not write kill-switch file: %s", exc)


def kill_switch_active() -> bool:
    """Return True if the kill-switch file exists."""
    return _KILL_SWITCH_PATH.exists()


def deactivate_kill_switch() -> None:
    """Remove the kill-switch file to resume trading."""
    if _KILL_SWITCH_PATH.exists():
        _KILL_SWITCH_PATH.unlink()
        logger.info("P50: Kill switch deactivated.")
    else:
        logger.info("P50: Kill switch was not active.")


# ---------------------------------------------------------------------------
# Convenience senders for the 5 required events
# ---------------------------------------------------------------------------

def alert_circuit_breaker_halted(drawdown_pct: float) -> None:
    send_alert(
        f"Circuit breaker HALTED — drawdown {drawdown_pct:.1%}. "
        "No new trades until manually reviewed.",
        level=AlertLevel.CRITICAL,
    )


def alert_circuit_breaker_warning(drawdown_pct: float) -> None:
    send_alert(
        f"Circuit breaker WARNING — drawdown {drawdown_pct:.1%}. "
        "Positions halved.",
        level=AlertLevel.WARNING,
    )


def alert_kill_switch_activated(reason: str) -> None:
    send_alert(
        f"KILL SWITCH activated: {reason}",
        level=AlertLevel.CRITICAL,
        activate_kill_switch_on_critical=False,   # already fired externally
    )


def alert_picks_ready(n_picks: int, ref_date: str, csv_path: Optional[str] = None) -> None:
    msg = f"Daily picks ready — {n_picks} stocks selected for {ref_date}."
    if csv_path:
        msg += f"\nFile: {csv_path}"
    send_alert(msg, level=AlertLevel.INFO)


def alert_data_source_failed(source_name: str, fallback: Optional[str] = None) -> None:
    msg = f"Data source '{source_name}' failed."
    if fallback:
        msg += f" Using fallback: {fallback}."
    send_alert(msg, level=AlertLevel.WARNING)
