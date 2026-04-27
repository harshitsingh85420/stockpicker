"""
exit_all_positions.py
=====================
P50 — Emergency kill switch: activates the kill-switch file and (in live mode)
sends market SELL orders for all open positions via BrokerAdapter.

Usage:
    python exit_all_positions.py                  # paper mode — logs only
    python exit_all_positions.py --live           # LIVE: sends real orders
    python exit_all_positions.py --deactivate     # remove kill switch file only
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description="Emergency position exit + kill switch")
    parser.add_argument("--live",       action="store_true",
                        help="Send real SELL orders via BrokerAdapter (default: paper)")
    parser.add_argument("--deactivate", action="store_true",
                        help="Deactivate kill switch without placing orders")
    args = parser.parse_args()

    sys.path.insert(0, str(_ROOT))

    from production.alerting import (
        send_alert, AlertLevel,
        _activate_kill_switch, deactivate_kill_switch, kill_switch_active,
    )

    if args.deactivate:
        deactivate_kill_switch()
        send_alert("Kill switch deactivated manually.", level=AlertLevel.WARNING)
        logger.info("Kill switch deactivated.")
        return

    # 1. Activate kill switch
    reason = "Manual exit_all_positions.py triggered"
    _activate_kill_switch(reason)
    send_alert(f"KILL SWITCH: {reason}", level=AlertLevel.CRITICAL,
               activate_kill_switch_on_critical=False)

    if not kill_switch_active():
        logger.error("Kill switch file could not be written — aborting.")
        sys.exit(1)
    logger.info("Kill switch file written.")

    # 2. Fetch open positions
    paper_mode = not args.live
    try:
        from production.broker_adapter import BrokerAdapter
        broker = BrokerAdapter(paper_mode=paper_mode)
        positions = broker.get_positions()
    except Exception as exc:
        logger.error("Could not connect to broker: %s", exc)
        positions = []

    if not positions:
        logger.info("No open positions found (or paper mode with empty list).")
        send_alert("Exit script: no open positions found.", level=AlertLevel.INFO)
        return

    # 3. Send SELL orders
    logger.info("Sending MARKET SELL for %d positions (live=%s).", len(positions), args.live)
    results = []
    for pos in positions:
        sc_code  = str(pos.get("tradingsymbol", pos.get("SC_CODE", "UNKNOWN")))
        qty      = int(pos.get("quantity", pos.get("qty", 0)))
        exchange = str(pos.get("exchange", "BSE"))
        if qty <= 0:
            continue
        res = broker.place_order(
            symbol=sc_code, qty=qty, side="SELL",
            order_type="MARKET", price=None, exchange=exchange,
        )
        logger.info("SELL %s x%d -> %s", sc_code, qty, res)
        results.append(res)

    n_ok  = sum(1 for r in results if "error" not in r)
    n_err = len(results) - n_ok
    send_alert(
        f"Exit complete: {n_ok} orders sent, {n_err} errors. "
        f"Live={args.live}.",
        level=AlertLevel.CRITICAL if n_err else AlertLevel.WARNING,
        activate_kill_switch_on_critical=False,
    )
    logger.info("Done. %d succeeded, %d failed.", n_ok, n_err)


if __name__ == "__main__":
    main()
