"""
broker_adapter.py
=================
P48 — Zerodha Kite Connect broker adapter.

CRITICAL SAFETY RULES:
- paper_mode=True is the DEFAULT.  Real orders only when explicitly set False.
- KITE_API_KEY and KITE_ACCESS_TOKEN must be in .env (never hardcoded).
- ALGO_ID tag is mandatory for SEBI algorithmic trading compliance.
- Access token expires daily — refresh via the 2FA flow before each run.

Usage:
    from production.broker_adapter import BrokerAdapter
    broker = BrokerAdapter(paper_mode=True)
    order = broker.place_order("532540", qty=10, side="BUY", price=3678.0)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_env():
    """Load .env file if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass


class BrokerAdapter:
    """
    P48: Paper-mode + live Zerodha Kite Connect broker adapter.

    Parameters
    ----------
    paper_mode : bool
        When True (default), all orders are logged only — no real API calls.
    """

    def __init__(self, paper_mode: bool = True):
        _load_env()
        self.paper_mode = paper_mode
        self._kite = None

        if not paper_mode:
            self._init_live()
        else:
            logger.info("P48 BrokerAdapter: PAPER MODE — no real orders will be placed.")

    # ------------------------------------------------------------------
    def _init_live(self):
        """Initialise Kite Connect for live trading."""
        try:
            from kiteconnect import KiteConnect
        except ImportError:
            raise ImportError(
                "P48: kiteconnect not installed. Run: pip install kiteconnect"
            )
        api_key = os.getenv("KITE_API_KEY")
        access_token = os.getenv("KITE_ACCESS_TOKEN")
        if not api_key or not access_token:
            raise EnvironmentError(
                "P48: KITE_API_KEY and KITE_ACCESS_TOKEN must be set in .env"
            )
        self._kite = KiteConnect(api_key=api_key)
        self._kite.set_access_token(access_token)
        logger.info("P48: Kite Connect initialised (LIVE MODE).")

    # ------------------------------------------------------------------
    def place_order(
        self,
        symbol: str,
        qty: int,
        side: str = "BUY",
        order_type: str = "LIMIT",
        price: float = None,
        exchange: str = "BSE",
    ) -> Dict[str, Any]:
        """
        Place a single equity delivery order.

        Parameters
        ----------
        symbol   : BSE/NSE trading symbol or SC_CODE
        qty      : Number of shares
        side     : 'BUY' or 'SELL'
        order_type: 'LIMIT', 'MARKET', 'SL'
        price    : Limit price (required for LIMIT orders)
        exchange : 'BSE' or 'NSE'
        """
        algo_id = os.getenv("ALGO_ID", "STOCKPICKER_V1")
        order_details = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "order_type": order_type,
            "price": price,
            "exchange": exchange,
            "algo_id": algo_id,
        }

        if self.paper_mode:
            paper_id = f"PAPER_{int(time.time() * 1000)}"
            logger.info("P48 PAPER ORDER: %s", order_details)
            return {"paper_order_id": paper_id, **order_details}

        # Live order
        try:
            tx_type = self._kite.TRANSACTION_TYPE_BUY if side == "BUY" else self._kite.TRANSACTION_TYPE_SELL
            order_id = self._kite.place_order(
                variety=self._kite.VARIETY_REGULAR,
                exchange=exchange,
                tradingsymbol=symbol,
                transaction_type=tx_type,
                quantity=qty,
                order_type=getattr(self._kite, f"ORDER_TYPE_{order_type}", self._kite.ORDER_TYPE_LIMIT),
                product=self._kite.PRODUCT_CNC,
                price=price,
                tag=algo_id,
            )
            logger.info("P48 LIVE ORDER placed: order_id=%s  %s", order_id, order_details)
            return {"order_id": order_id, **order_details}
        except Exception as exc:
            logger.error("P48: order placement failed: %s  details=%s", exc, order_details)
            return {"error": str(exc), **order_details}

    # ------------------------------------------------------------------
    def place_stop_loss(
        self,
        symbol: str,
        qty: int,
        trigger_price: float,
        exchange: str = "BSE",
    ) -> Dict[str, Any]:
        """Place a SL-M (stop-loss market) sell order."""
        return self.place_order(
            symbol=symbol, qty=qty, side="SELL",
            order_type="SL", price=trigger_price, exchange=exchange,
        )

    # ------------------------------------------------------------------
    def get_positions(self) -> list:
        """Return open net positions."""
        if self.paper_mode:
            logger.debug("P48: get_positions() in paper mode — returns empty list.")
            return []
        try:
            return self._kite.positions().get("net", [])
        except Exception as exc:
            logger.warning("P48: get_positions failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an open order."""
        if self.paper_mode:
            logger.info("P48 PAPER CANCEL: order_id=%s", order_id)
            return {"cancelled": order_id}
        try:
            self._kite.cancel_order(variety=self._kite.VARIETY_REGULAR, order_id=order_id)
            return {"cancelled": order_id}
        except Exception as exc:
            logger.error("P48: cancel_order failed: %s", exc)
            return {"error": str(exc)}
