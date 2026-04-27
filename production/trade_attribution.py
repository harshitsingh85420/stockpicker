"""
production/trade_attribution.py -- P26: Trade attribution log + BSE safety filters.

Two components:

1. TradeAttributionLog
   Per-trade record of which model features and signals drove each entry.
   Stores SHAP top-features, regime state, VIX level, bucket, circuit-breaker
   state, and exit metadata.  Persisted to JSONL so performance can be
   attributed to signal quality post-trade.

2. BSESafetyFilter
   Validates each pick against BSE-specific hard constraints before
   order submission:
     - Price band / circuit limit (20% default for T-group, 5/10% for others)
     - Upper/lower circuit hit detection (close == high or == low)
     - Minimum lot / value threshold (SEBI: Rs 10,000 minimum)
     - T+1 settlement -- no same-day exit assumption
     - SEBI large-trader position limit (5% of company market-cap, default)
     - BSE SME / T-group / Z-group exclusion list
     - Illiquidity filter: 20-day avg volume < threshold

Usage
-----
from production.trade_attribution import TradeAttributionLog, BSESafetyFilter

log = TradeAttributionLog()
log.record_entry(
    sc_code='500325', entry_price=2800.0, shares=20,
    probability=0.72, model_version='v12',
    shap_top_features=[('RSI14', -0.18), ('ATRpct', 0.14), ('RS_Composite', 0.11)],
    regime='BULL', vix_level=17.4, bucket='MODERATE', circuit_state='NORMAL',
)

sf = BSESafetyFilter()
safe, reason = sf.check(row)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROOT        = Path(__file__).parent.parent
ATTR_DIR     = _ROOT / "stock_picker_data" / "attribution"
ATTR_DIR.mkdir(parents=True, exist_ok=True)

ATTR_LOG_PATH = ATTR_DIR / "trade_attribution.jsonl"


# ===========================================================================
# 1. TradeAttributionLog
# ===========================================================================

class TradeAttributionLog:
    """
    P26 -- Per-trade attribution recorder (JSONL, append-only).

    Each record links a trade to the exact signals, regime, and model state
    that generated it.  On exit, the record is updated with outcome.
    Because JSONL is append-only, updates are written as NEW records with
    event_type="EXIT_ATTRIBUTION" keyed by the same trade_id.
    """

    def __init__(self, log_path: Optional[Path] = None) -> None:
        self.log_path = log_path or ATTR_LOG_PATH

    # ------------------------------------------------------------------
    # Entry recording
    # ------------------------------------------------------------------

    def record_entry(
        self,
        sc_code:           str,
        entry_price:       float,
        shares:            int,
        probability:       float,
        model_version:     str      = "unknown",
        shap_top_features: List[Tuple[str, float]] = None,
        meta_probability:  Optional[float] = None,
        regime:            str      = "UNKNOWN",
        vix_level:         Optional[float] = None,
        vix_regime:        str      = "UNKNOWN",
        bucket:            str      = "MODERATE",
        circuit_state:     str      = "NORMAL",
        kelly_fraction:    Optional[float] = None,
        signal_date:       str      = "",
        trade_id:          Optional[str] = None,
    ) -> str:
        """
        Record entry attribution for a new trade.

        Returns the trade_id (use when calling record_exit later).
        """
        if trade_id is None:
            trade_id = f"{sc_code}_{signal_date or datetime.utcnow().strftime('%Y%m%d')}"

        record = {
            "event_type":       "ENTRY_ATTRIBUTION",
            "ts":               datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "trade_id":         trade_id,
            "sc_code":          sc_code,
            "signal_date":      signal_date,
            "entry_price":      round(entry_price, 2),
            "shares":           shares,
            "position_value":   round(entry_price * shares, 2),
            "probability":      round(probability, 6),
            "meta_probability": round(meta_probability, 6) if meta_probability is not None else None,
            "model_version":    model_version,
            "shap_top_features": [
                {"feature": f, "value": round(v, 6)} for f, v in (shap_top_features or [])
            ],
            "regime":           regime,
            "vix_level":        round(vix_level, 2) if vix_level is not None else None,
            "vix_regime":       vix_regime,
            "bucket":           bucket,
            "circuit_state":    circuit_state,
            "kelly_fraction":   round(kelly_fraction, 6) if kelly_fraction is not None else None,
        }
        self._append(record)
        logger.info(
            "P26 ENTRY_ATTR: %s  prob=%.3f  regime=%s  bucket=%s",
            sc_code, probability, regime, bucket,
        )
        return trade_id

    # ------------------------------------------------------------------
    # Exit recording
    # ------------------------------------------------------------------

    def record_exit(
        self,
        trade_id:    str,
        sc_code:     str,
        exit_price:  float,
        shares:      int,
        exit_reason: str,
        sessions_held: int,
        pnl:         float,
        gross_return: float,
        exit_date:   str = "",
    ) -> None:
        """Append exit outcome for a previously recorded entry."""
        record = {
            "event_type":    "EXIT_ATTRIBUTION",
            "ts":            datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "trade_id":      trade_id,
            "sc_code":       sc_code,
            "exit_date":     exit_date,
            "exit_price":    round(exit_price, 2),
            "shares":        shares,
            "exit_reason":   exit_reason,
            "sessions_held": sessions_held,
            "pnl":           round(pnl, 2),
            "gross_return":  round(gross_return, 6),
            "winner":        pnl > 0,
        }
        self._append(record)
        logger.info(
            "P26 EXIT_ATTR: %s  reason=%s  pnl=%.0f (%.2f%%)",
            sc_code, exit_reason, pnl, gross_return * 100,
        )

    # ------------------------------------------------------------------
    # Read / query helpers
    # ------------------------------------------------------------------

    def read_entries(self, n: int = 50) -> List[Dict[str, Any]]:
        """Return last n ENTRY_ATTRIBUTION records."""
        return [r for r in self._read_all()[-n * 2:] if r.get("event_type") == "ENTRY_ATTRIBUTION"][-n:]

    def read_exits(self, n: int = 50) -> List[Dict[str, Any]]:
        """Return last n EXIT_ATTRIBUTION records."""
        return [r for r in self._read_all()[-n * 2:] if r.get("event_type") == "EXIT_ATTRIBUTION"][-n:]

    def get_trade_pair(self, trade_id: str) -> Dict[str, Any]:
        """Return entry + exit records for a given trade_id."""
        records = [r for r in self._read_all() if r.get("trade_id") == trade_id]
        out: Dict[str, Any] = {}
        for r in records:
            out[r["event_type"]] = r
        return out

    def feature_win_rate(self) -> List[Dict[str, Any]]:
        """
        For each SHAP top-feature that appeared in entry records,
        compute the win rate of trades where it ranked #1.
        """
        all_records = self._read_all()
        entries = {r["trade_id"]: r for r in all_records if r.get("event_type") == "ENTRY_ATTRIBUTION"}
        exits   = {r["trade_id"]: r for r in all_records if r.get("event_type") == "EXIT_ATTRIBUTION"}

        feature_stats: Dict[str, Dict[str, int]] = {}
        for tid, entry in entries.items():
            top = entry.get("shap_top_features", [])
            if not top:
                continue
            top_feat = top[0].get("feature", "unknown") if isinstance(top[0], dict) else top[0][0]
            exit_rec = exits.get(tid)
            if exit_rec is None:
                continue
            fs = feature_stats.setdefault(top_feat, {"n": 0, "wins": 0})
            fs["n"] += 1
            if exit_rec.get("winner"):
                fs["wins"] += 1

        return [
            {"feature": f, "n": s["n"], "win_rate": round(s["wins"] / s["n"], 4) if s["n"] else None}
            for f, s in sorted(feature_stats.items(), key=lambda x: -x[1]["n"])
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _append(self, record: Dict[str, Any]) -> None:
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.error("P26 TradeAttributionLog: write failed: %s", exc)

    def _read_all(self) -> List[Dict[str, Any]]:
        if not self.log_path.exists():
            return []
        try:
            lines = self.log_path.read_text(encoding="utf-8").splitlines()
            return [json.loads(l) for l in lines if l.strip()]
        except Exception as exc:
            logger.warning("P26: attribution log read failed: %s", exc)
            return []


# ===========================================================================
# 2. BSESafetyFilter
# ===========================================================================

# BSE scrip groups that are excluded from algo trading
_EXCLUDED_GROUPS = frozenset({"Z", "XT", "ZP", "ZY"})

# Default price band by BSE group (fraction, symmetric)
_PRICE_BAND: Dict[str, float] = {
    "A":  0.20,
    "B":  0.20,
    "S":  0.05,   # SME segment -- tighter band
    "T":  0.05,   # Trade-to-Trade
    "TS": 0.05,
    "MT": 0.05,
    "X":  0.20,
    "DEFAULT": 0.20,
}


class BSESafetyFilter:
    """
    P26 -- BSE-specific hard-stop filters applied before order submission.

    Parameters
    ----------
    min_trade_value   : SEBI minimum trade value in INR (default 10,000).
    max_position_pct  : Max fraction of daily volume per order (default 5%).
    min_avg_volume    : 20-day average volume below which stock is illiquid (default 10,000).
    max_market_cap_pct: Max fraction of company market-cap per position (SEBI: 5%).
    """

    def __init__(
        self,
        min_trade_value:    float = 10_000.0,
        max_position_pct:   float = 0.05,
        min_avg_volume:     int   = 10_000,
        max_market_cap_pct: float = 0.05,
    ) -> None:
        self.min_trade_value    = min_trade_value
        self.max_position_pct   = max_position_pct
        self.min_avg_volume     = min_avg_volume
        self.max_market_cap_pct = max_market_cap_pct

    def check(self, row: pd.Series) -> Tuple[bool, str]:
        """
        Run all BSE safety checks on a single candidate pick row.

        Expected columns (use what's available):
          SC_CODE, Close, High, Low, SC_GROUP (optional), AvgVol20 (optional),
          MarketCap (optional), shares (optional, for value check)

        Returns
        -------
        (is_safe: bool, reason: str)   -- reason is empty string if safe.
        """
        sc_code = str(row.get("SC_CODE", row.get("sc_code", "UNKNOWN")))

        # 1. Excluded group check
        group = str(row.get("SC_GROUP", row.get("Group", ""))).upper().strip()
        if group in _EXCLUDED_GROUPS:
            return False, f"EXCLUDED_GROUP:{group}"

        close = float(row.get("Close", row.get("close", 0)) or 0)
        high  = float(row.get("High",  row.get("high",  0)) or 0)
        low   = float(row.get("Low",   row.get("low",   0)) or 0)

        if close <= 0:
            return False, "ZERO_PRICE"

        # 2. Circuit hit detection: if close == high (upper circuit) or close == low (lower circuit)
        if high > 0 and abs(close - high) / close < 0.001:
            return False, "UPPER_CIRCUIT_HIT"
        if low > 0 and abs(close - low) / close < 0.001:
            return False, "LOWER_CIRCUIT_HIT"

        # 3. Price band sanity -- previous close not in row, so we just flag extreme intraday move
        band = _PRICE_BAND.get(group, _PRICE_BAND["DEFAULT"])
        if high > 0 and low > 0:
            intraday_range = (high - low) / low
            if intraday_range > band + 0.02:   # small tolerance
                return False, f"INTRADAY_RANGE_EXCEEDS_BAND:{intraday_range:.2%}"

        # 4. Minimum trade value
        shares = int(row.get("shares", row.get("Shares", 0)) or 0)
        if shares > 0:
            trade_value = shares * close
            if trade_value < self.min_trade_value:
                return False, f"BELOW_MIN_TRADE_VALUE:{trade_value:.0f}<{self.min_trade_value:.0f}"

        # 5. Illiquidity check (20-day avg volume)
        avg_vol = float(row.get("AvgVol20", row.get("avg_vol_20", 0)) or 0)
        if avg_vol > 0 and avg_vol < self.min_avg_volume:
            return False, f"ILLIQUID:avg_vol={avg_vol:.0f}<{self.min_avg_volume}"

        # 6. Large-trader position limit (SEBI: 5% of market-cap)
        market_cap = float(row.get("MarketCap", row.get("market_cap", 0)) or 0)
        if market_cap > 0 and shares > 0:
            position_value = shares * close
            if position_value / market_cap > self.max_market_cap_pct:
                return False, (
                    f"EXCEEDS_MARKET_CAP_LIMIT:{position_value/market_cap:.2%}"
                    f">{self.max_market_cap_pct:.0%}"
                )

        # 7. Volume participation limit
        daily_vol = float(row.get("Volume", row.get("volume", 0)) or 0)
        if daily_vol > 0 and shares > 0 and shares / daily_vol > self.max_position_pct:
            return False, (
                f"EXCEEDS_VOLUME_PARTICIPATION:{shares/daily_vol:.2%}"
                f">{self.max_position_pct:.0%}"
            )

        # 8. T+1 settlement note: we don't allow intraday exits (informational only --
        #    the caller must not exit on the same day it enters)
        # This check is enforced at the orchestrator level; no rejection here.

        return True, ""

    def filter_picks(
        self,
        picks_df: pd.DataFrame,
        bhav_df:  Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply BSE safety filter to an entire picks DataFrame.

        Parameters
        ----------
        picks_df : DataFrame with at least SC_CODE, Close columns.
        bhav_df  : Optional full bhav DataFrame to enrich with High, Low, Volume, AvgVol20.

        Returns
        -------
        (safe_df, rejected_df)  -- rejected_df has an extra 'rejection_reason' column.
        """
        df = picks_df.copy()

        # Merge bhav columns if provided and not already present
        if bhav_df is not None:
            needed = [c for c in ["High", "Low", "Volume", "SC_GROUP", "MarketCap"]
                      if c not in df.columns and c in bhav_df.columns]
            if needed:
                sc_col  = "SC_CODE" if "SC_CODE" in bhav_df.columns else "sc_code"
                merge_cols = [sc_col] + needed
                df = df.merge(
                    bhav_df[merge_cols].drop_duplicates(sc_col),
                    left_on="SC_CODE", right_on=sc_col, how="left",
                )

        safe_rows = []
        rejected_rows = []
        for _, row in df.iterrows():
            is_safe, reason = self.check(row)
            if is_safe:
                safe_rows.append(row)
            else:
                row = row.copy()
                row["rejection_reason"] = reason
                rejected_rows.append(row)
                logger.warning("P26 BSE_FILTER: %s rejected -- %s", row.get("SC_CODE", "?"), reason)

        safe_df     = pd.DataFrame(safe_rows)
        rejected_df = pd.DataFrame(rejected_rows)
        logger.info(
            "P26 BSESafetyFilter: %d safe / %d rejected from %d picks",
            len(safe_df), len(rejected_df), len(df),
        )
        return safe_df, rejected_df
