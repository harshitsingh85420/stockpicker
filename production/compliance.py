"""
compliance.py
=============
Production compliance and audit trail for the Indian market stock picker.

Handles:
  - Gap 19: SEBI Retail Algo Framework Compliance
  - Immutable audit trail (one JSON file per trading day)
  - Order rate limits, algo tagging, position limit checks
  - CSV export for regulatory submission

Usage:
    from production.compliance import AuditTrail, SEBIComplianceChecker
    from production.compliance import create_audit_trail

    audit = create_audit_trail()
    audit.log_signal('500209', 'INFOSYS', probability=0.78, features_dict={}, regime='BULL')
"""

import csv
import json
import logging
import datetime
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid event types
# ---------------------------------------------------------------------------

VALID_EVENT_TYPES = {
    "ORDER_GENERATED",
    "ORDER_MODIFIED",
    "ORDER_CANCELLED",
    "TRADE_EXECUTED",
    "SIGNAL_GENERATED",
    "RISK_CHECK",
    "MODEL_PREDICTION",
    "SYSTEM_START",
    "SYSTEM_STOP",
    "KILL_SWITCH_ACTIVATED",
}


# ---------------------------------------------------------------------------
# AuditEvent dataclass
# ---------------------------------------------------------------------------

@dataclass
class AuditEvent:
    """
    Immutable record of a single auditable system event.

    Fields
    ------
    event_id : str
        UUID4 — unique identifier for this event.
    timestamp : str
        ISO-8601 timestamp (UTC).
    event_type : str
        One of VALID_EVENT_TYPES.
    sc_code : str
        BSE stock code (empty string for system events).
    sc_name : str
        Company name (empty string for system events).
    details : dict
        Event-specific payload.
    user_id : str
        Operator / account identifier.
    algo_id : str
        Algo strategy identifier registered with the broker.
    """

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat() + "Z"
    )
    event_type: str = "SIGNAL_GENERATED"
    sc_code: str = ""
    sc_name: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    user_id: str = "DEFAULT"
    algo_id: str = "STOCKPICKER_V1"

    def __post_init__(self) -> None:
        if self.event_type not in VALID_EVENT_TYPES:
            raise ValueError(
                f"Invalid event_type '{self.event_type}'. "
                f"Must be one of: {sorted(VALID_EVENT_TYPES)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# AuditTrail
# ---------------------------------------------------------------------------

class AuditTrail:
    """
    Append-only audit trail stored as daily JSON files.

    One file per trading day: ``<log_dir>/audit_YYYYMMDD.json``

    Each file contains a JSON array of AuditEvent dicts.  Appends are
    atomic at the file level (read -> append -> write) to avoid corruption
    from concurrent processes.

    Parameters
    ----------
    log_dir : str
        Directory where daily audit files are stored.
    algo_id : str
        SEBI-registered algo identifier.
    user_id : str
        Broker account / operator identifier.
    """

    def __init__(
        self,
        log_dir: str = "stock_picker_data/audit",
        algo_id: str = "STOCKPICKER_V1",
        user_id: str = "DEFAULT",
    ) -> None:
        self.log_dir = Path(log_dir)
        self.algo_id = algo_id
        self.user_id = user_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        logger.info("AuditTrail initialised — log_dir=%s  algo_id=%s", self.log_dir, self.algo_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _daily_log_path(self, date: Optional[datetime.date] = None) -> Path:
        d = date or datetime.date.today()
        return self.log_dir / f"audit_{d.strftime('%Y%m%d')}.json"

    def _read_daily_events(self, path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
            return []
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Could not read audit file %s: %s", path, exc)
            return []

    def _append_event(self, event: AuditEvent) -> None:
        path = self._daily_log_path()
        events = self._read_daily_events(path)
        events.append(event.to_dict())
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(events, fh, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.critical("AUDIT WRITE FAILED — event %s lost: %s", event.event_id, exc)
            raise

    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------

    def log_event(
        self,
        event_type: str,
        sc_code: str = "",
        sc_name: str = "",
        **details: Any,
    ) -> AuditEvent:
        """
        Create and persist an AuditEvent.

        All keyword arguments beyond the fixed fields are stored in the
        ``details`` dict, making this API forward-compatible.

        Returns the created AuditEvent for downstream use.
        """
        event = AuditEvent(
            event_type=event_type,
            sc_code=sc_code,
            sc_name=sc_name,
            details=dict(details),
            user_id=self.user_id,
            algo_id=self.algo_id,
        )
        self._append_event(event)
        logger.debug(
            "AUDIT  %s  %s  %s  id=%s",
            event.event_type,
            event.sc_code,
            event.timestamp,
            event.event_id,
        )
        return event

    # ------------------------------------------------------------------
    # Domain-specific helpers
    # ------------------------------------------------------------------

    def log_signal(
        self,
        sc_code: str,
        sc_name: str,
        probability: float,
        features_dict: Dict[str, Any],
        regime: str,
    ) -> AuditEvent:
        """Log a model-generated trading signal."""
        return self.log_event(
            "SIGNAL_GENERATED",
            sc_code=sc_code,
            sc_name=sc_name,
            probability=probability,
            regime=regime,
            feature_count=len(features_dict),
            # Store only the top-level feature values (truncate to 30 features)
            features=dict(list(features_dict.items())[:30]),
        )

    def log_order(
        self,
        sc_code: str,
        sc_name: str,
        order_type: str,
        quantity: int,
        price: float,
        order_id: str,
    ) -> AuditEvent:
        """Log an order placement."""
        return self.log_event(
            "ORDER_GENERATED",
            sc_code=sc_code,
            sc_name=sc_name,
            order_type=order_type,
            quantity=quantity,
            price=price,
            order_id=order_id,
            notional_value=round(quantity * price, 2),
        )

    def log_trade(
        self,
        sc_code: str,
        sc_name: str,
        trade_type: str,
        quantity: int,
        price: float,
        trade_id: str,
        order_id: str,
    ) -> AuditEvent:
        """Log a trade execution confirmation."""
        return self.log_event(
            "TRADE_EXECUTED",
            sc_code=sc_code,
            sc_name=sc_name,
            trade_type=trade_type,
            quantity=quantity,
            price=price,
            trade_id=trade_id,
            order_id=order_id,
            notional_value=round(quantity * price, 2),
        )

    def log_risk_check(
        self,
        check_type: str,
        result: bool,
        details: Dict[str, Any],
    ) -> AuditEvent:
        """Log the outcome of a risk check."""
        return self.log_event(
            "RISK_CHECK",
            check_type=check_type,
            result="PASS" if result else "FAIL",
            **details,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_daily_log(self, date: Optional[datetime.date] = None) -> List[AuditEvent]:
        """
        Return all AuditEvents for a given date.

        Returns
        -------
        list of AuditEvent
        """
        path = self._daily_log_path(date or datetime.date.today())
        raw = self._read_daily_events(path)
        events = []
        for item in raw:
            try:
                events.append(AuditEvent(**item))
            except (TypeError, ValueError) as exc:
                logger.warning("Skipping malformed audit record: %s — %s", item, exc)
        return events

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_audit_report(
        self,
        start_date: datetime.date,
        end_date: datetime.date,
        output_file: str,
    ) -> None:
        """
        Export all audit events between ``start_date`` and ``end_date``
        (inclusive) to a CSV file suitable for regulatory submission.

        Parameters
        ----------
        start_date, end_date : datetime.date
        output_file : str
            Destination CSV path.
        """
        all_events: List[Dict[str, Any]] = []
        current = start_date
        while current <= end_date:
            path = self._daily_log_path(current)
            all_events.extend(self._read_daily_events(path))
            current += datetime.timedelta(days=1)

        if not all_events:
            logger.warning(
                "No audit events found between %s and %s", start_date, end_date
            )

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Flatten the 'details' dict into top-level columns
        flat_rows = []
        for ev in all_events:
            row = {k: v for k, v in ev.items() if k != "details"}
            details = ev.get("details", {})
            for k, v in details.items():
                row[f"detail_{k}"] = v
            flat_rows.append(row)

        if not flat_rows:
            flat_rows = [{}]

        fieldnames = sorted(
            {col for row in flat_rows for col in row},
            key=lambda x: (x.startswith("detail_"), x),
        )

        try:
            with open(output_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(flat_rows)
            logger.info(
                "Audit report exported: %s (%d events, %s to %s)",
                output_path,
                len(all_events),
                start_date,
                end_date,
            )
        except OSError as exc:
            logger.error("Failed to export audit report: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Completeness validation
    # ------------------------------------------------------------------

    def validate_audit_completeness(
        self, date: Optional[datetime.date] = None
    ) -> Dict[str, Any]:
        """
        Check whether all expected event types are present for a given day.

        A complete trading day should have at minimum:
            SYSTEM_START, SIGNAL_GENERATED, RISK_CHECK, SYSTEM_STOP

        Returns
        -------
        dict
            is_complete, present_types, missing_types, total_events
        """
        required = {"SYSTEM_START", "SIGNAL_GENERATED", "RISK_CHECK"}
        events = self.get_daily_log(date or datetime.date.today())
        present = {e.event_type for e in events}
        missing = required - present

        return {
            "is_complete": len(missing) == 0,
            "present_types": sorted(present),
            "missing_types": sorted(missing),
            "total_events": len(events),
            "date": (date or datetime.date.today()).isoformat(),
        }


# ---------------------------------------------------------------------------
# SEBIComplianceChecker
# ---------------------------------------------------------------------------

class SEBIComplianceChecker:
    """
    Checks orders and system behaviour against SEBI Retail Algo Framework
    requirements (applicable from April 2024 onwards).

    Key constraints enforced
    ------------------------
    - Maximum order rate: 10/sec, 100/min
    - All orders must carry an algo_id tag
    - No single position may exceed 15% of total capital

    Parameters
    ----------
    max_orders_per_second : int
    max_orders_per_minute : int
    broker_api_algo_id : str, optional
        The algo identifier that the broker API expects in each order.
    """

    SEBI_MAX_SINGLE_POSITION_PCT = 0.15  # 15% per stock for retail algo

    def __init__(
        self,
        max_orders_per_second: int = 10,
        max_orders_per_minute: int = 100,
        broker_api_algo_id: Optional[str] = None,
    ) -> None:
        self.max_orders_per_second = max_orders_per_second
        self.max_orders_per_minute = max_orders_per_minute
        self.broker_api_algo_id = broker_api_algo_id
        self._violations: List[Dict[str, Any]] = []

    def _record_violation(self, check: str, message: str) -> None:
        self._violations.append(
            {
                "check": check,
                "message": message,
                "at": datetime.datetime.utcnow().isoformat() + "Z",
            }
        )
        logger.warning("COMPLIANCE VIOLATION [%s]: %s", check, message)

    # ------------------------------------------------------------------
    # Order rate check
    # ------------------------------------------------------------------

    def check_order_rate(
        self, recent_orders_timestamps: List[datetime.datetime]
    ) -> Tuple[bool, str]:
        """
        Verify that order frequency is within SEBI limits.

        Parameters
        ----------
        recent_orders_timestamps : list of datetime
            Timestamps of recently placed orders (any rolling window).

        Returns
        -------
        (is_compliant, message) : (bool, str)
        """
        now = datetime.datetime.utcnow()

        one_second_ago = now - datetime.timedelta(seconds=1)
        one_minute_ago = now - datetime.timedelta(minutes=1)

        count_1s = sum(1 for t in recent_orders_timestamps if t >= one_second_ago)
        count_1m = sum(1 for t in recent_orders_timestamps if t >= one_minute_ago)

        if count_1s > self.max_orders_per_second:
            msg = (
                f"Order rate {count_1s}/sec exceeds limit of "
                f"{self.max_orders_per_second}/sec"
            )
            self._record_violation("ORDER_RATE_PER_SECOND", msg)
            return False, msg

        if count_1m > self.max_orders_per_minute:
            msg = (
                f"Order rate {count_1m}/min exceeds limit of "
                f"{self.max_orders_per_minute}/min"
            )
            self._record_violation("ORDER_RATE_PER_MINUTE", msg)
            return False, msg

        return True, (
            f"Order rate compliant: {count_1s}/sec, {count_1m}/min"
        )

    # ------------------------------------------------------------------
    # Algo tag validation
    # ------------------------------------------------------------------

    def validate_algo_tag(self, order_dict: Dict[str, Any]) -> bool:
        """
        Verify that an order dict carries the required algo_id tag.

        The broker API requires every algo-generated order to be tagged
        with the SEBI-registered algo identifier.

        Returns True if the tag is present and (optionally) matches the
        configured broker algo ID.
        """
        algo_id_in_order = order_dict.get("algo_id") or order_dict.get("algoId")

        if not algo_id_in_order:
            self._record_violation(
                "MISSING_ALGO_TAG",
                f"Order missing algo_id tag: {order_dict.get('order_id', 'unknown')}",
            )
            return False

        if self.broker_api_algo_id and algo_id_in_order != self.broker_api_algo_id:
            self._record_violation(
                "WRONG_ALGO_TAG",
                f"Order algo_id '{algo_id_in_order}' != expected '{self.broker_api_algo_id}'",
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Position limits
    # ------------------------------------------------------------------

    def check_position_limits(
        self,
        positions_dict: Dict[str, float],
        total_capital: float,
    ) -> Tuple[bool, str]:
        """
        Verify no single stock position exceeds 15% of total capital
        (SEBI individual stock concentration limit for retail algo accounts).

        Parameters
        ----------
        positions_dict : dict
            {sc_code: current_market_value}
        total_capital : float
            Total portfolio capital.

        Returns
        -------
        (is_compliant, message) : (bool, str)
        """
        if total_capital <= 0:
            return False, "total_capital must be positive."

        breaches = []
        for sc_code, market_value in positions_dict.items():
            pct = market_value / total_capital
            if pct > self.SEBI_MAX_SINGLE_POSITION_PCT:
                msg = (
                    f"{sc_code}: {pct:.1%} of capital exceeds "
                    f"SEBI limit of {self.SEBI_MAX_SINGLE_POSITION_PCT:.0%}"
                )
                breaches.append(msg)
                self._record_violation("POSITION_LIMIT", msg)

        if breaches:
            return False, "; ".join(breaches)

        return True, (
            f"All {len(positions_dict)} positions within "
            f"{self.SEBI_MAX_SINGLE_POSITION_PCT:.0%} SEBI limit."
        )

    # ------------------------------------------------------------------
    # Daily compliance report
    # ------------------------------------------------------------------

    def generate_compliance_report(self, date: Optional[datetime.date] = None) -> Dict[str, Any]:
        """
        Generate a SEBI compliance summary for a trading day.

        Uses the AuditTrail to count orders and trades.
        """
        date = date or datetime.date.today()
        audit = AuditTrail()
        events = audit.get_daily_log(date)

        order_count = sum(1 for e in events if e.event_type == "ORDER_GENERATED")
        trade_count = sum(1 for e in events if e.event_type == "TRADE_EXECUTED")
        signal_count = sum(1 for e in events if e.event_type == "SIGNAL_GENERATED")

        session_violations = list(self._violations)
        status = "COMPLIANT" if not session_violations else "NON_COMPLIANT"

        return {
            "date": date.isoformat(),
            "order_count": order_count,
            "trade_count": trade_count,
            "signal_count": signal_count,
            "violations": session_violations,
            "violation_count": len(session_violations),
            "status": status,
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "algo_id": self.broker_api_algo_id or "NOT_CONFIGURED",
        }


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def create_audit_trail(
    log_dir: str = "stock_picker_data/audit",
    algo_id: str = "STOCKPICKER_V1",
    user_id: str = "DEFAULT",
) -> AuditTrail:
    """
    Factory that creates an AuditTrail with default settings and logs SYSTEM_START.

    Returns
    -------
    AuditTrail
    """
    trail = AuditTrail(log_dir=log_dir, algo_id=algo_id, user_id=user_id)
    trail.log_event(
        "SYSTEM_START",
        system_version="1.0",
        started_at=datetime.datetime.utcnow().isoformat() + "Z",
    )
    return trail


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )

    print("=" * 60)
    print("AuditTrail demo")
    print("=" * 60)

    trail = create_audit_trail(log_dir="stock_picker_data/demo_audit")

    trail.log_signal(
        sc_code="500209",
        sc_name="INFOSYS",
        probability=0.78,
        features_dict={"rsi": 62.3, "macd": 0.5, "momentum_5d": 0.04},
        regime="BULL",
    )

    trail.log_order(
        sc_code="500209",
        sc_name="INFOSYS",
        order_type="BUY",
        quantity=10,
        price=1850.0,
        order_id="ORD20240423001",
    )

    trail.log_trade(
        sc_code="500209",
        sc_name="INFOSYS",
        trade_type="BUY",
        quantity=10,
        price=1852.5,
        trade_id="TRD20240423001",
        order_id="ORD20240423001",
    )

    trail.log_risk_check(
        check_type="POSITION_SIZE",
        result=True,
        details={"proposed_pct": 0.04, "limit_pct": 0.15},
    )

    events = trail.get_daily_log()
    print(f"  Events logged today: {len(events)}")
    for e in events:
        print(f"    {e.timestamp}  {e.event_type:25s}  {e.sc_code}")

    completeness = trail.validate_audit_completeness()
    print(f"\n  Audit completeness: {completeness}")

    print()
    print("=" * 60)
    print("SEBIComplianceChecker demo")
    print("=" * 60)

    checker = SEBIComplianceChecker(broker_api_algo_id="STOCKPICKER_V1")

    # Order rate check
    now = datetime.datetime.utcnow()
    timestamps = [now - datetime.timedelta(seconds=i * 0.1) for i in range(5)]
    ok, msg = checker.check_order_rate(timestamps)
    print(f"  Order rate check: compliant={ok}  msg={msg}")

    # Algo tag check
    good_order = {"order_id": "X1", "algo_id": "STOCKPICKER_V1"}
    bad_order = {"order_id": "X2"}
    print(f"  Algo tag (good): {checker.validate_algo_tag(good_order)}")
    print(f"  Algo tag (bad):  {checker.validate_algo_tag(bad_order)}")

    # Position limits
    positions = {"500209": 70_000, "532540": 30_000, "500112": 25_000}
    ok, msg = checker.check_position_limits(positions, total_capital=500_000)
    print(f"  Position limits: compliant={ok}  msg={msg}")

    # Compliance report
    report = checker.generate_compliance_report()
    print(f"\n  Compliance report: status={report['status']}  violations={report['violation_count']}")
