"""
production/audit_trail.py -- P25: SEBI audit trail, human override, model versioning.

Three components:

1. AuditLogger
   Append-only JSONL log of every signal, order, override, and model event.
   Each entry is timestamped, tagged with a model version, and includes the
   operator ID.  Required for SEBI algo-trading compliance.

2. HumanOverrideGate
   Checks a persistent override list before any trade is entered.
   Operators can block a stock (BLOCK), force-exit (FORCE_EXIT), or cap
   the position (CAP_SIZE).

3. ModelVersionRegistry
   Tracks model versions (LightGBM + XGBoost), training dates, AUC scores,
   and which version is currently active.  Prevents using a stale or
   untested model in production.

Usage
-----
from production.audit_trail import AuditLogger, HumanOverrideGate, ModelVersionRegistry

audit = AuditLogger()
audit.log_signal('500325', 0.75, model_version='v12')
audit.log_order('500325', 'BUY', 200, 2800.0)

gate = HumanOverrideGate()
gate.add_override('500325', 'BLOCK', reason='Promoter pledging alert')
if gate.is_blocked('500325'):
    print('Trade blocked by operator')

registry = ModelVersionRegistry()
registry.register('v12', auc=0.664, notes='Retrained Apr-2026')
registry.set_active('v12')
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ROOT      = Path(__file__).parent.parent
AUDIT_DIR  = _ROOT / "stock_picker_data" / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_LOG_PATH    = AUDIT_DIR / "audit_trail.jsonl"
OVERRIDE_PATH     = AUDIT_DIR / "human_overrides.json"
MODEL_REGISTRY_PATH = AUDIT_DIR / "model_registry.json"


# ===========================================================================
# 1. AuditLogger
# ===========================================================================

class AuditLogger:
    """
    P25 -- Append-only audit log (JSONL format).

    Each log entry contains: timestamp, event_type, sc_code, operator_id,
    model_version, and event-specific payload.  The file is append-only and
    should never be modified (SEBI requirement).

    Parameters
    ----------
    log_path    : Path for the JSONL audit file.
    operator_id : Identifier of the human/system writing the log entry.
    """

    EVENT_TYPES = {
        "SIGNAL",       # Model generated a signal
        "ORDER",        # Order submitted (simulated or live)
        "FILL",         # Order filled
        "EXIT",         # Position closed
        "OVERRIDE",     # Human override applied
        "MODEL_LOAD",   # Model version loaded
        "MODEL_TRAIN",  # Retraining completed
        "CIRCUIT",      # Circuit breaker state change
        "SYSTEM",       # System event (start, stop, error)
    }

    def __init__(
        self,
        log_path:    Optional[Path] = None,
        operator_id: str = "SYSTEM",
    ) -> None:
        self.log_path    = log_path or AUDIT_LOG_PATH
        self.operator_id = operator_id

    def _write(self, event_type: str, payload: Dict[str, Any]) -> None:
        entry = {
            "ts":           datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "event":        event_type,
            "operator_id":  self.operator_id,
            **payload,
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.error("P25 AuditLogger: write failed: %s", exc)

    def log_signal(
        self,
        sc_code: str,
        probability: float,
        model_version: str = "unknown",
        meta_prob: Optional[float] = None,
        shap_top1: Optional[str] = None,
    ) -> None:
        self._write("SIGNAL", {
            "sc_code":       sc_code,
            "probability":   round(probability, 6),
            "model_version": model_version,
            "meta_prob":     meta_prob,
            "shap_top1":     shap_top1,
        })

    def log_order(
        self,
        sc_code:  str,
        side:     str,   # 'BUY' | 'SELL'
        shares:   int,
        price:    float,
        model_version: str = "unknown",
        bucket:   str = "MODERATE",
    ) -> None:
        self._write("ORDER", {
            "sc_code":       sc_code,
            "side":          side,
            "shares":        shares,
            "price":         round(price, 2),
            "value":         round(shares * price, 2),
            "model_version": model_version,
            "bucket":        bucket,
        })

    def log_exit(
        self,
        sc_code:  str,
        reason:   str,
        pnl:      float,
        sessions: int,
        model_version: str = "unknown",
    ) -> None:
        self._write("EXIT", {
            "sc_code":       sc_code,
            "reason":        reason,
            "pnl":           round(pnl, 2),
            "sessions_held": sessions,
            "model_version": model_version,
        })

    def log_override(self, sc_code: str, override_type: str, reason: str) -> None:
        self._write("OVERRIDE", {
            "sc_code":       sc_code,
            "override_type": override_type,
            "reason":        reason,
        })

    def log_model_event(self, event_type: str, version: str, auc: Optional[float] = None,
                        notes: str = "") -> None:
        assert event_type in ("MODEL_LOAD", "MODEL_TRAIN")
        self._write(event_type, {"version": version, "auc": auc, "notes": notes})

    def log_circuit(self, state: str, drawdown: float) -> None:
        self._write("CIRCUIT", {"state": state, "drawdown": round(drawdown, 4)})

    def log_system(self, message: str) -> None:
        self._write("SYSTEM", {"message": message})

    def read_recent(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return last n entries from the audit log."""
        if not self.log_path.exists():
            return []
        try:
            lines = self.log_path.read_text(encoding="utf-8").splitlines()
            return [json.loads(l) for l in lines[-n:] if l.strip()]
        except Exception as exc:
            logger.warning("P25: audit log read failed: %s", exc)
            return []


# ===========================================================================
# 2. HumanOverrideGate
# ===========================================================================

class HumanOverrideGate:
    """
    P25 -- Operator override list.

    Operators can add per-stock overrides:
      BLOCK      : Prevent any new entry.
      FORCE_EXIT : Mark for immediate exit on next run.
      CAP_SIZE   : Limit position to cap_value INR.
      NOTES_ONLY : Informational flag, does not block trading.

    State is persisted to a JSON file so overrides survive restarts.
    """

    VALID_TYPES = {"BLOCK", "FORCE_EXIT", "CAP_SIZE", "NOTES_ONLY"}

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or OVERRIDE_PATH
        self._overrides: Dict[str, Dict[str, Any]] = {}
        self._load()

    def add_override(
        self,
        sc_code:       str,
        override_type: str,
        reason:        str = "",
        cap_value:     Optional[float] = None,
        operator_id:   str = "OPERATOR",
    ) -> None:
        assert override_type in self.VALID_TYPES, f"Invalid type: {override_type}"
        self._overrides[sc_code] = {
            "type":        override_type,
            "reason":      reason,
            "cap_value":   cap_value,
            "operator_id": operator_id,
            "added_at":    datetime.now().isoformat(timespec="seconds"),
        }
        self._save()
        logger.warning(
            "P25 Override: %s -> %s by %s (%s)",
            sc_code, override_type, operator_id, reason,
        )

    def remove_override(self, sc_code: str) -> bool:
        if sc_code in self._overrides:
            del self._overrides[sc_code]
            self._save()
            logger.info("P25: Override removed for %s", sc_code)
            return True
        return False

    def is_blocked(self, sc_code: str) -> bool:
        ov = self._overrides.get(sc_code)
        return ov is not None and ov["type"] == "BLOCK"

    def needs_force_exit(self, sc_code: str) -> bool:
        ov = self._overrides.get(sc_code)
        return ov is not None and ov["type"] == "FORCE_EXIT"

    def get_cap(self, sc_code: str) -> Optional[float]:
        ov = self._overrides.get(sc_code)
        if ov and ov["type"] == "CAP_SIZE":
            return ov.get("cap_value")
        return None

    def check(self, sc_code: str) -> Dict[str, Any]:
        ov = self._overrides.get(sc_code)
        if ov is None:
            return {"action": "ALLOW", "reason": None}
        return {"action": ov["type"], "reason": ov["reason"], "cap_value": ov.get("cap_value")}

    def list_overrides(self) -> List[Dict[str, Any]]:
        return [{"sc_code": k, **v} for k, v in self._overrides.items()]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._overrides, indent=2))

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self._overrides = json.loads(self.path.read_text())
        except Exception as exc:
            logger.warning("P25: Override load failed: %s", exc)


# ===========================================================================
# 3. ModelVersionRegistry
# ===========================================================================

class ModelVersionRegistry:
    """
    P25 -- Track trained model versions and control which is active.

    Prevents deploying an untested or stale model to production.

    Parameters
    ----------
    min_auc : float
        Minimum acceptable CV AUC to allow a model into production (default 0.55).
    """

    def __init__(self, path: Optional[Path] = None, min_auc: float = 0.55) -> None:
        self.path    = path or MODEL_REGISTRY_PATH
        self.min_auc = min_auc
        self._registry: Dict[str, Any] = {"active_version": None, "versions": {}}
        self._load()

    def register(
        self,
        version: str,
        auc:     float,
        n_features:     int = 0,
        n_samples:      int = 0,
        trained_on:     str = "",
        notes:          str = "",
    ) -> None:
        self._registry["versions"][version] = {
            "auc":          round(auc, 6),
            "n_features":   n_features,
            "n_samples":    n_samples,
            "trained_on":   trained_on or str(datetime.now().date()),
            "notes":        notes,
            "registered_at": datetime.now().isoformat(timespec="seconds"),
            "approved":     False,
        }
        self._save()
        logger.info("P25: Model version %s registered (AUC=%.4f)", version, auc)

    def approve(self, version: str) -> None:
        """Mark a version as production-approved."""
        if version not in self._registry["versions"]:
            raise KeyError(f"Version {version} not registered.")
        v = self._registry["versions"][version]
        if v["auc"] < self.min_auc:
            raise ValueError(
                f"Version {version} AUC={v['auc']:.4f} < min_auc={self.min_auc} -- cannot approve."
            )
        v["approved"] = True
        self._save()
        logger.info("P25: Model version %s APPROVED for production.", version)

    def set_active(self, version: str) -> None:
        """Promote an approved version to active production."""
        v = self._registry["versions"].get(version)
        if v is None:
            raise KeyError(f"Version {version} not registered.")
        if not v.get("approved", False):
            raise PermissionError(
                f"Version {version} is not approved -- call approve() first."
            )
        self._registry["active_version"] = version
        self._save()
        logger.info("P25: Active model version set to %s", version)

    @property
    def active_version(self) -> Optional[str]:
        return self._registry.get("active_version")

    def get_active_info(self) -> Optional[Dict[str, Any]]:
        av = self.active_version
        if av is None:
            return None
        return {"version": av, **self._registry["versions"].get(av, {})}

    def list_versions(self) -> List[Dict[str, Any]]:
        return [
            {"version": k, **v}
            for k, v in self._registry["versions"].items()
        ]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._registry, indent=2))

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            self._registry = json.loads(self.path.read_text())
        except Exception as exc:
            logger.warning("P25: Registry load failed: %s", exc)
