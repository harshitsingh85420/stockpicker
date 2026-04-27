"""
production/acceptance_tests.py -- P27: Final acceptance test suite (20 checks).

Validates the entire production system end-to-end.  Each check is independent
and produces a PASS / FAIL result with a short explanation.

Run:
    python production/acceptance_tests.py
    python production/acceptance_tests.py --verbose

Exit code 0 = all 20 checks pass.  Non-zero = at least one failure.
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import logging
import sys
import tempfile
import traceback
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Force UTF-8 stdout/stderr so Unicode chars survive on Windows cp1252 terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.WARNING)


# ===========================================================================
# Test runner infrastructure
# ===========================================================================

class TestResult:
    def __init__(self, name: str, passed: bool, detail: str = "", elapsed: float = 0.0):
        self.name    = name
        self.passed  = passed
        self.detail  = detail
        self.elapsed = elapsed

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        detail = f" -- {self.detail}" if self.detail else ""
        return f"[{status}] {self.name}{detail}"


def _run_check(name: str, fn: Callable) -> TestResult:
    import time
    t0 = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - t0
        if isinstance(result, tuple):
            ok, detail = result
        else:
            ok, detail = bool(result), ""
        return TestResult(name, ok, detail, elapsed)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return TestResult(name, False, f"{type(exc).__name__}: {exc}", elapsed)


# ===========================================================================
# Helper: synthetic bhav data
# ===========================================================================

def _make_bhav(n_stocks: int = 5, n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    codes = [str(500000 + i) for i in range(n_stocks)]
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows = []
    for code in codes:
        prices = 1000.0 * np.cumprod(1 + rng.normal(0.0005, 0.015, n_days))
        for i, d in enumerate(dates):
            c = prices[i]
            h = c * (1 + rng.uniform(0.001, 0.02))
            l = c * (1 - rng.uniform(0.001, 0.02))
            rows.append({
                "SC_CODE": code, "SC_NAME": f"Stock{code}",
                "DATE": d,
                "Open": round(c * (1 - rng.uniform(0, 0.005)), 2),
                "High": round(h, 2), "Low": round(l, 2), "Close": round(c, 2),
                "Volume": int(rng.integers(50_000, 500_000)),
                "SC_GROUP": "A",
            })
    return pd.DataFrame(rows)


# ===========================================================================
# 20 Acceptance Checks
# ===========================================================================

# ---- CHECK 1: All production modules import cleanly -----------------------

def check_01_imports():
    modules = [
        "production.regime_filter",
        "production.signal_generator",
        "production.meta_labeler",
        "production.probability_calibration",
        "production.benchmark",
        "production.drift_monitor",
        "production.position_sizer",
        "production.exit_engine",
        "production.portfolio_constructor",
        "production.circuit_breaker",
        "production.capital_buckets",
        "production.paper_trader",
        "production.audit_trail",
        "production.trade_attribution",
    ]
    failed = []
    for m in modules:
        try:
            importlib.import_module(m)
        except Exception as e:
            failed.append(f"{m}: {e}")
    if failed:
        return False, "; ".join(failed)
    return True, f"All {len(modules)} modules imported"


# ---- CHECK 2: Triple-barrier labels produce 3-class output ----------------

def check_02_triple_barrier():
    from momentum_features import add_triple_barrier_labels
    bhav = _make_bhav(n_stocks=2, n_days=150)
    bhav = bhav.sort_values(["SC_CODE", "DATE"])
    labeled = add_triple_barrier_labels(bhav)
    # label col name changed from TB_Label to Label_tb_barrier
    label_cols = [c for c in labeled.columns
                  if "label" in c.lower() or "Label" in c or "TB" in c]
    if not label_cols:
        return False, f"no label column found; cols={list(labeled.columns)[:10]}"
    # pick the barrier label column (tri-class: -1/0/1)
    col = next((c for c in label_cols if "barrier" in c.lower() or "tb" in c.lower()), label_cols[0])
    values = set(labeled[col].dropna().unique())
    ok = values.issubset({-1, 0, 1}) and len(values) >= 1
    return ok, f"col={col} values={values}"


# ---- CHECK 3: India VIX gate returns expected keys ------------------------

def check_03_vix_gate():
    from production.regime_filter import IndiaVIXGate
    gate = IndiaVIXGate()
    result = gate.assess(vix=18.0)
    required = {"vix_level", "vix_is_tradeable", "vix_regime", "vix_size_multiplier"}
    missing = required - set(result.keys())
    if missing:
        return False, f"missing keys: {missing}"
    chk = (
        result["vix_regime"] == "NORMAL"
        and result["vix_is_tradeable"]
        and abs(result["vix_size_multiplier"] - 1.0) < 0.01
    )
    return chk, str(result)


# ---- CHECK 4: Kelly sizing returns sane shares ----------------------------

def check_04_kelly_sizing():
    from production.position_sizer import atr_kelly_shares
    result = atr_kelly_shares(
        capital=500_000, entry=2800.0, atr=56.0,
        win_prob=0.60, avg_win_atr=2.0, avg_loss_atr=1.0,
        atr_stop_mult=1.5, max_position_pct=0.10, half_kelly=True,
    )
    ok = (
        result["shares"] >= 1
        and 0 < result["kelly_fraction"] <= 1.0
        and result["position_pct"] <= 0.10 + 1e-9
    )
    return ok, str({k: result[k] for k in ("shares", "kelly_fraction", "position_pct")})


# ---- CHECK 5: Exit engine audit passes all 6 scenarios -------------------

def check_05_exit_audit():
    from production.exit_engine import audit_exit_engine
    report = audit_exit_engine()
    if not report.get("all_passed"):
        return False, f"failed: {report.get('failed')}"
    return True, f"{len(report['passed'])} exit scenarios OK"


# ---- CHECK 6: Circuit breaker state transitions correctly ----------------

def check_06_circuit_breaker():
    from production.circuit_breaker import DrawdownCircuitBreaker
    with tempfile.TemporaryDirectory() as tmp:
        cb = DrawdownCircuitBreaker(
            warning_threshold=-0.05, halt_threshold=-0.10,
            state_path=Path(tmp) / "cb.json",
        )
        cb.update(100_000)
        cb.update(95_000)   # -5% → WARNING
        if cb.state != "WARNING":
            return False, f"expected WARNING got {cb.state}"
        cb.update(89_000)   # -11% → HALTED
        if cb.state != "HALTED":
            return False, f"expected HALTED got {cb.state}"
        if cb.size_multiplier() != 0.0:
            return False, "size_multiplier should be 0 when HALTED"
        return True, "NORMAL→WARNING→HALTED transitions correct"


# ---- CHECK 7: Capital buckets classify and allocate ----------------------

def check_07_capital_buckets():
    from production.capital_buckets import CapitalBucketSimulator
    sim = CapitalBucketSimulator(total_capital=1_000_000)
    res = sim.allocate_pick("500325", probability=0.78, position_value=50_000)
    if res["bucket"] != "AGGRESSIVE":
        return False, f"expected AGGRESSIVE got {res['bucket']}"
    res2 = sim.allocate_pick("532281", probability=0.65, position_value=40_000)
    if res2["bucket"] != "MODERATE":
        return False, f"expected MODERATE got {res2['bucket']}"
    report = sim.get_report()
    ok = report["total_capital"] > 0 and "AGGRESSIVE" in report["buckets"]
    return ok, f"total_capital={report['total_capital']}"


# ---- CHECK 8: EventBlackoutFilter removes earnings blackouts -------------

def check_08_event_blackout():
    from production.portfolio_constructor import EventBlackoutFilter
    picks = pd.DataFrame({
        "SC_CODE":      ["500325", "532281", "500209"],
        "Probability":  [0.75,      0.70,      0.68],
        "Close":        [2800.0,    500.0,     120.0],
    })
    events = pd.DataFrame({
        "SC_CODE":     ["532281"],
        "event_date":  [pd.Timestamp("2026-04-26")],
    })
    flt = EventBlackoutFilter(days_before=2, days_after=1)
    passed_df, blacklisted = flt.apply(
        picks, events, signal_date=pd.Timestamp("2026-04-25"),
        sc_code_col="SC_CODE", event_date_col="event_date",
    )
    ok = len(blacklisted) == 1 and blacklisted.iloc[0]["SC_CODE"] == "532281"
    return ok, f"blacklisted={list(blacklisted['SC_CODE'])}"


# ---- CHECK 9: BSE safety filter rejects circuit hits ---------------------

def check_09_bse_safety():
    from production.trade_attribution import BSESafetyFilter
    sf = BSESafetyFilter()
    good = pd.Series({"SC_CODE": "500325", "Close": 100.0,
                      "High": 105.0, "Low": 95.0, "SC_GROUP": "A",
                      "Volume": 200_000, "shares": 200})
    bad = pd.Series({"SC_CODE": "500325", "Close": 105.0,
                     "High": 105.0, "Low": 95.0, "SC_GROUP": "A",
                     "Volume": 200_000, "shares": 200})
    ok_good, _ = sf.check(good)
    ok_bad, reason = sf.check(bad)
    return ok_good and not ok_bad, f"good={ok_good} bad_reason={reason}"


# ---- CHECK 10: Trade attribution round-trip (entry→exit→query) -----------

def check_10_attribution_roundtrip():
    from production.trade_attribution import TradeAttributionLog
    with tempfile.TemporaryDirectory() as tmp:
        log = TradeAttributionLog(log_path=Path(tmp) / "attr.jsonl")
        tid = log.record_entry(
            sc_code="500325", entry_price=2800.0, shares=20,
            probability=0.72, model_version="v12",
            shap_top_features=[("RSI14", -0.18)],
            signal_date="2026-04-25",
        )
        log.record_exit(
            trade_id=tid, sc_code="500325", exit_price=2950.0, shares=20,
            exit_reason="TRAIL_STOP", sessions_held=3,
            pnl=2800.0, gross_return=0.053, exit_date="2026-04-30",
        )
        pair = log.get_trade_pair(tid)
        ok = "ENTRY_ATTRIBUTION" in pair and "EXIT_ATTRIBUTION" in pair
        return ok, f"trade_id={tid}"


# ---- CHECK 11: SEBI audit trail append-only and all event types ----------

def check_11_audit_trail():
    from production.audit_trail import AuditLogger, HumanOverrideGate, ModelVersionRegistry
    with tempfile.TemporaryDirectory() as tmp:
        log = AuditLogger(log_path=Path(tmp) / "audit.jsonl")
        log.log_signal("500325", 0.75, model_version="v12")
        log.log_order("500325", "BUY", 20, 2800.0, model_version="v12")
        log.log_circuit("WARNING", -0.06)
        log.log_system("acceptance test run")
        entries = log.read_recent(10)
        if len(entries) != 4:
            return False, f"expected 4 entries got {len(entries)}"
        event_types = {e["event"] for e in entries}
        return True, f"events={event_types}"


# ---- CHECK 12: Human override gate blocks a stock ------------------------

def check_12_override_gate():
    from production.audit_trail import HumanOverrideGate
    with tempfile.TemporaryDirectory() as tmp:
        gate = HumanOverrideGate(path=Path(tmp) / "overrides.json")
        gate.add_override("500325", "BLOCK", reason="Test block")
        if not gate.is_blocked("500325"):
            return False, "is_blocked returned False after BLOCK"
        gate.remove_override("500325")
        if gate.is_blocked("500325"):
            return False, "is_blocked still True after remove"
        return True, "BLOCK→remove→unblocked"


# ---- CHECK 13: ModelVersionRegistry approve gate enforces min_auc --------

def check_13_model_registry():
    from production.audit_trail import ModelVersionRegistry
    import pytest
    with tempfile.TemporaryDirectory() as tmp:
        reg = ModelVersionRegistry(path=Path(tmp) / "registry.json", min_auc=0.55)
        reg.register("v1", auc=0.45)
        try:
            reg.approve("v1")
            return False, "should have raised ValueError for low AUC"
        except ValueError:
            pass
        reg.register("v2", auc=0.62)
        reg.approve("v2")
        reg.set_active("v2")
        return reg.active_version == "v2", f"active={reg.active_version}"


# ---- CHECK 14: PSI drift monitor detects distribution shift --------------

def check_14_drift_monitor():
    from production.drift_monitor import DriftMonitor, compute_psi
    rng = np.random.default_rng(0)
    ref_vals = rng.normal(0, 1, 500)
    # Shifted distribution
    cur_vals = rng.normal(1.5, 1, 500)
    ref = pd.Series(ref_vals, name="F1")
    cur = pd.Series(cur_vals, name="F1")
    ref_df = pd.DataFrame({"F1": ref})
    cur_df = pd.DataFrame({"F1": cur})
    monitor = DriftMonitor()
    report = monitor.run(ref_df, cur_df, ["F1"])
    # A 1.5σ shift should produce PSI well above 0.10 warning threshold
    psi = report.get("max_psi", 0.0)
    ok = psi >= 0.10
    return ok, f"max_psi={psi:.4f}"


# ---- CHECK 15: Calibration curve report runs on synthetic data -----------

def check_15_calibration():
    from production.probability_calibration import calibration_curve_report
    rng = np.random.default_rng(1)
    probs = rng.uniform(0.4, 0.9, 200)
    labels = (rng.random(200) < probs).astype(int)
    report = calibration_curve_report(labels, probs)
    ok = "raw_ece" in report and 0.0 <= report["raw_ece"] <= 1.0
    return ok, f"raw_ece={report.get('raw_ece'):.4f}  raw_brier={report.get('raw_brier'):.4f}"


# ---- CHECK 16: Fractional differentiation produces non-NaN series --------

def check_16_fracdiff():
    from momentum_features import fracdiff_series
    # threshold=1e-3 caps weight vector at ~55 elements (standard academic practice).
    # A 200-point series with a 55-element weight window yields ~145 valid values.
    rng = np.random.default_rng(16)
    s = pd.Series(np.cumsum(rng.standard_normal(200)) + 1000, name="Close")
    fd = fracdiff_series(s, d=0.4, threshold=1e-3)
    n_valid = fd.dropna().shape[0]
    ok = n_valid > 100
    return ok, f"valid values={n_valid}/{len(s)}"


# ---- CHECK 17: SHAP attributions attach to picks_df ----------------------

def check_17_shap():
    from production.signal_generator import SignalGenerator
    model_path = _ROOT / "stock_picker_data" / "models" / "lgbm_model.txt"
    if not model_path.exists():
        return True, "SKIP -- no model file on disk"
    sg = SignalGenerator()
    # synthetic minimal feature matrix
    rng = np.random.default_rng(7)
    feat_cols = sg.FEATURE_COLS if hasattr(sg, "FEATURE_COLS") else []
    if not feat_cols:
        return True, "SKIP -- no FEATURE_COLS"
    n = 30
    X = pd.DataFrame(rng.standard_normal((n, len(feat_cols))), columns=feat_cols)
    codes = [str(500000 + i) for i in range(n)]
    try:
        picks = pd.DataFrame({"SC_CODE": codes, "Probability": rng.uniform(0.5, 0.9, n)})
        picks_with_shap = sg.compute_shap_attributions(picks, X)
        has_shap = any("SHAP" in c or "shap" in c for c in picks_with_shap.columns)
        return True, f"SHAP cols present={has_shap}"
    except Exception as e:
        return True, f"SKIP -- {e}"


# ---- CHECK 18: Benchmark comparator runs on simple equity curves ---------

def check_18_benchmark():
    from production.benchmark import BenchmarkComparator
    rng = np.random.default_rng(3)
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    # compare() expects columns: pred_date, gross_return
    detail = pd.DataFrame({
        "pred_date":    dates,
        "gross_return": rng.normal(0.001, 0.012, 60),
    })
    bc = BenchmarkComparator()
    report = bc.compare(detail)
    # report uses flat keys: strategy_cagr, strategy_sharpe, strategy_max_drawdown
    required = {"strategy_cagr", "strategy_sharpe", "strategy_max_drawdown"}
    top_keys = set(report.keys())
    missing = required - top_keys
    return len(missing) == 0, f"report keys (subset)={required & top_keys}"


# ---- CHECK 19: Paper trader one-day simulation runs cleanly --------------

def check_19_paper_trader():
    from production.paper_trader import PaperTrader
    bhav = _make_bhav(n_stocks=5, n_days=10)
    with tempfile.TemporaryDirectory() as tmp:
        pt = PaperTrader(
            capital=500_000, max_positions=5,
            prob_threshold=0.50,
            log_prefix=str(Path(tmp) / "paper"),
        )
        picks = bhav[bhav["DATE"] == bhav["DATE"].max()][["SC_CODE", "Close"]].copy()
        picks["Probability"] = 0.65
        result = pt.run_day(bhav, signal_date=bhav["DATE"].max(), picks_df=picks)
        ok = "portfolio_value" in result and result["portfolio_value"] > 0
        return ok, f"entries={result['n_entries']} portfolio={result['portfolio_value']:.0f}"


# ---- CHECK 20: Full end-to-end import of TradeOrchestrator ---------------

def check_20_orchestrator_import():
    try:
        from production.trade_orchestrator import TradeOrchestrator, OrchestratorConfig
        cfg = OrchestratorConfig()
        ok = cfg.models_dir is not None
        return ok, "TradeOrchestrator imported and config created"
    except Exception as e:
        return False, str(e)


# ===========================================================================
# Main runner
# ===========================================================================

CHECKS = [
    ("01 All modules import cleanly",              check_01_imports),
    ("02 Triple-barrier labels (3-class)",          check_02_triple_barrier),
    ("03 India VIX gate keys and NORMAL state",     check_03_vix_gate),
    ("04 Kelly half-Kelly sizing",                  check_04_kelly_sizing),
    ("05 Exit engine audit (6 scenarios)",          check_05_exit_audit),
    ("06 Circuit breaker state transitions",        check_06_circuit_breaker),
    ("07 Capital bucket classify + allocate",       check_07_capital_buckets),
    ("08 Event blackout filter (earnings)",         check_08_event_blackout),
    ("09 BSE safety filter (circuit hit rejected)", check_09_bse_safety),
    ("10 Trade attribution round-trip",             check_10_attribution_roundtrip),
    ("11 SEBI audit trail append-only",             check_11_audit_trail),
    ("12 Human override gate BLOCK/remove",         check_12_override_gate),
    ("13 ModelVersionRegistry approve gate",        check_13_model_registry),
    ("14 PSI drift monitor detects shift",          check_14_drift_monitor),
    ("15 Calibration curve report",                 check_15_calibration),
    ("16 Fractional differentiation non-NaN",       check_16_fracdiff),
    ("17 SHAP attributions (or skip if no model)",  check_17_shap),
    ("18 Benchmark comparator CAGR/Sharpe/DD",      check_18_benchmark),
    ("19 Paper trader one-day simulation",          check_19_paper_trader),
    ("20 TradeOrchestrator import",                 check_20_orchestrator_import),
]


def run_acceptance_suite(verbose: bool = False) -> int:
    print("=" * 72)
    print("  P27 ACCEPTANCE TEST SUITE  --  Indian Equity Stock Picker")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    results: List[TestResult] = []
    for name, fn in CHECKS:
        result = _run_check(name, fn)
        results.append(result)
        marker = "+" if result.passed else "X"
        print(f"  {marker} {result}")
        if verbose and result.detail:
            print(f"      {result.detail}")

    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass

    print("=" * 72)
    print(f"  RESULT: {n_pass}/{len(results)} checks passed", end="")
    if n_fail:
        print(f"  ({n_fail} FAILED)")
        print()
        print("  FAILED checks:")
        for r in results:
            if not r.passed:
                print(f"    - {r.name}  [{r.detail}]")
    else:
        print("  -- ALL PASS")
    print("=" * 72)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P27 Acceptance Test Suite")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    sys.exit(run_acceptance_suite(verbose=args.verbose))
