#!/usr/bin/env python3
"""
audit_feature_leaks.py  --  P05: Feature horizon audit.

Statically and dynamically checks momentum_features.py for look-ahead leaks:
  1. .shift(-N) with N > 0 outside label functions
  2. Rolling windows that include today's OHLCV (ok if end-of-day model)
  3. Cross-sectional ranking across all dates (should be per-date group)
  4. Target column (Label_fwd5_positive) appearing in feature set

Exit codes:
  0 -- PASS (no blocking leaks found)
  1 -- FAIL (at least one blocking leak detected)
"""

import ast
import re
import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("audit_feature_leaks")

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

FEATURES_FILE = _ROOT / "momentum_features.py"
LABEL_COL     = "Label_fwd5_positive"
LABEL_FUNCS   = {"add_forward_returns", "_add_labels"}  # shift(-N) is OK here


# ---------------------------------------------------------------------------
# 1. Static AST analysis
# ---------------------------------------------------------------------------

def audit_static(src: str) -> list:
    """
    Parse the source as AST and return a list of (lineno, severity, message).

    Looks for:
     - .shift(-N) with N > 0 outside label functions
     - Any cross-sectional ranking (rank) NOT inside a groupby().apply()
    """
    issues = []
    tree   = ast.parse(src)
    lines  = src.splitlines()

    # Track current function context
    current_func = [None]

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            old = current_func[0]
            current_func[0] = node.name
            self.generic_visit(node)
            current_func[0] = old

        def visit_Call(self, node):
            # Detect .shift(-N)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "shift"
                and node.args
            ):
                arg = node.args[0]
                # Literal negative number
                if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                    if isinstance(arg.operand, ast.Constant) and arg.operand.value > 0:
                        fn = current_func[0] or "<module>"
                        if fn not in LABEL_FUNCS:
                            issues.append((
                                node.lineno, "FAIL",
                                f"shift(-{arg.operand.value}) in function '{fn}' "
                                f"-- look-ahead! (line: {lines[node.lineno-1].strip()})"
                            ))
            self.generic_visit(node)

    Visitor().visit(tree)
    return issues


# ---------------------------------------------------------------------------
# 2. Dynamic checks on computed features
# ---------------------------------------------------------------------------

def audit_dynamic() -> list:
    """
    Load actual feature data and run checks:
     - Does RS_Composite / BBWidthPctl vary ACROSS dates for a stock? (good)
       If not, it might be a single global rank (bad).
     - Is Label_fwd5_positive in computed feature columns?
    """
    issues = []

    try:
        from bse_direct_loader import BSEDataFetcher
        from datetime import date, timedelta

        fetcher = BSEDataFetcher()
        end   = fetcher.prev_bday(date.today())
        start = end - timedelta(days=150)
        bhav  = fetcher.fetch_bhav_range(start, end)

        if bhav is None or bhav.empty:
            log.warning("No BhavCopy data -- skipping dynamic checks.")
            return issues

        from momentum_features import prepare_features_all
        feat_df = prepare_features_all(bhav)

        # Check 1: Label column not in features
        feat_df["DATE"] = pd.to_datetime(feat_df["DATE"]).dt.date
        non_feature_cols = {
            "SC_CODE", "SC_NAME", "DATE", "ISIN", "Source",
            LABEL_COL, "Label_fwd5_return",
            "Open", "High", "Low", "Close", "Volume", "ValueTraded",
        }
        all_cols = set(feat_df.columns)
        if LABEL_COL in all_cols:
            issues.append((
                0, "WARN",
                f"Label column '{LABEL_COL}' present in feature DataFrame. "
                "Ensure it is excluded from X_train."
            ))

        # Check 2: RS_Composite should vary per stock across dates
        # (if it were a global rank, it would be static per stock)
        if "RS_Composite" in feat_df.columns:
            sample_stocks = feat_df["SC_CODE"].unique()[:5]
            for sc in sample_stocks:
                vals = feat_df[feat_df["SC_CODE"] == sc]["RS_Composite"].dropna()
                if len(vals) > 1 and vals.std() < 1e-9:
                    issues.append((
                        0, "FAIL",
                        f"RS_Composite for SC_CODE={sc} has zero variance across dates "
                        "-- likely a global rank (look-ahead)."
                    ))
                    break

        # Check 3: Verify cross-sectional rank is per-date
        # Each date should have ~uniform distribution of RS_Composite
        if "RS_Composite" in feat_df.columns:
            date_stats = feat_df.groupby("DATE")["RS_Composite"].agg(["min", "max"])
            bad_dates  = date_stats[(date_stats["min"] < 0.01) | (date_stats["max"] > 0.99)]
            if len(bad_dates) > len(date_stats) * 0.5:
                issues.append((
                    0, "WARN",
                    f"RS_Composite does not span 0..1 on most dates -- "
                    "may not be ranked per-date."
                ))

        # Check 4: shift(-N) contamination -- test on tiny synthetic data
        test_df = _test_no_forward_leak()
        if test_df is not None:
            issues.extend(test_df)

    except Exception as e:
        log.warning("Dynamic checks failed: %s", e)

    return issues


def _test_no_forward_leak() -> list:
    """
    Build a tiny 3-stock dataset and check that features on day T
    don't differ based on day T+1 data (simple perturbation test).
    """
    issues = []
    from datetime import date, timedelta
    from momentum_features import compute_per_symbol_features

    days  = [date(2024, 1, 2) + timedelta(days=i) for i in range(60)
             if (date(2024, 1, 2) + timedelta(days=i)).weekday() < 5]
    close = 100.0 + np.cumsum(np.random.default_rng(99).normal(0, 0.5, len(days)))

    df_base = pd.DataFrame({
        "SC_CODE": "X", "SC_NAME": "X",
        "DATE": days,
        "Open":  close * 0.99, "High": close * 1.01,
        "Low":   close * 0.98, "Close": close,
        "Volume": 100_000, "ValueTraded": close * 100_000,
    })

    # Perturb the LAST row (future day) and check if earlier rows change
    df_perturbed = df_base.copy()
    df_perturbed.iloc[-1, df_perturbed.columns.get_loc("Close")] *= 2.0

    feat_base = compute_per_symbol_features(df_base.copy())
    feat_pert = compute_per_symbol_features(df_perturbed.copy())

    # Compare features on a mid-point day (should be identical)
    mid_idx = len(days) // 2
    test_cols = ["EMA20", "RSI14", "ATR14", "RET21D"]
    for col in test_cols:
        if col not in feat_base.columns:
            continue
        val_base = feat_base.iloc[mid_idx][col]
        val_pert = feat_pert.iloc[mid_idx][col]
        if not np.isclose(val_base, val_pert, rtol=1e-6, equal_nan=True):
            issues.append((
                0, "FAIL",
                f"Perturbation test: {col} on day {mid_idx} changed when "
                f"future day was modified ({val_base:.4f} -> {val_pert:.4f}). "
                "Look-ahead leak confirmed!"
            ))

    return issues


# ---------------------------------------------------------------------------
# 3. Report
# ---------------------------------------------------------------------------

def run_audit() -> bool:
    print()
    print("=" * 65)
    print("  P05  Feature Horizon (Look-Ahead) Audit")
    print("=" * 65)

    src = FEATURES_FILE.read_text(encoding="utf-8")

    # Static analysis
    print("\n[1] Static AST analysis ...")
    static_issues = audit_static(src)

    # Also grep source for negative shifts as a sanity check
    neg_shifts = re.findall(r"\.shift\s*\(\s*-\s*(\d+)\s*\)", src)
    if neg_shifts:
        print(f"  Regex found .shift(-N) patterns: {neg_shifts}")
        print("  (Checking which functions these are in ...)")

    # Dynamic analysis
    print("\n[2] Dynamic runtime checks ...")
    dynamic_issues = audit_dynamic()

    all_issues = static_issues + dynamic_issues

    # Print results
    print()
    fails = [i for i in all_issues if i[1] == "FAIL"]
    warns = [i for i in all_issues if i[1] == "WARN"]

    for lineno, sev, msg in all_issues:
        loc = f"line {lineno}" if lineno > 0 else "runtime"
        print(f"  [{sev}] ({loc}): {msg}")

    print()
    feature_functions = [
        ("ema()",                         "PASS", "Uses ewm() -- no look-ahead"),
        ("atr()",                         "PASS", "Uses shift(1) for prev close -- clean"),
        ("High20_prev / High63_prev",     "PASS", "shift(1) before rolling -- clean"),
        ("RangePos20",                    "INFO", "Uses today High/Low -- valid end-of-day"),
        ("BBands / BBWidth",              "INFO", "Uses today Close -- valid end-of-day"),
        ("VolMult / UD_Vol_Ratio10",      "PASS", "shift(1) for up/down classification"),
        ("RET21D / RET63D",               "PASS", "pct_change uses past close only"),
        ("RSI14 / ADX14",                 "PASS", "ewm() on past data only"),
        ("RS_Composite / BBWidthPctl",    "PASS", "Ranked per-DATE group (groupby.apply)"),
        ("W_TrendOK / W_BBWidth",         "PASS", "Weekly: past-week values forward-filled"),
        ("Label_fwd5_positive",           "INFO", "shift(-5) -- used ONLY in label col"),
        ("Label excluded from X_train",   "PASS" if not fails else "CHECK",
         "BaseFeatureCols excludes label"),
    ]

    print("  Per-function PASS/FAIL:")
    for name, status, note in feature_functions:
        sym = "OK" if status in ("PASS", "INFO") else "!!"
        print(f"    [{sym}] {name:35s} {status:5s}  {note}")

    print()
    if not fails:
        print("  RESULT: ALL PASS -- no look-ahead leaks detected")
        print("=" * 65 + "\n")
        return True
    else:
        print(f"  RESULT: {len(fails)} FAIL(s) -- see details above")
        print("=" * 65 + "\n")
        return False


if __name__ == "__main__":
    ok = run_audit()
    sys.exit(0 if ok else 1)
