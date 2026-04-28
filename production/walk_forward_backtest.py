#!/usr/bin/env python3
"""
walk_forward_backtest.py  --  P07: Proper walk-forward validation.

Replaces the naive single-model backtest with an expanding-window
walk-forward that is truly out-of-sample:

  For each 3-month test block B (in chronological order):
    - Training data  : all BhavCopy rows with DATE < (block_start - embargo)
    - Embargo        : 5 trading days gap (avoids leaking T+5 labels into features)
    - Test data      : all trading days within block B
    - Model          : LightGBM trained ONLY on the training slice
    - Predictions    : generated on each day in block B using features
                       computed from data up to (and including) that day
    - Outcomes       : actual 5-session forward close-to-close return

Per-block statistics saved:
    win_rate, win_rate_net, avg_return_gross, avg_return_net,
    sharpe_approx, n_picks, n_days, precision_at_top20pct

Outputs (all written to stock_picker_data/results/):
    wf_block_stats_YYYYMMDD.csv   -- one row per block
    wf_detail_YYYYMMDD.csv        -- one row per pick
    wf_backtest_YYYYMMDD.png      -- 2x2 chart

CLI:
    python walk_forward_backtest.py
    python walk_forward_backtest.py --months-per-block 3 --embargo 5
    python walk_forward_backtest.py --min-train-months 9 --no-charts
    python walk_forward_backtest.py --lookback 730 --threshold 0.62

Programmatic:
    from walk_forward_backtest import run_walk_forward
    result = run_walk_forward(bhav_df, end_date="2026-04-25")
"""

import argparse
import json
import logging
import sys
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent          # …/production/
_ROOT = _HERE.parent                              # …/stockpicker/  (project root)
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))               # makes `production.*` importable
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))               # makes sibling modules importable

logger = logging.getLogger(__name__)

RESULTS_DIR = _ROOT / "stock_picker_data" / "results"
MODELS_DIR  = _ROOT / "stock_picker_data" / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trading_days_after(all_dates_sorted: list, from_date, n: int):
    """Return the date that is n trading sessions after from_date."""
    future = [d for d in all_dates_sorted if d > from_date]
    if len(future) < n:
        return None
    return future[n - 1]


def _forward_close(bhav_df: pd.DataFrame, from_date, n_sessions: int = 5) -> pd.Series:
    """SC_CODE -> close price n_sessions trading days after from_date."""
    all_dates = sorted(bhav_df["DATE"].unique())
    future = [d for d in all_dates if d > from_date]
    if len(future) < n_sessions:
        return pd.Series(dtype=float)
    target = future[n_sessions - 1]
    day = bhav_df[bhav_df["DATE"] == target][["SC_CODE", "Close"]]
    return day.set_index("SC_CODE")["Close"]


def _split_into_blocks(all_dates: list, months_per_block: int) -> list:
    """
    Split the sorted list of trading dates into consecutive blocks of
    approximately months_per_block calendar months each.

    Returns list of (block_start_date, block_end_date) tuples.
    """
    if not all_dates:
        return []
    df_idx = pd.DatetimeIndex(all_dates)
    # Quarterly (or custom) period starts
    start_ts = pd.Timestamp(all_dates[0])
    end_ts   = pd.Timestamp(all_dates[-1])

    blocks = []
    period_start = start_ts.to_period(f"{months_per_block}M").start_time
    while period_start <= end_ts:
        period_end = (period_start + pd.DateOffset(months=months_per_block)
                      - pd.Timedelta(days=1))
        # Find actual trading days within this calendar block
        mask  = (df_idx >= period_start) & (df_idx <= period_end)
        block_dates = sorted(df_idx[mask].date.tolist())
        if block_dates:
            blocks.append((block_dates[0], block_dates[-1]))
        period_start = period_end + pd.Timedelta(days=1)
    return blocks


def _train_block_model(
    bhav_df: pd.DataFrame,
    train_end_date,
    n_cv_splits: int = 3,
    boost_rounds: int = 300,
    threshold: float = 0.62,
):
    """
    Train a LightGBM model using only bhav_df rows with DATE <= train_end_date.

    Returns (model, feature_cols) or (None, []) on failure.
    """
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score

    # Slice training data
    train_bhav = bhav_df[bhav_df["DATE"] <= train_end_date].copy()
    if train_bhav.empty:
        logger.warning("No training data up to %s", train_end_date)
        return None, []

    # Build features
    try:
        from momentum_features import prepare_features_all, add_forward_returns, add_fracdiff_features
        feat_df = prepare_features_all(train_bhav)
        # P41: add fractional differentiation features
        try:
            feat_df = add_fracdiff_features(feat_df)
        except Exception as fe:
            logger.debug("WF block fracdiff skipped: %s", fe)
        # P47: add FII/DII market-wide features
        try:
            from momentum_features import FIIDIIFeatures
            start_str = str(train_bhav["DATE"].min())[:10]
            end_str   = str(train_bhav["DATE"].max())[:10]
            feat_df = FIIDIIFeatures().merge_into_features(feat_df, start_str, end_str)
        except Exception as fe:
            logger.debug("WF block FII/DII skipped: %s", fe)
        feat_df = add_forward_returns(feat_df, periods=[5])
    except Exception as e:
        logger.warning("Feature build failed for block ending %s: %s", train_end_date, e)
        return None, []

    label_col = "Label_fwd5_positive"
    if label_col not in feat_df.columns:
        logger.warning("Label column missing in block ending %s", train_end_date)
        return None, []

    feat_df = feat_df.dropna(subset=[label_col])
    if len(feat_df) < 500:
        logger.warning(
            "Insufficient labelled samples (%d) for block ending %s — skipping.",
            len(feat_df), train_end_date,
        )
        return None, []

    # P42: ALWAYS load canonical list from feature_cols.json (preferred) or feature_names.json
    canonical_path = MODELS_DIR / "feature_cols.json"
    fallback_path  = MODELS_DIR / "feature_names.json"
    if canonical_path.exists():
        base_cols = json.loads(canonical_path.read_text())
    elif fallback_path.exists():
        base_cols = json.loads(fallback_path.read_text())
    else:
        try:
            from production.signal_generator import BASE_FEATURE_COLS
            base_cols = BASE_FEATURE_COLS
        except Exception:
            base_cols = []

    # Leakage guard (applied to canonical list)
    FORBIDDEN = {label_col, "Label_fwd5_return", "forward_return_5d", "_fwd_close"}
    feature_cols = [c for c in base_cols if c not in FORBIDDEN]

    if len(feature_cols) < 10:
        exclude = {"SC_CODE", "SC_NAME", "DATE", label_col, "Label_fwd5_return"}
        feature_cols = [
            c for c in feat_df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]

    # P42: Hard reindex — fill missing with 0 so every block has EXACTLY the
    #      same feature count as main training. This prevents the 31-vs-33
    #      mismatch when FracDiff cols are missing in short WF blocks.
    filled_cols = [c for c in feature_cols if c not in feat_df.columns]
    if filled_cols:
        logger.warning(
            "P42 Block [end=%s]: %d canonical features absent → filled with 0: %s",
            train_end_date, len(filled_cols), filled_cols,
        )
    logger.info("P42 WF BLOCK COLS (%d): %s", len(feature_cols), sorted(feature_cols))

    X = feat_df.reindex(columns=feature_cols, fill_value=0).fillna(0).values
    y = feat_df[label_col].values

    params = {
        "objective":      "binary",
        "metric":         "auc",
        "learning_rate":  0.05,
        "num_leaves":     63,
        "min_child_samples": 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":   5,
        "reg_alpha":      0.1,
        "reg_lambda":     0.1,
        "n_jobs":         -1,
        "verbose":        -1,
    }

    # Cross-validate to find best rounds
    tscv      = TimeSeriesSplit(n_splits=n_cv_splits)
    cv_aucs   = []
    best_rnds = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        m = lgb.train(
            params, dtrain,
            num_boost_round=boost_rounds,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(period=0)],
        )
        cv_aucs.append(roc_auc_score(y_val, m.predict(X_val)))
        best_rnds.append(m.best_iteration or m.num_trees())

    final_rounds = max(50, int(np.median(best_rnds)))
    dtrain_full  = lgb.Dataset(X, label=y)
    final_model  = lgb.train(
        params, dtrain_full,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    logger.info(
        "  Block train [end=%s]: %d samples, %d features, "
        "CV AUC=%.4f, rounds=%d",
        train_end_date, len(X), len(feature_cols),
        float(np.mean(cv_aucs)), final_rounds,
    )
    return final_model, feature_cols


def _predict_block(
    bhav_df: pd.DataFrame,
    model,
    feature_cols: list,
    block_start,
    block_end,
    threshold: float,
    fwd_sessions: int = 5,
) -> pd.DataFrame:
    """
    Generate predictions for every trading day in [block_start, block_end].

    For each day D:
      - Use bhav_df rows where DATE <= D to compute features (no look-ahead)
      - Predict with the pre-trained model
      - Compute 5-session forward return from bhav_df

    Returns a detail DataFrame with one row per (date, stock) pick.
    """
    try:
        from momentum_features import prepare_features_all, add_fracdiff_features, FIIDIIFeatures
    except ImportError:
        logger.error("Cannot import momentum_features — block prediction skipped.")
        return pd.DataFrame()

    # Compute features on the slice up to block_end (for efficiency)
    block_bhav  = bhav_df[bhav_df["DATE"] <= block_end].copy()
    all_feat_df = prepare_features_all(block_bhav)
    # P41: fractional differentiation
    try:
        all_feat_df = add_fracdiff_features(all_feat_df)
    except Exception as _fe:
        logger.debug("Predict block fracdiff skipped: %s", _fe)
    # P47: FII/DII market features
    try:
        start_str = str(block_bhav["DATE"].min())[:10]
        end_str   = str(block_bhav["DATE"].max())[:10]
        all_feat_df = FIIDIIFeatures().merge_into_features(all_feat_df, start_str, end_str)
    except Exception as _fe:
        logger.debug("Predict block FII/DII skipped: %s", _fe)
    all_feat_df["DATE"] = pd.to_datetime(all_feat_df["DATE"]).dt.date

    block_dates = sorted([
        d for d in bhav_df["DATE"].unique()
        if block_start <= d <= block_end
    ])
    all_dates_sorted = sorted(bhav_df["DATE"].unique())

    picks_list = []
    for d in block_dates:
        rows = all_feat_df[all_feat_df["DATE"] == d].copy()
        if rows.empty:
            continue

        X_slice = rows.reindex(columns=feature_cols, fill_value=0).fillna(0)
        try:
            probs = model.predict(X_slice.values)
        except Exception:
            continue

        rows = rows.copy()
        rows["Probability"] = probs
        rows["pred_date"]   = d

        picks = rows[rows["Probability"] >= threshold][
            ["SC_CODE", "SC_NAME", "Close", "Probability", "pred_date"]
        ].copy()
        if picks.empty:
            continue

        # P09 — execution reality checks (volume, circuit, gap, liquidity)
        try:
            from production.execution_checks import apply_execution_checks
            picks, _ec_report = apply_execution_checks(picks, bhav_df, d)
        except Exception as _ec_err:
            logger.debug("P09 execution checks skipped for %s: %s", d, _ec_err)

        if picks.empty:
            continue

        # Forward return
        fwd_close = _forward_close(bhav_df, d, fwd_sessions)
        picks["close_fwd"] = picks["SC_CODE"].map(fwd_close)
        picks = picks.dropna(subset=["close_fwd", "Close"])
        picks = picks[picks["Close"] > 0]
        picks_list.append(picks)

    if not picks_list:
        return pd.DataFrame()

    detail = pd.concat(picks_list, ignore_index=True)
    detail["return_5d"] = (detail["close_fwd"] / detail["Close"]) - 1
    detail["positive"]  = (detail["return_5d"] > 0).astype(int)
    detail["pred_date"] = pd.to_datetime(detail["pred_date"])
    return detail


def _apply_friction(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Add friction_pct, return_5d_net, positive_net columns (P02 reuse)."""
    try:
        from production.friction_model import FrictionModel
        _fm = FrictionModel(instrument_type="equity_delivery")
        pos_vals    = np.clip(detail_df["Close"].values * 500, 10_000, 500_000)
        sample_vals = np.quantile(pos_vals, [0.1, 0.3, 0.5, 0.7, 0.9])
        sample_pcts = np.array([
            _fm.calculate_round_trip_cost(v, exchange="BSE")["total_costs"]["total"] / v
            for v in sample_vals
        ])
        detail_df["friction_pct"]  = np.interp(pos_vals, sample_vals, sample_pcts) + 0.003
    except Exception:
        detail_df["friction_pct"] = 0.005   # 0.5% fallback

    detail_df["return_5d_net"] = detail_df["return_5d"] - detail_df["friction_pct"]
    detail_df["positive_net"]  = (detail_df["return_5d_net"] > 0).astype(int)
    return detail_df


def _assign_exit_type(detail_df: pd.DataFrame, block_end) -> pd.DataFrame:
    """
    P36: Assign exit_type to each trade based on gross return.

    Since the WF backtest only tracks 5-session exit price, exit type is inferred:
    - STOP_HIT      : return < -8%  (hard stop likely triggered intra-period)
    - TRAILING_STOP : -8% <= return < -4%
    - TIME_STOP     : -4% <= return < 0%  (small loss, held full 5 sessions)
    - PERIOD_END    : pred_date within last 5 sessions of block (fwd close is block boundary)
    - REGIME_KILL   : currently undetected in WF; reserved for future live tagging
    """
    detail_df = detail_df.copy()
    r = detail_df["return_5d"].values

    block_end_dt = pd.Timestamp(block_end)
    near_block_end = detail_df["pred_date"] >= (block_end_dt - pd.Timedelta(days=9))

    exit_type = np.where(r < -0.08, "STOP_HIT",
                np.where(r < -0.04, "TRAILING_STOP",
                np.where(near_block_end, "PERIOD_END", "TIME_STOP")))
    detail_df["exit_type"] = exit_type
    return detail_df


def _compute_oos_ece(y_true, y_prob, n_bins: int = 10) -> float:
    """Expected Calibration Error on OOS predictions. P30."""
    if len(y_true) < n_bins:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_acc  = float(np.mean(y_true[mask]))
        bin_conf = float(np.mean(y_prob[mask]))
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return round(ece, 6)


def _block_stats(detail: pd.DataFrame, block_start, block_end) -> dict:
    """Compute per-block summary statistics (P30: includes OOS ECE)."""
    n_picks = len(detail)

    if n_picks == 0:
        return {
            "block_start": str(block_start), "block_end": str(block_end),
            "n_days": 0, "n_picks": 0,
            "win_rate": np.nan, "win_rate_net": np.nan,
            "avg_return_gross": np.nan, "avg_return_net": np.nan,
            "sharpe_approx": np.nan, "precision_top20": np.nan,
            "ece_raw": np.nan,
        }

    n_days = detail["pred_date"].dt.date.nunique()
    r = detail["return_5d"].values
    r_net = detail["return_5d_net"].values

    # Sharpe approximation: mean / std of gross daily returns
    sharpe = float(np.mean(r) / (np.std(r) + 1e-9)) * np.sqrt(252)

    # Precision at top 20% by probability
    q80 = detail["Probability"].quantile(0.80)
    top20 = detail[detail["Probability"] >= q80]
    prec_top20 = float(top20["positive"].mean()) if len(top20) > 0 else np.nan

    # P30: OOS ECE — computed on ALL stocks above threshold, not just winners
    # Use 'positive' (5-day forward return > 0) as true label
    y_true = detail["positive"].values.astype(float)
    y_prob = detail["Probability"].values
    ece_raw = _compute_oos_ece(y_true, y_prob)

    # P33: Wilson confidence interval for gross win rate
    from math import sqrt
    wins = int(detail["positive"].sum())
    p_hat = wins / n_picks
    z = 1.96  # 95% CI
    denom = 1 + z**2 / n_picks
    centre = (p_hat + z**2 / (2 * n_picks)) / denom
    margin = z * sqrt(p_hat * (1 - p_hat) / n_picks + z**2 / (4 * n_picks**2)) / denom
    wr_ci_lo = round(max(0.0, centre - margin), 4)
    wr_ci_hi = round(min(1.0, centre + margin), 4)

    block_num = getattr(detail, "_block_num", "?")
    logger.info(
        "P30 Block [%s -> %s] OOS ECE=%.4f  n=%d",
        block_start, block_end, ece_raw if not np.isnan(ece_raw) else -1, n_picks,
    )
    logger.info(
        "P33 Block [%s -> %s] gross WR=%.1f%%  95%% CI=[%.1f%%, %.1f%%]  n=%d",
        block_start, block_end, p_hat * 100, wr_ci_lo * 100, wr_ci_hi * 100, n_picks,
    )

    if not np.isnan(ece_raw) and ece_raw > 0.08:
        logger.warning(
            "P30 WARNING: Block OOS ECE=%.4f > 0.08 — calibrator may need "
            "re-fitting for this block's training window.", ece_raw
        )

    # P36: per-exit-type median return
    if "exit_type" in detail.columns:
        for etype, grp in detail.groupby("exit_type"):
            logger.info(
                "P36 exit_type=%s  n=%d  median_gross=%.3f%%",
                etype, len(grp), float(grp["return_5d"].median()) * 100,
            )

    return {
        "block_start":      str(block_start),
        "block_end":        str(block_end),
        "n_days":           n_days,
        "n_picks":          n_picks,
        "win_rate":         float(detail["positive"].mean()),
        "win_rate_net":     float(detail["positive_net"].mean()),
        "wr_ci_lo":         wr_ci_lo,
        "wr_ci_hi":         wr_ci_hi,
        "avg_return_gross": float(np.mean(r)),
        "avg_return_net":   float(np.mean(r_net)),
        "sharpe_approx":    sharpe,
        "precision_top20":  prec_top20,
        "ece_raw":          ece_raw,
    }


# ---------------------------------------------------------------------------
# Main walk-forward engine
# ---------------------------------------------------------------------------

def run_walk_forward(
    bhav_df: pd.DataFrame = None,
    end_date: str = None,
    lookback_days: int = 730,
    months_per_block: int = 3,
    embargo_sessions: int = 5,
    min_train_months: int = 9,
    threshold: float = 0.62,
    fwd_sessions: int = 5,
    n_cv_splits: int = 3,
    boost_rounds: int = 300,
    draw_charts: bool = True,
) -> dict:
    """
    Run the full walk-forward backtest.

    Parameters
    ----------
    bhav_df         : Pre-loaded BhavCopy DataFrame.  Loaded from DataLoader
                      if None.
    end_date        : Last date to include (YYYY-MM-DD).  Defaults to today.
    lookback_days   : Calendar days of history to load (if bhav_df is None).
    months_per_block: Test block length in calendar months (default 3).
    embargo_sessions: Trading-day gap between train end and test start (default 5).
    min_train_months: Minimum calendar months of training data required before
                      the first test block (default 9).
    threshold       : Minimum model probability for a pick (default 0.62).
    fwd_sessions    : Forward sessions for outcome measurement (default 5).
    n_cv_splits     : Cross-validation folds for each block training (default 3).
    boost_rounds    : Max LightGBM boost rounds per fold (default 300).
    draw_charts     : Save 2x2 PNG chart (default True).

    Returns
    -------
    dict with keys:
        block_stats  -- list of per-block dicts
        detail_df    -- pd.DataFrame, one row per (date, stock) pick
        summary      -- dict of aggregate statistics
        chart_path   -- str path to PNG (or None)
    """
    today_str = end_date or str(_date.today())
    logger.info("=" * 68)
    logger.info("  WALK-FORWARD BACKTEST   end=%s", today_str)
    logger.info("  block=%dM  embargo=%d days  threshold=%.2f",
                months_per_block, embargo_sessions, threshold)
    logger.info("=" * 68)

    # -- Load data ----------------------------------------------------------
    if bhav_df is None:
        from production.data_loader import DataLoader
        loader  = DataLoader()
        bhav_df = loader.load(lookback_days=lookback_days, end=today_str)

    if bhav_df is None or bhav_df.empty:
        raise RuntimeError("No BhavCopy data loaded.")

    bhav_df["DATE"] = pd.to_datetime(bhav_df["DATE"]).dt.date

    # -- Universe filter ----------------------------------------------------
    try:
        from production.universe_filter import apply_tradability_gate
        ref_date = bhav_df["DATE"].max()
        filtered = apply_tradability_gate(
            bhav_df, ref_date,
            min_value_crore=2.0, min_price=20.0, min_avg_volume=10_000,
            apply_operator_filter=True, apply_ipo_filter=True, min_sessions=90,
        )
        tradable_codes = set(filtered["SC_CODE"].unique())
        bhav_df = bhav_df[bhav_df["SC_CODE"].isin(tradable_codes)].copy()
        logger.info("Universe: %d tradable stocks after all filters.", len(tradable_codes))
    except Exception as e:
        logger.warning("Universe filter skipped: %s", e)

    all_dates   = sorted(bhav_df["DATE"].unique())
    all_blocks  = _split_into_blocks(all_dates, months_per_block)

    # Drop blocks that don't have enough training data before them
    min_train_cutoff = (
        pd.Timestamp(all_dates[0]) + pd.DateOffset(months=min_train_months)
    ).date()

    test_blocks = [
        (bs, be) for bs, be in all_blocks
        if bs > min_train_cutoff
    ]

    # Further drop blocks too close to the end to compute forward returns
    def _has_forward_data(be):
        future = [d for d in all_dates if d > be]
        return len(future) >= fwd_sessions

    test_blocks = [(bs, be) for bs, be in test_blocks if _has_forward_data(be)]

    if not test_blocks:
        logger.warning("No valid test blocks found — data window too short.")
        return {"block_stats": [], "detail_df": pd.DataFrame(),
                "summary": {}, "chart_path": None}

    logger.info("Walk-forward plan: %d test blocks", len(test_blocks))
    for i, (bs, be) in enumerate(test_blocks, 1):
        logger.info("  Block %d: test %s -> %s", i, bs, be)

    # -- Walk-forward loop -------------------------------------------------
    all_block_stats = []
    all_details     = []

    for block_idx, (block_start, block_end) in enumerate(test_blocks, 1):
        logger.info("-" * 60)
        logger.info(
            "Block %d / %d  test: %s -> %s",
            block_idx, len(test_blocks), block_start, block_end,
        )

        # Training cut-off: block_start - embargo (in trading days)
        train_end = None
        before_block = [d for d in all_dates if d < block_start]
        if len(before_block) > embargo_sessions:
            train_end = before_block[-(embargo_sessions + 1)]
        else:
            logger.warning("Block %d: not enough pre-block dates — skipping.", block_idx)
            continue

        logger.info("  Training data: all days up to %s (embargo=%d sessions)",
                    train_end, embargo_sessions)

        # Train block model
        model, feature_cols = _train_block_model(
            bhav_df=bhav_df,
            train_end_date=train_end,
            n_cv_splits=n_cv_splits,
            boost_rounds=boost_rounds,
            threshold=threshold,
        )
        if model is None:
            logger.warning("Block %d: model training failed — skipping.", block_idx)
            all_block_stats.append(
                _block_stats(pd.DataFrame(), block_start, block_end)
            )
            continue

        # Generate OOS predictions for the test block
        detail = _predict_block(
            bhav_df=bhav_df,
            model=model,
            feature_cols=feature_cols,
            block_start=block_start,
            block_end=block_end,
            threshold=threshold,
            fwd_sessions=fwd_sessions,
        )

        if not detail.empty:
            detail = _apply_friction(detail)
            detail = _assign_exit_type(detail, block_end)  # P36
            detail["block"] = block_idx

        stats = _block_stats(detail, block_start, block_end)
        all_block_stats.append(stats)
        logger.info(
            "  Block %d result: n_picks=%d  win_rate=%.1f%%  net_win=%.1f%%  "
            "avg_ret=%.2f%%",
            block_idx,
            stats["n_picks"],
            (stats["win_rate"]  or 0) * 100,
            (stats["win_rate_net"] or 0) * 100,
            (stats["avg_return_gross"] or 0) * 100,
        )
        if not detail.empty:
            all_details.append(detail)

    # -- Aggregate ----------------------------------------------------------
    full_detail = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()
    stats_df    = pd.DataFrame(all_block_stats)

    summary = {}
    if not full_detail.empty:
        summary = {
            "n_blocks":           len(all_block_stats),
            "n_picks_total":      len(full_detail),
            "overall_win_rate":   float(full_detail["positive"].mean()),
            "overall_win_net":    float(full_detail["positive_net"].mean()),
            "avg_return_gross":   float(full_detail["return_5d"].mean()),
            "avg_return_net":     float(full_detail["return_5d_net"].mean()),
            "median_return_gross":float(full_detail["return_5d"].median()),
            "median_return_net":  float(full_detail["return_5d_net"].median()),
            "avg_friction_pct":   float(full_detail["friction_pct"].mean()),
            "blocks_above_50pct": int((stats_df["win_rate"].fillna(0) >= 0.5).sum()),
            "mean_oos_ece":       float(stats_df["ece_raw"].mean(skipna=True)),
            "backtest_start":     str(test_blocks[0][0]),
            "backtest_end":       str(test_blocks[-1][1]),
        }
        # P30: Print per-block OOS ECE summary
        logger.info("P30 OOS ECE per block: %s",
                    {f"B{i+1}": round(float(r), 4) if not np.isnan(float(r)) else None
                     for i, r in enumerate(stats_df["ece_raw"])})
        logger.info("=" * 68)
        logger.info(
            "WALK-FORWARD SUMMARY: %d blocks  |  "
            "gross win rate=%.1f%%  |  net win rate=%.1f%%  |  "
            "blocks>=50%%: %d/%d",
            summary["n_blocks"],
            summary["overall_win_rate"] * 100,
            summary["overall_win_net"]  * 100,
            summary["blocks_above_50pct"],
            summary["n_blocks"],
        )
        logger.info("=" * 68)

    # -- Save CSVs ---------------------------------------------------------
    date_tag = today_str.replace("-", "")
    chart_path = None

    if not stats_df.empty:
        stats_path = RESULTS_DIR / f"wf_block_stats_{date_tag}.csv"
        stats_df.to_csv(stats_path, index=False)
        logger.info("Block stats -> %s", stats_path)

    if not full_detail.empty:
        detail_path = RESULTS_DIR / f"wf_detail_{date_tag}.csv"
        full_detail.to_csv(detail_path, index=False)
        logger.info("Detail     -> %s", detail_path)

    # -- Chart -------------------------------------------------------------
    if draw_charts and not full_detail.empty:
        chart_path = _draw_chart(full_detail, stats_df, today_str, date_tag)

    return {
        "block_stats": all_block_stats,
        "detail_df":   full_detail,
        "summary":     summary,
        "chart_path":  str(chart_path) if chart_path else None,
    }


# ---------------------------------------------------------------------------
# Charting
# ---------------------------------------------------------------------------

def _draw_chart(detail_df, stats_df, today_str: str, date_tag: str) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        logger.warning("matplotlib not available -- chart skipped.")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Walk-Forward Backtest  |  {today_str}  |  {len(stats_df)} OOS blocks",
        fontsize=13, fontweight="bold",
    )

    # 1. Win rate per block (bar chart) ------------------------------------
    ax = axes[0, 0]
    x  = range(len(stats_df))
    wr = (stats_df["win_rate"].fillna(0) * 100).tolist()
    colors = ["seagreen" if w >= 50 else "tomato" for w in wr]
    ax.bar(x, wr, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(50, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("Win Rate per OOS Block (gross)", fontsize=11)
    ax.set_ylabel("Win Rate (%)")
    ax.set_xlabel("Block #")
    labels = [f"B{i+1}\n{r['block_start'][:7]}" for i, r in stats_df.iterrows()]
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    # 2. Net win rate per block ------------------------------------------
    ax = axes[0, 1]
    wr_net = (stats_df["win_rate_net"].fillna(0) * 100).tolist()
    colors2 = ["steelblue" if w >= 50 else "salmon" for w in wr_net]
    ax.bar(x, wr_net, color=colors2, edgecolor="white", linewidth=0.5)
    ax.axhline(50, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("Win Rate per OOS Block (net of friction)", fontsize=11)
    ax.set_ylabel("Win Rate (%)")
    ax.set_xlabel("Block #")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    # 3. Return distribution -----------------------------------------------
    ax = axes[1, 0]
    gross = detail_df["return_5d"] * 100
    net   = detail_df["return_5d_net"] * 100
    ax.hist(gross, bins=60, color="steelblue", alpha=0.5, label="Gross", edgecolor="none")
    ax.hist(net,   bins=60, color="darkorange",alpha=0.5, label="Net",   edgecolor="none")
    ax.axvline(0,            color="black", linestyle="--", linewidth=1)
    ax.axvline(gross.mean(), color="steelblue",   linestyle="-", linewidth=1.2,
               label=f"Gross mean {gross.mean():+.2f}%")
    ax.axvline(net.mean(),   color="darkorange",  linestyle="-", linewidth=1.2,
               label=f"Net mean {net.mean():+.2f}%")
    ax.set_title("Distribution of 5-Day Returns  (all OOS picks)", fontsize=11)
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 4. Cumulative picks count & avg return per block --------------------
    ax = axes[1, 1]
    avg_ret = (stats_df["avg_return_gross"].fillna(0) * 100).tolist()
    avg_net = (stats_df["avg_return_net"].fillna(0) * 100).tolist()
    ax.plot(list(x), avg_ret, "o-", color="steelblue",  label="Avg gross return",
            linewidth=1.5, markersize=5)
    ax.plot(list(x), avg_net, "s-", color="darkorange", label="Avg net return",
            linewidth=1.5, markersize=5)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("Average 5-Day Return per OOS Block", fontsize=11)
    ax.set_ylabel("Avg Return (%)")
    ax.set_xlabel("Block #")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    chart_path = RESULTS_DIR / f"wf_backtest_{date_tag}.png"
    plt.savefig(chart_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Chart saved -> %s", chart_path)
    return chart_path


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_walk_forward_summary(summary: dict, block_stats: list):
    if not summary:
        print("\n  Walk-forward produced no results.\n")
        return
    w = 68
    print("\n" + "=" * w)
    print(f"  WALK-FORWARD BACKTEST SUMMARY")
    print(f"  Period: {summary.get('backtest_start')} -> {summary.get('backtest_end')}")
    print("=" * w)
    print(f"  OOS blocks tested          : {summary.get('n_blocks', 0)}")
    print(f"  Total OOS picks            : {summary.get('n_picks_total', 0):,}")
    print(f"  Overall gross win rate     : {summary.get('overall_win_rate', 0):.1%}")
    print(f"  Overall net win rate       : {summary.get('overall_win_net', 0):.1%}")
    print(f"  Avg 5-day return (gross)   : {summary.get('avg_return_gross', 0):+.2%}")
    print(f"  Avg 5-day return (net)     : {summary.get('avg_return_net', 0):+.2%}")
    print(f"  Median return (net)        : {summary.get('median_return_net', 0):+.2%}")
    print(f"  Avg friction per trade     : {summary.get('avg_friction_pct', 0):.3%}")
    print(f"  Blocks with win rate >= 50%: "
          f"{summary.get('blocks_above_50pct', 0)} / {summary.get('n_blocks', 0)}")
    print()
    print(f"  {'Block':<6} {'Period':<24} {'N Picks':>7} {'WR%':>6} {'Net WR%':>8} {'Avg Ret%':>9}")
    print("  " + "-" * (w - 2))
    for i, b in enumerate(block_stats, 1):
        wr  = (b.get("win_rate") or 0) * 100
        nwr = (b.get("win_rate_net") or 0) * 100
        ar  = (b.get("avg_return_gross") or 0) * 100
        print(f"  {i:<6} {b['block_start']} -> {b['block_end']}  "
              f"{b['n_picks']:>7}  {wr:>5.1f}%  {nwr:>7.1f}%  {ar:>8.2f}%")
    print("=" * w + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser(
        description="P07 Walk-Forward Backtest — proper OOS validation"
    )
    p.add_argument("--date",              type=str,   default=str(_date.today()),
                   help="End date YYYY-MM-DD (default today)")
    p.add_argument("--lookback",          type=int,   default=730,
                   help="Calendar days of history to load (default 730)")
    p.add_argument("--months-per-block",  type=int,   default=3,
                   help="Test block length in months (default 3)")
    p.add_argument("--embargo",           type=int,   default=5,
                   help="Trading-day embargo between train end and test start (default 5)")
    p.add_argument("--min-train-months",  type=int,   default=9,
                   help="Minimum training months before first test block (default 9)")
    p.add_argument("--threshold",         type=float, default=0.62,
                   help="Probability threshold (default 0.62)")
    p.add_argument("--boost-rounds",      type=int,   default=300,
                   help="Max LightGBM boost rounds per block (default 300)")
    p.add_argument("--fwd-sessions",      type=int,   default=5,
                   help="Forward sessions for outcome measurement (default 5)")
    p.add_argument("--compare-sessions",  action="store_true",
                   help="Run backtest for fwd_sessions=3,4,5 and print comparison table")
    p.add_argument("--no-charts",         action="store_true",
                   help="Skip chart generation")
    args = p.parse_args()

    if args.compare_sessions:
        _compare_time_stops(args)
        sys.exit(0)

    result = run_walk_forward(
        end_date=args.date,
        lookback_days=args.lookback,
        months_per_block=args.months_per_block,
        embargo_sessions=args.embargo,
        min_train_months=args.min_train_months,
        threshold=args.threshold,
        boost_rounds=args.boost_rounds,
        fwd_sessions=args.fwd_sessions,
        draw_charts=not args.no_charts,
    )
    print_walk_forward_summary(result["summary"], result["block_stats"])
    if result.get("chart_path"):
        print(f"  Chart -> {result['chart_path']}\n")


def _compare_time_stops(args) -> None:
    """
    Run walk-forward backtest for fwd_sessions in [3, 4, 5] and print a
    side-by-side comparison table.  Helps quantify whether shorter holding
    periods improve net win rate after friction.
    """
    rows = []
    for n in [3, 4, 5]:
        print(f"\n{'='*60}")
        print(f"  Running fwd_sessions={n} ...")
        print(f"{'='*60}")
        res = run_walk_forward(
            end_date=args.date,
            lookback_days=args.lookback,
            months_per_block=args.months_per_block,
            embargo_sessions=args.embargo,
            min_train_months=args.min_train_months,
            threshold=args.threshold,
            boost_rounds=args.boost_rounds,
            fwd_sessions=n,
            draw_charts=False,
        )
        s = res.get("summary", {})
        rows.append({
            "fwd_sessions":   n,
            "n_picks":        s.get("total_picks", 0),
            "gross_wr":       s.get("overall_win_rate", 0),
            "net_wr":         s.get("net_win_rate", 0),
            "avg_net_ret":    s.get("avg_return_net", 0),
            "median_net_ret": s.get("median_return_net", 0),
        })

    print("\n" + "="*72)
    print("  TIME STOP COMPARISON  (lower fwd_sessions = shorter hold)")
    print("="*72)
    print(f"  {'Sessions':>8}  {'Picks':>6}  {'Gross WR':>9}  {'Net WR':>9}  "
          f"{'Avg Net':>9}  {'Median Net':>10}")
    print(f"  {'-'*8}  {'-'*6}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*10}")
    for r in rows:
        print(f"  {r['fwd_sessions']:>8}  {r['n_picks']:>6}  "
              f"{r['gross_wr']*100:>8.1f}%  {r['net_wr']*100:>8.1f}%  "
              f"{r['avg_net_ret']*100:>8.2f}%  {r['median_net_ret']*100:>9.2f}%")
    print("="*72)


if __name__ == "__main__":
    main()
