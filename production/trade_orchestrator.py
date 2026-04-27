"""
TradeOrchestrator — complete 9-layer production pipeline.

Single entry point:

    python production/trade_orchestrator.py              # run today
    python production/trade_orchestrator.py --capital 500000

Or from Python:

    from production.trade_orchestrator import TradeOrchestrator
    result = TradeOrchestrator().run_daily()
    picks  = result["picks"]   # pd.DataFrame — write to broker / manual execution

Output CSV columns (written to stock_picker_data/results/picks_YYYYMMDD.csv):
    SC_CODE, SC_NAME, Close, Probability, Probability_Raw,
    ATR14, Stop_Loss_Price, Trailing_Stop_ATR,
    Shares, Position_Value, Risk_Amount, Risk_Pct,
    Friction_Pct, Net_Expected_Return,
    Sector, Regime, Signal_Threshold, Rank,
    RSI14, ADX14, VolMult, RS_Composite, DistTo52W,
    Prediction_Date, Rationale (SHAP text)
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class OrchestratorConfig:
    def __init__(self, base_dir: str = "stock_picker_data"):
        d = Path(base_dir)
        self.base_dir       = d
        self.models_dir     = d / "models"
        self.cache_dir      = d / "cache"
        self.results_dir    = d / "results"
        self.audit_dir      = d / "audit"
        self.shap_log_dir   = d / "shap_logs"
        self.alert_log      = d / "alerts.log"
        for p in [self.models_dir, self.cache_dir / "bse",
                  self.results_dir, self.audit_dir, self.shap_log_dir]:
            p.mkdir(parents=True, exist_ok=True)

        # Capital & sizing
        self.total_capital          = 1_000_000
        self.risk_pct_per_trade     = 0.01        # 1% capital at risk per trade
        self.max_position_pct       = 0.10
        self.max_positions          = 15
        self.max_capital_deployed   = 0.70
        self.max_sector_positions   = 2
        self.max_correlation        = 0.70

        # Universe gate
        self.min_value_crore        = 2.0
        self.min_price              = 20.0
        self.min_avg_volume         = 10_000
        self.lookback_days          = 730

        # Signal
        self.base_threshold         = 0.62
        self.min_threshold          = 0.52

        # Exit engine
        self.initial_stop_atr       = 2.5
        self.trailing_stop_atr      = 1.5
        self.time_stop_sessions     = 5

        # Risk controls
        self.drawdown_warning       = -0.05
        self.drawdown_halt          = -0.10

        # Event blackout
        self.blackout_days_before   = 2
        self.blackout_days_after    = 2

        self.instrument_type        = "equity_delivery"
        self.exchange               = "BSE"


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TradeOrchestrator:
    """
    Runs all 9 production layers in sequence and writes the daily picks CSV.
    """

    def __init__(self, config: OrchestratorConfig = None):
        self.cfg = config or OrchestratorConfig()

    # ====================================================================
    # PUBLIC ENTRY POINT
    # ====================================================================

    def run_daily(
        self,
        reference_date: str = None,
        total_capital:  float = None,
        bhav_df:        pd.DataFrame = None,
    ) -> dict:
        """
        Execute the full pipeline for one trading day.

        Args:
            reference_date: 'YYYY-MM-DD'; defaults to today.
            total_capital:  Override config capital.
            bhav_df:        Pre-loaded BhavCopy (fetched automatically if None).

        Returns dict with keys:
            picks        — pd.DataFrame of approved trades (empty if SKIP_DAY)
            action       — 'TRADE' | 'SKIP_DAY' | 'REDUCED_SIZE'
            regime       — dict (label, threshold, is_tradeable, …)
            circuit_state— 'NORMAL' | 'WARNING' | 'HALTED'
            explanations — list of SHAP dicts
            output_csv   — path to written CSV (None if no picks)
        """
        ref_date = reference_date or date.today().strftime("%Y-%m-%d")
        if total_capital:
            self.cfg.total_capital = total_capital

        _sep = "=" * 68
        logger.info(_sep)
        logger.info("  TRADE ORCHESTRATOR  —  %s  —  capital Rs.%s",
                    ref_date, f"{self.cfg.total_capital:,}")
        logger.info(_sep)

        result = dict(
            reference_date=ref_date,
            picks=pd.DataFrame(),
            action="SKIP_DAY",
            regime={},
            circuit_state="UNKNOWN",
            explanations=[],
            output_csv=None,
        )

        # -- PRE-CHECK: Kill switch ------------------------------------
        if self._kill_switch_active():
            logger.warning("Kill switch ACTIVE — no trades today.")
            self._audit("SYSTEM_STOP", reason="kill_switch")
            try:
                from production.alerting import alert_kill_switch_activated
                alert_kill_switch_activated("Kill switch file present at run start")
            except Exception:
                pass
            self._write_daily_summary(result, ref_date)
            return result

        # -- PRE-CHECK: Circuit breaker --------------------------------
        cb_state, size_mult = self._check_circuit_breaker()
        result["circuit_state"] = cb_state
        if cb_state == "HALTED":
            logger.error("Circuit breaker HALTED — no new trades.")
            self._audit("RISK_CHECK", circuit_breaker="HALTED")
            try:
                from production.alerting import alert_circuit_breaker_halted
                alert_circuit_breaker_halted(self.cfg.drawdown_halt)
            except Exception:
                pass
            self._write_daily_summary(result, ref_date)
            return result
        if cb_state == "WARNING":
            logger.warning("Circuit breaker WARNING — positions halved.")
            result["action"] = "REDUCED_SIZE"
            try:
                from production.alerting import alert_circuit_breaker_warning
                alert_circuit_breaker_warning(self.cfg.drawdown_warning)
            except Exception:
                pass

        # -- L1: Data Integrity ----------------------------------------
        logger.info("-- L1  Data Integrity ----------------------------------")
        bhav_df = self._l1_data(bhav_df, ref_date)
        if bhav_df is None:
            logger.error("L1: No data — SKIP_DAY.")
            self._write_daily_summary(result, ref_date)
            return result

        # -- L2: Universe Filter ---------------------------------------
        logger.info("-- L2  Universe Filter ---------------------------------")
        bhav_df, tradable_codes = self._l2_universe(bhav_df, ref_date)
        if not tradable_codes:
            logger.error("L2: Zero tradable stocks — SKIP_DAY.")
            self._write_daily_summary(result, ref_date)
            return result
        logger.info("L2: %d tradable stocks.", len(tradable_codes))

        # -- L3: Market Regime -----------------------------------------
        logger.info("-- L3  Market Regime & Events --------------------------")
        regime, threshold = self._l3_regime(ref_date)
        result["regime"] = regime
        if not regime.get("is_tradeable", True):
            logger.warning("L3: BEAR/SKIP_DAY regime — SKIP_DAY.")
            result["action"] = "SKIP_DAY"
            self._write_daily_summary(result, ref_date)
            return result
        logger.info("L3: regime=%s  threshold=%.2f", regime.get("regime"), threshold)

        # -- L4+L5: Signal Generation ----------------------------------
        logger.info("-- L4+L5  Feature Engineering & Signal Generation -------")
        picks_df, feature_df, explanations, shap_df = self._l45_signal(
            bhav_df, tradable_codes, ref_date, threshold
        )
        result["explanations"] = explanations
        if picks_df.empty:
            logger.info("L5: No picks above threshold %.2f.", threshold)
            result["action"] = "SKIP_DAY"
            self._write_daily_summary(result, ref_date)
            return result
        logger.info("L5: %d raw picks.", len(picks_df))

        # -- L3b: Event calendar blackout ------------------------------
        picks_df = self._l3b_events(picks_df, ref_date)
        if picks_df.empty:
            logger.info("L3b: All picks in event blackout — SKIP_DAY.")
            result["action"] = "SKIP_DAY"
            self._write_daily_summary(result, ref_date)
            return result

        # -- L6: Portfolio Construction --------------------------------
        logger.info("-- L6  Portfolio Construction --------------------------")
        picks_df = self._l6_portfolio(picks_df, bhav_df)
        if picks_df.empty:
            result["action"] = "SKIP_DAY"
            self._write_daily_summary(result, ref_date)
            return result
        logger.info("L6: %d picks after portfolio filters.", len(picks_df))

        # -- L7: Execution & Sizing ------------------------------------
        logger.info("-- L7  Execution & Sizing ------------------------------")
        picks_df = self._l7_execution(picks_df, bhav_df, size_mult, regime)

        # -- L8: Annotate & validate -----------------------------------
        picks_df = self._annotate(picks_df, regime, ref_date, explanations, shap_df)

        # -- P49: Risk analytics (VaR + stress) ------------------------
        logger.info("-- P49 Risk Analytics ----------------------------------")
        var_report = self._p49_risk(picks_df)
        result["var_report"] = var_report

        # -- L9: Compliance & audit ------------------------------------
        logger.info("-- L9  Compliance & Audit ------------------------------")
        self._l9_compliance(picks_df, ref_date)

        # -- Write CSV -------------------------------------------------
        csv_path = self._write_csv(picks_df, ref_date)
        self._print_summary(picks_df, ref_date)

        # P50: Alert picks ready
        try:
            from production.alerting import alert_picks_ready
            alert_picks_ready(len(picks_df), ref_date, str(csv_path))
        except Exception:
            pass

        final_action = result["action"] if result["action"] == "REDUCED_SIZE" else "TRADE"
        result.update(picks=picks_df, action=final_action, output_csv=str(csv_path))
        logger.info(_sep)
        logger.info("  DONE  action=%s  picks=%d  csv=%s", final_action, len(picks_df), csv_path)
        logger.info(_sep)
        self._write_daily_summary(result, ref_date)
        return result

    # ====================================================================
    # LAYER IMPLEMENTATIONS
    # ====================================================================

    # -- L1 ---------------------------------------------------------------
    def _l1_data(self, bhav_df, ref_date) -> Optional[pd.DataFrame]:
        """Fetch, standardise, health-check, corporate-action-adjust, QC."""

        # 1. Fetch if not provided — P06: multi-source failover
        if bhav_df is None:
            try:
                from production.operational_fallback import DataSourceFailover, AlertManager
                failover = DataSourceFailover(
                    cache_dir=str(self.cfg.cache_dir / "bse"),
                    lookback_days=self.cfg.lookback_days,
                    alert_manager=AlertManager(alert_log=str(self.cfg.alert_log)),
                )
                bhav_df = failover.fetch_bhav(ref_date)
            except Exception as e:
                logger.error("L1: DataSourceFailover failed: %s", e)
                bhav_df = None

            if bhav_df is None:
                logger.critical("L1: All data sources exhausted — SKIP_DAY.")
                try:
                    from production.alerting import alert_data_source_failed
                    alert_data_source_failed("BSE BhavCopy (all failover sources)", fallback=None)
                except Exception:
                    pass
                return None
        else:
            # Standardise if caller passed raw data
            try:
                from production.data_loader import DataLoader
                bhav_df = DataLoader().standardise(bhav_df)
            except Exception:
                pass

        if bhav_df is None or bhav_df.empty:
            return None

        # 2. Health check via OperationalFallback
        try:
            from production.operational_fallback import DataSourceHealthChecker
            hc = DataSourceHealthChecker(cache_dir=str(self.cfg.cache_dir / "bse"))
            health = hc.check_bhav_data(bhav_df, ref_date)
            if not health["is_healthy"]:
                logger.error("L1: Data health check failed: %s", health["issues"])
                # Alert but don't hard-stop here; let caller decide
                try:
                    from production.operational_fallback import AlertManager
                    AlertManager(alert_log=str(self.cfg.alert_log)).send(
                        "Data Health Warning",
                        "; ".join(health["issues"]),
                        level="WARNING",
                    )
                except Exception:
                    pass
        except Exception as e:
            logger.debug("L1: Health checker skipped: %s", e)

        # 3. Auto-fetch and apply corporate actions (P28: use BSE package)
        try:
            from production.corporate_actions_fetcher import CorporateActionsFetcher
            from production.data_integrity import CorporateActionAdjuster

            ca_fetcher = CorporateActionsFetcher()
            sc_codes = bhav_df["SC_CODE"].unique().tolist() if bhav_df is not None else []

            # P28: batch-fetch via bse Python package for the full universe
            if sc_codes:
                ca_df = ca_fetcher.fetch_for_universe(sc_codes, cache_date=ref_date)
            else:
                ca_df = ca_fetcher.get(
                    start_date=(pd.to_datetime(ref_date) - pd.Timedelta(days=730)).strftime("%Y-%m-%d"),
                    end_date=ref_date,
                    action_types=["SPLIT", "BONUS"],
                )

            if not ca_df.empty:
                corp_path = self.cfg.base_dir / "corporate_actions.csv"
                ca_df.to_csv(corp_path, index=False)
                adjuster = CorporateActionAdjuster()
                adjuster.load_corporate_actions(str(corp_path))
                bhav_df = adjuster.adjust_all(bhav_df)
                logger.info("L1: %d corporate actions applied (source: %s).",
                            len(ca_df),
                            ca_df["SOURCE"].value_counts().to_dict() if "SOURCE" in ca_df.columns else "unknown")
            else:
                logger.info("L1: No corporate actions found for this universe.")
        except Exception as e:
            logger.warning("L1: Corporate action adjustment skipped: %s", e)

        # 4. QC
        try:
            from production.data_integrity import DataQualityChecker
            checker = DataQualityChecker()
            bhav_df, qc_report = checker.validate_bhav_data(bhav_df)
            removed = qc_report.get("total_removed", 0)
            if removed:
                logger.info("L1: QC removed %d bad rows.", removed)
        except Exception as e:
            logger.debug("L1: QC skipped: %s", e)

        logger.info("L1: %d rows, %d stocks after data integrity.",
                    len(bhav_df), bhav_df["SC_CODE"].nunique())
        return bhav_df

    # -- L2 ---------------------------------------------------------------
    def _l2_universe(self, bhav_df, ref_date):
        """Apply liquidity / price / volume gate + P06 operator & IPO filters."""
        try:
            from production.universe_filter import (
                TradabilityGate,
                filter_operator_driven,
                filter_recent_listings,
            )
            gate = TradabilityGate(
                min_value_crore=self.cfg.min_value_crore,
                min_price=self.cfg.min_price,
                min_avg_volume=self.cfg.min_avg_volume,
            )
            filtered = gate.apply(bhav_df, ref_date)
            report   = gate.get_filter_report(bhav_df, ref_date)
            logger.info("L2: liquidity gate %s", {k: v for k, v in report.items() if "count" in k.lower() or "final" in k.lower()})

            # P06 — operator / pump filter
            n_before_op = filtered["SC_CODE"].nunique()
            filtered = filter_operator_driven(filtered, lookback=10, max_circuits=2)
            n_after_op = filtered["SC_CODE"].nunique()
            if n_before_op != n_after_op:
                logger.info("L2: operator filter removed %d pumped stocks.", n_before_op - n_after_op)

            # P06 — post-IPO / insufficient-history filter
            n_before_ipo = filtered["SC_CODE"].nunique()
            filtered = filter_recent_listings(filtered, min_sessions=90)
            n_after_ipo = filtered["SC_CODE"].nunique()
            if n_before_ipo != n_after_ipo:
                logger.info("L2: IPO filter removed %d recent listings.", n_before_ipo - n_after_ipo)

            codes = filtered["SC_CODE"].unique().tolist()
            logger.info("L2: %d tradable stocks after all L2 filters.", len(codes))
            return filtered, codes
        except Exception as e:
            logger.warning("L2: Universe filter failed (%s) — using all stocks.", e)
            return bhav_df, bhav_df["SC_CODE"].unique().tolist()

    # -- L3 ---------------------------------------------------------------
    def _l3_regime(self, ref_date):
        """P32/P34/P38: Nifty EMA (3-layer) + 2-state HMM + VIX gate. Returns (regime_dict, threshold)."""
        try:
            from production.regime_filter import get_combined_regime, IndiaVIXGate
            ctx = get_combined_regime()

            regime    = ctx.get("ema_regime", "SIDEWAYS")
            hmm_label = ctx.get("hmm_regime", "UNKNOWN")
            vix_level = ctx.get("vix_level")
            vix_size  = ctx.get("vix_size_multiplier", 1.0) if "vix_size_multiplier" in ctx else (
                ctx.get("position_size_multiplier", 1.0)
            )
            three_action = ctx.get("three_layer_action", "NORMAL")

            # P38: mandatory VIX log line on every L3 run
            action_str = "SKIP_DAY" if not ctx.get("is_tradeable", True) else three_action
            logger.info(
                "P38 India VIX = %s | action=%s | size_factor=%.2f",
                f"{vix_level:.1f}" if vix_level is not None else "N/A",
                action_str,
                vix_size,
            )

            # P38: HMM-EMA conflict alert
            if hmm_label not in ("UNKNOWN", regime.replace("BULL", "BULL_TRENDING").replace("BEAR", "BEAR_TRENDING")):
                logger.warning(
                    "P38 HMM-EMA CONFLICT ALERT: HMM=%s  EMA=%s  combined=%s",
                    hmm_label, regime, ctx.get("combined_label", "?"),
                )

            threshold = ctx.get("recommended_threshold", self.cfg.base_threshold)
            threshold = max(threshold, self.cfg.base_threshold)

            report = {
                "regime":               regime,
                "combined_label":       ctx.get("combined_label", regime),
                "hmm_regime":           hmm_label,
                "hmm_confidence":       ctx.get("hmm_confidence", 0.0),
                "hmm_converged":        ctx.get("hmm_converged", False),
                "is_tradeable":         ctx.get("is_tradeable", True),
                "recommended_threshold":round(threshold, 4),
                "size_multiplier":      ctx.get("position_size_multiplier", 1.0),
                "max_positions":        ctx.get("max_positions", self.cfg.max_positions),
                "nifty_close":          ctx.get("nifty_close"),
                "ema50":                ctx.get("ema50"),
                "ema200":               ctx.get("ema200"),
                "vix_level":            vix_level,
                "vix_regime":           ctx.get("vix_regime", "UNKNOWN"),
                "three_layer_action":   three_action,
                "three_layer_swing":    ctx.get("three_layer_swing"),
                "three_layer_medium":   ctx.get("three_layer_medium"),
                "three_layer_long":     ctx.get("three_layer_long"),
            }
            return report, threshold
        except Exception as e:
            logger.warning("L3: Regime filter failed (%s) — defaults.", e)
            logger.info("P38 India VIX = N/A | action=UNKNOWN | size_factor=1.00")
            return {"regime": "UNKNOWN", "is_tradeable": True}, self.cfg.base_threshold

    # -- L4+L5 ------------------------------------------------------------
    def _l45_signal(self, bhav_df, tradable_codes, ref_date, threshold):
        """Feature engineering + LightGBM prediction + SHAP."""
        # Filter bhav to tradable universe to save compute
        tradable_bhav = bhav_df[bhav_df["SC_CODE"].isin(tradable_codes)].copy()

        # Signal generation (features + model)
        try:
            from production.signal_generator import SignalGenerator
            sg = SignalGenerator(base_dir=str(self.cfg.base_dir))
            picks_df, feature_df = sg.generate(
                tradable_bhav,
                threshold=threshold,
                calibrate=True,
            )
        except Exception as e:
            logger.error("L5: SignalGenerator failed: %s", e)
            return pd.DataFrame(), pd.DataFrame(), []

        # P35: SHAP explanations — structured batch (best-effort)
        explanations = []
        shap_df = pd.DataFrame()
        if not picks_df.empty:
            try:
                from production.signal_generator import SignalGenerator as _SG
                sg2 = _SG(base_dir=str(self.cfg.base_dir))
                model = sg2._ensure_model(feature_df)
                feat_cols = sg2._feature_cols or []

                if model and feat_cols:
                    from production.shap_explainability import explain_batch
                    shap_df = explain_batch(
                        model=model,
                        feature_names=feat_cols,
                        picks_df=picks_df,
                        feature_df=feature_df,
                        date=ref_date.replace("-", ""),
                        log_dir=str(self.cfg.shap_log_dir),
                        top_n=6,
                    )
                    logger.info(
                        "P35: SHAP batch complete — %d picks  dominated=%d",
                        len(shap_df),
                        int(shap_df["shap_dominated"].sum()) if not shap_df.empty else 0,
                    )
            except Exception as e:
                logger.debug("P35: SHAP skipped: %s", e)

        # Model drift check (best-effort)
        try:
            if not feature_df.empty:
                from production.monitoring import ModelDriftMonitor
                ref_path = self.cfg.models_dir / "reference_distribution.pkl"
                if ref_path.exists():
                    monitor = ModelDriftMonitor(str(ref_path))
                    feat_num = feature_df.select_dtypes(include=[np.number])
                    pred_vals = picks_df["Probability"].values if not picks_df.empty else np.array([])
                    if len(pred_vals) > 0:
                        report = monitor.compute_drift_report(feat_num, pred_vals)
                        logger.info("L5: Drift status = %s.", report.get("overall_status"))
        except Exception as e:
            logger.debug("L5: Drift monitor skipped: %s", e)

        return picks_df, feature_df, explanations, shap_df

    # -- L3b --------------------------------------------------------------
    def _l3b_events(self, picks_df, ref_date):
        """Remove stocks within ±N days of earnings / dividends / board meetings."""
        try:
            from production.event_calendar import EventCalendarBlackout
            bl = EventCalendarBlackout(
                blackout_days_before=self.cfg.blackout_days_before,
                blackout_days_after=self.cfg.blackout_days_after,
            )
            sc_codes = picks_df["SC_CODE"].tolist()
            events   = bl.fetch_bse_announcements(
                sc_codes, ref_date, ref_date,
                cache_dir=str(self.cfg.cache_dir / "events"),
            )
            filtered, excluded = bl.filter_picks(picks_df, ref_date, events)
            if not excluded.empty:
                logger.info("L3b: Excluded %d picks (event blackout).", len(excluded))
            return filtered
        except Exception as e:
            logger.debug("L3b: Event blackout skipped: %s", e)
            return picks_df

    # -- L6 ---------------------------------------------------------------
    def _l6_portfolio(self, picks_df, bhav_df):
        """Sector cap, correlation filter, position / capital limits."""
        try:
            from production.portfolio_constructor import PortfolioConstructor
            pc = PortfolioConstructor(
                max_positions=self.cfg.max_positions,
                max_sector_positions=self.cfg.max_sector_positions,
                max_capital_deployed=self.cfg.max_capital_deployed,
                max_correlation=self.cfg.max_correlation,
            )
            picks_df = pc.construct_portfolio(
                picks_df, bhav_df,
                total_capital=self.cfg.total_capital,
            )
            stats = pc.get_portfolio_stats(picks_df, bhav_df)
            logger.info(
                "L6: sectors=%s  n=%d  capital_deployed=%.0f%%",
                list(stats.get("sector_distribution", {}).keys())[:5],
                stats.get("num_positions", len(picks_df)),
                stats.get("total_capital_deployed", 0) * 100,
            )
        except Exception as e:
            logger.warning("L6: Portfolio construction partial (%s).", e)
            picks_df = picks_df.head(self.cfg.max_positions)
        return picks_df

    # -- L7 ---------------------------------------------------------------
    def _l7_execution(self, picks_df, bhav_df, size_mult, regime):
        """ATR stop-loss + position sizing + friction model + ADV cap."""
        # Ensure Close column
        if "Close" not in picks_df.columns:
            latest = (bhav_df.sort_values("DATE")
                              .groupby("SC_CODE")["Close"]
                              .last())
            picks_df = picks_df.copy()
            picks_df["Close"] = picks_df["SC_CODE"].map(latest).fillna(100)

        picks_df["Close"] = pd.to_numeric(picks_df["Close"], errors="coerce").fillna(100)

        # ATR lookup
        try:
            from production.exit_engine import ATRCalculator
            atr_calc = ATRCalculator()
            atr_dict = atr_calc.get_atr_for_universe(
                bhav_df, bhav_df["DATE"].max()
            )
            picks_df["ATR14"] = picks_df["SC_CODE"].map(atr_dict).fillna(
                picks_df["Close"] * 0.02
            )
        except Exception as e:
            logger.debug("L7: ATR calc failed (%s) — using 2%% proxy.", e)
            if "ATR14" not in picks_df.columns:
                picks_df["ATR14"] = picks_df["Close"] * 0.02

        # Stop-loss prices
        picks_df["Stop_Loss_Price"] = (
            picks_df["Close"] - self.cfg.initial_stop_atr * picks_df["ATR14"]
        ).round(2)
        picks_df["Trailing_Stop_ATR"] = self.cfg.trailing_stop_atr

        # Position sizing
        try:
            from production.position_sizer import PositionSizer
            sizer = PositionSizer(
                total_capital=self.cfg.total_capital,
                risk_pct_per_trade=self.cfg.risk_pct_per_trade,
                max_position_pct=self.cfg.max_position_pct,
                max_positions=self.cfg.max_positions,
            )
            atr_dict_for_sizer = dict(
                zip(picks_df["SC_CODE"], picks_df["ATR14"])
            )
            picks_df = sizer.calculate_portfolio_allocation(
                picks_df, self.cfg.total_capital, atr_dict_for_sizer
            )
        except Exception as e:
            logger.warning("L7: Position sizer failed (%s) — equal-weight fallback.", e)
            n = max(len(picks_df), 1)
            alloc = self.cfg.total_capital * self.cfg.max_capital_deployed / n
            picks_df["Position_Value"] = alloc
            picks_df["Shares"] = (alloc / picks_df["Close"]).astype(int)
            picks_df["Risk_Amount"] = picks_df["Shares"] * picks_df["ATR14"] * self.cfg.initial_stop_atr
            picks_df["Risk_Pct"]    = picks_df["Risk_Amount"] / self.cfg.total_capital

        # Apply circuit-breaker size multiplier
        if size_mult != 1.0:
            for col in ["Shares", "Position_Value"]:
                if col in picks_df.columns:
                    picks_df[col] = (picks_df[col] * size_mult)
                    if col == "Shares":
                        picks_df[col] = picks_df[col].astype(int)
            logger.info("L7: Sizes scaled by %.0f%% (circuit breaker).", size_mult * 100)

        # Also apply HMM size multiplier if present
        hmm_mult = regime.get("size_multiplier", 1.0)
        if hmm_mult != 1.0 and "Shares" in picks_df.columns:
            picks_df["Shares"] = (picks_df["Shares"] * hmm_mult).astype(int)
            logger.info("L7: HMM regime size multiplier %.2f applied.", hmm_mult)

        # ADV volume cap (≤ 1% ADV)
        try:
            from production.universe_filter import TradabilityGate
            gate = TradabilityGate()
            for idx, row in picks_df.iterrows():
                max_sh = gate.get_max_tradeable_shares(bhav_df, row["SC_CODE"])
                if max_sh > 0 and picks_df.at[idx, "Shares"] > max_sh:
                    picks_df.at[idx, "Shares"] = int(max_sh)
                    picks_df.at[idx, "Position_Value"] = max_sh * row["Close"]
        except Exception as e:
            logger.debug("L7: ADV cap skipped: %s", e)

        # P09 — Execution reality checks (volume, circuit, gap, liquidity)
        try:
            from production.execution_checks import apply_execution_checks
            _exec_ref_date = bhav_df["DATE"].max()  # latest date in BhavCopy
            n_before = len(picks_df)
            picks_df, ec_report = apply_execution_checks(
                picks_df, bhav_df, _exec_ref_date,
                max_adv_fraction=0.50,
                circuit_pct=0.195,
                max_gap_pct=0.04,
                max_impact_fraction=0.01,
            )
            n_removed = n_before - len(picks_df)
            if n_removed > 0:
                logger.info(
                    "L7: P09 execution checks removed %d picks "
                    "(vol=%d, circuit=%d, gap=%d, liq=%d).",
                    n_removed,
                    ec_report.get("volume_rejected", 0),
                    ec_report.get("circuit_rejected", 0),
                    ec_report.get("gap_rejected", 0),
                    ec_report.get("liquidity_rejected", 0),
                )
        except Exception as e:
            logger.debug("L7: P09 execution checks skipped: %s", e)

        # Friction model
        try:
            from production.friction_model import FrictionModel
            fm = FrictionModel(instrument_type=self.cfg.instrument_type)
            def _friction(row):
                val = row.get("Position_Value", row.get("Close", 100))
                rt  = fm.calculate_round_trip_cost(float(val))
                return rt.get("total_pct", 0.002)   # fraction
            picks_df["Friction_Pct"] = picks_df.apply(_friction, axis=1)
            picks_df["Net_Expected_Return"] = (
                picks_df["Probability"] * 0.05 - picks_df["Friction_Pct"]
            ).round(5)
            logger.info("L7: Avg friction = %.3f%%.",
                        picks_df["Friction_Pct"].mean() * 100)
        except Exception as e:
            logger.debug("L7: Friction model skipped: %s", e)
            picks_df["Friction_Pct"] = 0.002
            picks_df["Net_Expected_Return"] = picks_df["Probability"] * 0.05 - 0.002

        return picks_df

    # -- L9 ---------------------------------------------------------------
    def _l9_compliance(self, picks_df, ref_date):
        """Log every signal + order to SEBI audit trail."""
        try:
            from production.compliance import AuditTrail, SEBIComplianceChecker
            audit = AuditTrail(log_dir=str(self.cfg.audit_dir))
            checker = SEBIComplianceChecker()

            for _, row in picks_df.iterrows():
                sc = str(row.get("SC_CODE", ""))
                nm = str(row.get("SC_NAME", ""))
                audit.log_signal(
                    sc_code=sc, sc_name=nm,
                    probability=float(row.get("Probability", 0)),
                    features_dict={
                        "RSI14":  float(row.get("RSI14", 0)),
                        "ADX14":  float(row.get("ADX14", 0)),
                        "VolMult": float(row.get("VolMult", 0)),
                    },
                    regime=str(row.get("Regime", "")),
                )
                audit.log_order(
                    sc_code=sc, sc_name=nm,
                    order_type="BUY",
                    quantity=int(row.get("Shares", 0)),
                    price=float(row.get("Close", 0)),
                    order_id=f"SIG_{sc}_{ref_date.replace('-','')}",
                )

            # SEBI position limit check
            pos_dict = dict(zip(picks_df["SC_CODE"],
                                picks_df.get("Position_Value", picks_df["Close"])))
            ok, msg = checker.check_position_limits(pos_dict, self.cfg.total_capital)
            if not ok:
                logger.warning("L9: SEBI position limit: %s", msg)

            audit.log_event("SYSTEM_STOP",
                            details={"picks_count": len(picks_df), "date": ref_date})
            logger.info("L9: Audit trail written -> %s.", self.cfg.audit_dir)
        except Exception as e:
            logger.debug("L9: Compliance logging skipped: %s", e)

    # ====================================================================
    # HELPERS
    # ====================================================================

    def _kill_switch_active(self) -> bool:
        try:
            from production.risk_controls import KillSwitch
            ks = KillSwitch()
            active = ks.is_active()
            if active:
                logger.warning("Kill switch: %s", ks.get_reason())
            return active
        except Exception:
            return False

    def _check_circuit_breaker(self):
        """Returns (state_name, size_multiplier)."""
        try:
            from production.risk_controls import DrawdownCircuitBreaker
            cb = DrawdownCircuitBreaker(
                warning_threshold=self.cfg.drawdown_warning,
                halt_threshold=self.cfg.drawdown_halt,
                state_file=str(self.cfg.base_dir / "circuit_breaker_state.json"),
            )
            cb.load_state()
            state = cb.check_state()
            mult  = cb.get_position_size_multiplier()
            cb.save_state()
            return state.name, mult
        except Exception:
            return "NORMAL", 1.0

    def _write_daily_summary(self, result: dict, ref_date: str) -> None:
        """
        P38: Write daily_summary_YYYYMMDD.json on every run, including SKIP_DAY.
        """
        try:
            date_str = ref_date.replace("-", "")
            out_path = self.cfg.results_dir / f"daily_summary_{date_str}.json"
            regime = result.get("regime", {})
            picks  = result.get("picks", pd.DataFrame())
            payload = {
                "date":           ref_date,
                "action":         result.get("action", "SKIP_DAY"),
                "n_picks":        len(picks) if picks is not None else 0,
                "circuit_state":  result.get("circuit_state", "UNKNOWN"),
                "regime":         regime.get("regime") if isinstance(regime, dict) else str(regime),
                "combined_label": regime.get("combined_label") if isinstance(regime, dict) else None,
                "hmm_regime":     regime.get("hmm_regime") if isinstance(regime, dict) else None,
                "hmm_converged":  regime.get("hmm_converged") if isinstance(regime, dict) else None,
                "vix_level":      regime.get("vix_level") if isinstance(regime, dict) else None,
                "vix_regime":     regime.get("vix_regime") if isinstance(regime, dict) else None,
                "three_layer_action": regime.get("three_layer_action") if isinstance(regime, dict) else None,
                "threshold":      regime.get("recommended_threshold") if isinstance(regime, dict) else None,
                "output_csv":     result.get("output_csv"),
                "generated_at":   datetime.now().isoformat(timespec="seconds"),
            }
            with open(out_path, "w") as fh:
                json.dump(payload, fh, indent=2, default=str)
            logger.info("P38 daily_summary written -> %s", out_path)
        except Exception as exc:
            logger.debug("P38 daily_summary write failed: %s", exc)

    def _p49_risk(self, picks_df: pd.DataFrame) -> dict:
        """P49: Compute VaR and run stress tests; log results."""
        try:
            from production.risk_analytics import PortfolioRiskMonitor
            risk = PortfolioRiskMonitor()
            var_report   = risk.compute_var(picks_df)
            stress_report = risk.run_stress_test(picks_df)
            return {"var": var_report, "stress": stress_report}
        except Exception as exc:
            logger.debug("P49: risk analytics skipped: %s", exc)
            return {}

    def _audit(self, event_type, **kwargs):
        try:
            from production.compliance import AuditTrail
            AuditTrail(log_dir=str(self.cfg.audit_dir)).log_event(event_type, **kwargs)
        except Exception:
            pass

    def _annotate(self, picks_df, regime, ref_date, explanations, shap_df=None) -> pd.DataFrame:
        """Add Regime, Sector, Rationale, and P35 SHAP columns for the output CSV."""
        picks_df = picks_df.copy()
        picks_df["Regime"] = regime.get("regime", "UNKNOWN")

        # Sector classification
        try:
            from production.portfolio_constructor import PortfolioConstructor
            pc = PortfolioConstructor()
            picks_df["Sector"] = picks_df["SC_NAME"].apply(pc.classify_sector)
        except Exception:
            picks_df["Sector"] = "OTHERS"

        # SHAP rationale text
        if explanations:
            rationale_map = {e["sc_code"]: e.get("rationale", "") for e in explanations}
            picks_df["Rationale"] = picks_df["SC_CODE"].map(rationale_map).fillna("")
        else:
            picks_df["Rationale"] = ""

        # P35: structured SHAP columns (top-3 features + dominated flag)
        shap_cols = [
            "shap_top1_feature", "shap_top1_value",
            "shap_top2_feature", "shap_top2_value",
            "shap_top3_feature", "shap_top3_value",
            "shap_dominated",
        ]
        if shap_df is not None and not shap_df.empty:
            picks_df = picks_df.set_index("SC_CODE").join(
                shap_df[shap_cols], how="left"
            ).reset_index()
        else:
            for col in shap_cols:
                picks_df[col] = "" if "feature" in col else (False if col == "shap_dominated" else 0.0)

        picks_df["Prediction_Date"] = ref_date
        return picks_df

    def _write_csv(self, picks_df, ref_date) -> Path:
        """Write picks to results directory."""
        date_str = ref_date.replace("-", "")
        out = self.cfg.results_dir / f"picks_{date_str}.csv"

        # Column order for the output file
        preferred_order = [
            "Rank", "SC_CODE", "SC_NAME", "Close",
            "Probability", "Probability_Raw",
            "ATR14", "Stop_Loss_Price", "Trailing_Stop_ATR",
            "Shares", "Position_Value", "Risk_Amount", "Risk_Pct",
            "Friction_Pct", "Net_Expected_Return",
            "Sector", "Regime", "Signal_Threshold",
            "RSI14", "ADX14", "VolMult", "RS_Composite", "DistTo52W",
            "Break63_Today", "EMA20", "EMA50", "EMA200",
            "Prediction_Date", "Rationale",
        ]
        cols = [c for c in preferred_order if c in picks_df.columns]
        extra = [c for c in picks_df.columns if c not in cols]
        picks_df[cols + extra].to_csv(out, index=False)
        logger.info("Picks CSV -> %s", out)
        return out

    def _print_summary(self, picks_df, ref_date):
        """Print a compact summary table to stdout."""
        print(f"\n{'='*72}")
        print(f"  SWING TRADE PICKS  —  {ref_date}  —  {len(picks_df)} stocks")
        print(f"{'='*72}")
        disp = [c for c in ["Rank","SC_NAME","Close","Probability",
                             "Shares","Stop_Loss_Price","Position_Value","Sector"]
                if c in picks_df.columns]
        pd.set_option("display.max_rows", 50)
        pd.set_option("display.width", 120)
        print(picks_df[disp].head(30).to_string(index=False))
        if "Position_Value" in picks_df.columns:
            total = picks_df["Position_Value"].sum()
            print(f"\n  Total capital deployed: Rs.{total:,.0f} "
                  f"({total/self.cfg.total_capital:.1%} of Rs.{self.cfg.total_capital:,})")
        print(f"{'='*72}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Run the production TradeOrchestrator for one trading day."
    )
    parser.add_argument("--date",      default=None,      help="YYYY-MM-DD (default: today)")
    parser.add_argument("--capital",   default=1_000_000, type=float, help="Total capital in Rs.")
    parser.add_argument("--min-price", default=20.0,      type=float, help="Min stock price filter")
    parser.add_argument("--min-value", default=2.0,       type=float, help="Min daily value (Rs. crore)")
    parser.add_argument("--threshold", default=None,      type=float, help="Probability threshold override")
    args = parser.parse_args()

    cfg = OrchestratorConfig()
    cfg.total_capital   = args.capital
    cfg.min_price       = args.min_price
    cfg.min_value_crore = args.min_value
    if args.threshold:
        cfg.base_threshold = args.threshold

    orch = TradeOrchestrator(config=cfg)
    orch.run_daily(reference_date=args.date)


if __name__ == "__main__":
    main()
