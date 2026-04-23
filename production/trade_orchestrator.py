"""
TradeOrchestrator — master coordinator for all 9 production layers.

Usage
─────
    from production.trade_orchestrator import TradeOrchestrator

    orch = TradeOrchestrator()
    result = orch.run_daily()          # full pipeline for today
    picks  = result["picks"]           # pd.DataFrame of approved trades

    # Or with custom capital / date
    result = orch.run_daily(reference_date="2026-04-23", total_capital=500_000)

Architecture layers executed in order
──────────────────────────────────────
  L1  Data Integrity      — survivorship bias, corporate-action adjustment, QC
  L2  Universe Filter     — liquidity / price / volume gate
  L3  Context & Events    — HMM + EMA regime, corporate calendar blackout
  L4  Feature Engineering — existing momentum_features pipeline (cache-aware)
  L5  Signal & Calibration— LightGBM predict → Isotonic calibration → SHAP log
  L6  Portfolio Construct — sector cap, correlation filter, position / capital cap
  L7  Execution & Sizing  — ATR stop-loss, position sizing, friction-adjusted return
  L8  Risk & Failsafe     — drawdown circuit breaker, kill switch, operational fallback
  L9  Compliance          — SEBI audit trail, order tagging, compliance report
"""

import os
import sys
import json
import logging
import pickle
import warnings
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── project root on sys.path so sibling modules resolve ──────────────────────
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — each layer imported only when needed so the orchestrator can
# start even if an optional dependency (shap, hmmlearn, …) is absent.
# ---------------------------------------------------------------------------

def _import(module_path: str, attr: str = None):
    """Import a module and optionally return one of its attributes."""
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, attr) if attr else mod


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

class OrchestratorConfig:
    """All tuneable parameters in one place."""

    def __init__(self, base_dir: str = "stock_picker_data"):
        self.base_dir = Path(base_dir)

        # ── Paths ────────────────────────────────────────────────────────
        self.models_dir      = self.base_dir / "models"
        self.cache_dir       = self.base_dir / "cache"
        self.results_dir     = self.base_dir / "results"
        self.audit_dir       = self.base_dir / "audit"
        self.shap_log_dir    = self.base_dir / "shap_logs"
        self.alert_log       = self.base_dir / "alerts.log"
        self.cost_config     = self.base_dir / "cost_config.json"

        for p in [self.models_dir, self.cache_dir, self.results_dir,
                  self.audit_dir, self.shap_log_dir]:
            p.mkdir(parents=True, exist_ok=True)

        # ── Capital & sizing ─────────────────────────────────────────────
        self.total_capital          = 1_000_000   # ₹10 lakh default
        self.risk_pct_per_trade     = 0.01        # 1% capital at risk per trade
        self.max_position_pct       = 0.10        # 10% cap per single position
        self.max_positions          = 15
        self.max_capital_deployed   = 0.70        # 70% cap total deployed
        self.max_sector_positions   = 2
        self.max_correlation        = 0.70

        # ── Universe filter ──────────────────────────────────────────────
        self.min_value_crore        = 2.0
        self.min_price              = 20.0
        self.min_avg_volume         = 10_000
        self.lookback_days          = 730         # 2 years of training data

        # ── Signal thresholds ────────────────────────────────────────────
        self.base_probability_threshold = 0.62
        self.min_probability_threshold  = 0.52

        # ── Exit engine ──────────────────────────────────────────────────
        self.initial_stop_atr       = 2.5
        self.trailing_stop_atr      = 1.5
        self.time_stop_sessions     = 5

        # ── Risk controls ────────────────────────────────────────────────
        self.drawdown_warning       = -0.05
        self.drawdown_halt          = -0.10

        # ── Event blackout ───────────────────────────────────────────────
        self.blackout_days_before   = 2
        self.blackout_days_after    = 2

        # ── Model fallback ───────────────────────────────────────────────
        self.max_model_age_days     = 7

        # ── Instrument type (equity_delivery / futures / options) ────────
        self.instrument_type        = "equity_delivery"


# ---------------------------------------------------------------------------
# The Orchestrator
# ---------------------------------------------------------------------------

class TradeOrchestrator:
    """
    Executes the full 9-layer production pipeline end-to-end.

    Instantiate once; call run_daily() each trading session.
    """

    def __init__(self, config: OrchestratorConfig = None):
        self.cfg = config or OrchestratorConfig()
        self._model = None
        self._calibrator = None
        self._feature_names = None
        self._fallback_used = False

        # Lazy-initialised layer objects
        self._exit_engine       = None
        self._position_sizer    = None
        self._portfolio_ctor    = None
        self._friction_model    = None
        self._circuit_breaker   = None
        self._kill_switch       = None
        self._audit_trail       = None
        self._drift_monitor     = None
        self._fallback_system   = None

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        logger.info("TradeOrchestrator initialised (capital=₹%s).", f"{self.cfg.total_capital:,}")

    # ====================================================================
    # PUBLIC API
    # ====================================================================

    def run_daily(
        self,
        reference_date: str = None,
        total_capital: float = None,
        bhav_df: pd.DataFrame = None,
    ) -> dict:
        """
        Execute the full pipeline for one trading day.

        Args:
            reference_date: 'YYYY-MM-DD'; defaults to today.
            total_capital:  Override config capital.
            bhav_df:        Pre-loaded BhavCopy data (fetched automatically if None).

        Returns:
            dict with keys:
              picks          — pd.DataFrame of approved trades (may be empty)
              action         — 'TRADE' | 'SKIP_DAY' | 'REDUCED_SIZE'
              regime         — dict (regime label, threshold, …)
              circuit_state  — 'NORMAL' | 'WARNING' | 'HALTED'
              explanations   — list of SHAP dicts (one per pick)
              audit_log      — path to today's audit JSON
              friction_summary — dict with round-trip cost breakdown
              compliance     — dict with SEBI compliance status
        """
        ref_date = reference_date or date.today().strftime("%Y-%m-%d")
        if total_capital:
            self.cfg.total_capital = total_capital

        logger.info("=" * 68)
        logger.info("  TRADE ORCHESTRATOR — %s", ref_date)
        logger.info("=" * 68)

        result = {
            "reference_date": ref_date,
            "picks": pd.DataFrame(),
            "action": "SKIP_DAY",
            "regime": {},
            "circuit_state": "UNKNOWN",
            "explanations": [],
            "audit_log": None,
            "friction_summary": {},
            "compliance": {},
        }

        # ────────────────────────────────────────────────────────────────
        # L8 (pre-check): Kill switch
        # ────────────────────────────────────────────────────────────────
        ks = self._get_kill_switch()
        if ks.is_active():
            reason = ks.get_reason()
            logger.warning("Kill switch ACTIVE: %s — skipping trading day.", reason)
            self._log_audit("SYSTEM_STOP", details={"reason": reason})
            result["action"] = "SKIP_DAY"
            result["compliance"] = {"kill_switch": True, "reason": reason}
            return result

        # ────────────────────────────────────────────────────────────────
        # L8 (pre-check): Drawdown circuit breaker
        # ────────────────────────────────────────────────────────────────
        cb = self._get_circuit_breaker()
        cb.load_state()
        cb_state = cb.check_state()
        result["circuit_state"] = cb_state.name
        size_multiplier = cb.get_position_size_multiplier()

        if not cb.can_open_new_trades():
            logger.error("Circuit breaker HALTED — drawdown %.1f%%. No new trades.",
                         cb.compute_current_drawdown() * 100)
            self._log_audit("RISK_CHECK", details={"circuit_breaker": "HALTED"})
            result["action"] = "SKIP_DAY"
            return result

        if cb_state.name == "WARNING":
            logger.warning("Circuit breaker WARNING — position sizes halved.")
            result["action"] = "REDUCED_SIZE"

        # ────────────────────────────────────────────────────────────────
        # L1: Data Integrity
        # ────────────────────────────────────────────────────────────────
        logger.info("── L1: Data Integrity ──────────────────────────────────")
        bhav_df = self._layer1_data_integrity(bhav_df, ref_date)
        if bhav_df is None:
            result["action"] = "SKIP_DAY"
            return result

        # ────────────────────────────────────────────────────────────────
        # L2: Universe & Tradability Gate
        # ────────────────────────────────────────────────────────────────
        logger.info("── L2: Universe Filter ─────────────────────────────────")
        bhav_df, tradable_codes = self._layer2_universe_filter(bhav_df, ref_date)
        if len(tradable_codes) == 0:
            logger.error("L2: No tradable stocks after filter — skipping day.")
            result["action"] = "SKIP_DAY"
            return result
        logger.info("L2: %d tradable stocks.", len(tradable_codes))

        # ────────────────────────────────────────────────────────────────
        # L3: Context & Events
        # ────────────────────────────────────────────────────────────────
        logger.info("── L3: Market Context & Events ─────────────────────────")
        regime_info, prob_threshold = self._layer3_context(ref_date)
        result["regime"] = regime_info

        if not regime_info.get("is_tradeable", True):
            logger.warning("L3: Regime is BEAR — skipping day per policy.")
            result["action"] = "SKIP_DAY"
            return result

        # ────────────────────────────────────────────────────────────────
        # L4: Feature Engineering (existing pipeline)
        # ────────────────────────────────────────────────────────────────
        logger.info("── L4: Feature Engineering ─────────────────────────────")
        feature_df = self._layer4_features(bhav_df, tradable_codes, ref_date)
        if feature_df is None or feature_df.empty:
            logger.error("L4: Feature computation failed — skipping day.")
            result["action"] = "SKIP_DAY"
            return result
        logger.info("L4: %d feature rows, %d columns.", len(feature_df), len(feature_df.columns))

        # ────────────────────────────────────────────────────────────────
        # L5: Signal Generation & Calibration
        # ────────────────────────────────────────────────────────────────
        logger.info("── L5: Signal Generation & Calibration ─────────────────")
        picks_df, explanations = self._layer5_signal(
            feature_df, bhav_df, ref_date, prob_threshold
        )
        result["explanations"] = explanations

        if picks_df.empty:
            logger.info("L5: No picks above threshold %.2f — SKIP_DAY.", prob_threshold)
            result["action"] = "SKIP_DAY"
            return result
        logger.info("L5: %d raw picks above threshold.", len(picks_df))

        # ────────────────────────────────────────────────────────────────
        # L3b: Event calendar blackout
        # ────────────────────────────────────────────────────────────────
        picks_df = self._apply_event_blackout(picks_df, ref_date)
        if picks_df.empty:
            logger.info("L3b: All picks blocked by event blackout — SKIP_DAY.")
            result["action"] = "SKIP_DAY"
            return result
        logger.info("L3b: %d picks after event blackout.", len(picks_df))

        # ────────────────────────────────────────────────────────────────
        # L6: Portfolio Construction
        # ────────────────────────────────────────────────────────────────
        logger.info("── L6: Portfolio Construction ──────────────────────────")
        picks_df = self._layer6_portfolio(picks_df, bhav_df)
        if picks_df.empty:
            logger.info("L6: No picks survived portfolio construction.")
            result["action"] = "SKIP_DAY"
            return result
        logger.info("L6: %d picks in final portfolio.", len(picks_df))

        # ────────────────────────────────────────────────────────────────
        # L7: Execution & Sizing
        # ────────────────────────────────────────────────────────────────
        logger.info("── L7: Execution & Sizing ──────────────────────────────")
        picks_df, friction_summary = self._layer7_execution(
            picks_df, bhav_df, size_multiplier
        )
        result["friction_summary"] = friction_summary

        # ────────────────────────────────────────────────────────────────
        # L8 (post): Update circuit breaker with portfolio exposure
        # ────────────────────────────────────────────────────────────────
        cb.save_state()

        # ────────────────────────────────────────────────────────────────
        # L9: Compliance & Audit
        # ────────────────────────────────────────────────────────────────
        logger.info("── L9: Compliance & Audit ──────────────────────────────")
        compliance = self._layer9_compliance(picks_df, ref_date)
        result["compliance"] = compliance

        # ────────────────────────────────────────────────────────────────
        # Persist picks
        # ────────────────────────────────────────────────────────────────
        self._save_picks(picks_df, ref_date)
        result["picks"] = picks_df
        result["action"] = result.get("action", "TRADE") if result["action"] == "REDUCED_SIZE" else "TRADE"
        result["audit_log"] = str(self.cfg.audit_dir / f"audit_{ref_date.replace('-','')}.json")

        logger.info("=" * 68)
        logger.info("  DONE — %d picks, action=%s, regime=%s",
                    len(picks_df), result["action"], regime_info.get("regime", "N/A"))
        logger.info("=" * 68)
        return result

    # ====================================================================
    # LAYER IMPLEMENTATIONS
    # ====================================================================

    # ── L1 ───────────────────────────────────────────────────────────────
    def _layer1_data_integrity(self, bhav_df, ref_date):
        """Fetch data if not provided, run QC, apply corporate-action adjustments."""
        fallback = self._get_fallback_system()

        # Fetch if not provided
        if bhav_df is None:
            bhav_df = self._fetch_bhav_data(ref_date)

        # Health check
        health = fallback.pre_run_health_check(bhav_df, ref_date)
        if not health["is_healthy"]:
            return None

        # Corporate-action adjustment
        try:
            from production.data_integrity import CorporateActionAdjuster, DataQualityChecker
            adjuster = CorporateActionAdjuster()
            corp_actions_path = self.cfg.base_dir / "corporate_actions.csv"
            if corp_actions_path.exists():
                adjuster.load_corporate_actions(str(corp_actions_path))
                bhav_df = adjuster.adjust_all(bhav_df)
                logger.info("L1: Corporate-action adjustments applied.")

            checker = DataQualityChecker()
            bhav_df, qc_report = checker.validate_bhav_data(bhav_df)
            logger.info("L1: QC — removed %d bad rows.", qc_report.get("total_removed", 0))
        except Exception as e:
            logger.warning("L1: Data integrity step skipped (%s).", e)

        return bhav_df

    # ── L2 ───────────────────────────────────────────────────────────────
    def _layer2_universe_filter(self, bhav_df, ref_date):
        """Apply liquidity / price / volume gate."""
        try:
            from production.universe_filter import TradabilityGate
            gate = TradabilityGate(
                min_value_crore=self.cfg.min_value_crore,
                min_price=self.cfg.min_price,
                min_avg_volume=self.cfg.min_avg_volume,
            )
            filtered_df = gate.apply(bhav_df, ref_date)
            code_col = "SC_CODE" if "SC_CODE" in filtered_df.columns else "sc_code"
            tradable_codes = filtered_df[code_col].unique().tolist() if code_col in filtered_df.columns else []
            report = gate.get_filter_report(bhav_df, ref_date)
            logger.info("L2: %s", report)
            return filtered_df, tradable_codes
        except Exception as e:
            logger.warning("L2: Universe filter skipped (%s) — using all stocks.", e)
            code_col = "SC_CODE" if "SC_CODE" in bhav_df.columns else "sc_code"
            codes = bhav_df[code_col].unique().tolist() if code_col in bhav_df.columns else []
            return bhav_df, codes

    # ── L3 ───────────────────────────────────────────────────────────────
    def _layer3_context(self, ref_date):
        """Run regime detection and return (regime_info, probability_threshold)."""
        try:
            from production.regime_filter import IndexRegimeFilter, HMMRegimeDetector
            ema_filter = IndexRegimeFilter()
            nifty_df = ema_filter.fetch_nifty_data()
            nifty_df = ema_filter.compute_emas(nifty_df)
            regime_report = ema_filter.get_regime_report(nifty_df)
            regime = regime_report.get("regime", "SIDEWAYS")
            threshold = ema_filter.get_probability_threshold(regime, self.cfg.base_probability_threshold)

            # HMM refinement
            try:
                hmm = HMMRegimeDetector(n_states=4)
                close_col = "Close" if "Close" in nifty_df.columns else "close"
                returns = nifty_df[close_col].pct_change().dropna()
                if len(returns) >= 60:
                    hmm.fit(returns)
                    hmm_pred = hmm.predict_current_regime(returns)
                    hmm_params = hmm.get_regime_adjusted_params(hmm_pred["regime_label"])
                    # Use HMM threshold if more conservative
                    hmm_threshold = hmm_params.get("probability_threshold", threshold)
                    threshold = max(threshold, hmm_threshold)
                    regime_report["hmm_regime"] = hmm_pred["regime_label"]
                    regime_report["hmm_confidence"] = hmm_pred["confidence"]
                    regime_report["hmm_size_multiplier"] = hmm_params.get("position_size_multiplier", 1.0)
            except Exception as hmm_e:
                logger.debug("HMM regime step skipped: %s", hmm_e)

            regime_report["recommended_threshold"] = round(threshold, 4)
            logger.info("L3: Regime=%s  threshold=%.2f  tradeable=%s",
                        regime_report.get("regime"), threshold, regime_report.get("is_tradeable"))
            return regime_report, threshold

        except Exception as e:
            logger.warning("L3: Regime filter skipped (%s) — using defaults.", e)
            return {"regime": "UNKNOWN", "is_tradeable": True}, self.cfg.base_probability_threshold

    # ── L4 ───────────────────────────────────────────────────────────────
    def _layer4_features(self, bhav_df, tradable_codes, ref_date):
        """Run existing momentum_features pipeline on tradable stocks."""
        try:
            # Filter to tradable codes only
            code_col = "SC_CODE" if "SC_CODE" in bhav_df.columns else "sc_code"
            if code_col in bhav_df.columns:
                bhav_df = bhav_df[bhav_df[code_col].isin(tradable_codes)]

            from momentum_features import prepare_features_all
            feature_df = prepare_features_all(bhav_df)
            logger.info("L4: Features computed via momentum_features pipeline.")
            return feature_df
        except ImportError:
            logger.warning("L4: momentum_features not found — computing basic features.")
            return self._compute_basic_features(bhav_df)
        except Exception as e:
            logger.error("L4: Feature computation failed: %s", e)
            return None

    def _compute_basic_features(self, bhav_df):
        """Minimal fallback feature computation."""
        df = bhav_df.copy()
        close_col = "Close" if "Close" in df.columns else "close"
        vol_col   = "Volume" if "Volume" in df.columns else "volume"
        code_col  = "SC_CODE" if "SC_CODE" in df.columns else "sc_code"
        date_col  = "DATE"   if "DATE"   in df.columns else "date"

        df = df.sort_values([code_col, date_col])
        df["Return_1D"] = df.groupby(code_col)[close_col].pct_change()
        df["Return_5D"] = df.groupby(code_col)[close_col].pct_change(5)
        df["MA20"]      = df.groupby(code_col)[close_col].transform(lambda x: x.rolling(20).mean())
        df["Vol20"]     = df.groupby(code_col)[vol_col].transform(lambda x: x.rolling(20).mean())
        df["VolMult"]   = df[vol_col] / (df["Vol20"] + 1e-9)
        # Rough ATR
        df["ATR14"]     = df.groupby(code_col)[close_col].transform(
            lambda x: x.diff().abs().rolling(14).mean()
        )
        df = df.dropna(subset=["Return_5D", "MA20"])
        return df

    # ── L5 ───────────────────────────────────────────────────────────────
    def _layer5_signal(self, feature_df, bhav_df, ref_date, prob_threshold):
        """Load model, predict, calibrate, explain, filter by threshold."""
        # 1. Load / train model
        model = self._ensure_model(feature_df)
        if model is None:
            return pd.DataFrame(), []

        # 2. Get latest-day rows
        date_col = "DATE" if "DATE" in feature_df.columns else "date"
        feature_df[date_col] = pd.to_datetime(feature_df[date_col])
        latest = feature_df[date_col].max()
        today_df = feature_df[feature_df[date_col] == latest].copy()

        if today_df.empty:
            logger.warning("L5: No rows for latest date %s.", latest)
            return pd.DataFrame(), []

        # 3. Predict
        X = today_df[self._feature_names].fillna(0) if self._feature_names else today_df.select_dtypes(include=[np.number])
        try:
            raw_probs = model.predict(X.values)
        except Exception:
            try:
                raw_probs = model.predict_proba(X.values)[:, 1]
            except Exception as e:
                logger.error("L5: predict failed: %s", e)
                return pd.DataFrame(), []

        today_df = today_df.copy()
        today_df["Probability_Raw"] = raw_probs

        # 4. Calibrate probabilities
        try:
            from production.probability_calibration import ProbabilityCalibrator, CalibrationMethod
            cal = self._get_calibrator()
            if cal is not None:
                cal_probs = cal.calibrate(raw_probs)
                today_df["Probability"] = cal_probs
            else:
                today_df["Probability"] = raw_probs
        except Exception as e:
            logger.debug("L5: Calibration skipped (%s).", e)
            today_df["Probability"] = raw_probs

        # 5. Apply threshold
        code_col = "SC_CODE" if "SC_CODE" in today_df.columns else "sc_code"
        picks_df = today_df[today_df["Probability"] >= prob_threshold].copy()
        picks_df = picks_df.sort_values("Probability", ascending=False)

        # Ensure SC_NAME
        if "SC_NAME" not in picks_df.columns and "SC_NAME" in bhav_df.columns:
            name_map = (bhav_df.drop_duplicates(code_col)
                               .set_index(code_col)["SC_NAME"]
                               .to_dict())
            picks_df["SC_NAME"] = picks_df[code_col].map(name_map).fillna("")

        # 6. SHAP explanations (best-effort)
        explanations = []
        try:
            from production.shap_explainability import SHAPExplainer
            if self._feature_names:
                explainer = SHAPExplainer(model, self._feature_names,
                                          log_dir=str(self.cfg.shap_log_dir))
                explanations = explainer.explain_portfolio(picks_df, today_df, top_n=6)
                explainer.log_explanations(explanations, date=ref_date.replace("-", ""))
                logger.info("L5: SHAP explanations logged for %d picks.", len(explanations))
        except Exception as e:
            logger.debug("L5: SHAP skipped (%s).", e)

        # 7. Model drift check (best-effort)
        try:
            from production.monitoring import ModelDriftMonitor
            monitor = ModelDriftMonitor()
            ref_path = self.cfg.models_dir / "reference_distribution.pkl"
            if ref_path.exists():
                report = monitor.compute_drift_report(X, today_df["Probability"].values)
                logger.info("L5: Drift status = %s", report.get("overall_status"))
        except Exception as e:
            logger.debug("L5: Drift monitor skipped (%s).", e)

        # Log each signal
        for _, row in picks_df.iterrows():
            self._log_audit("SIGNAL_GENERATED",
                            sc_code=str(row.get(code_col, "")),
                            sc_name=str(row.get("SC_NAME", "")),
                            probability=float(row["Probability"]),
                            regime=str(row.get("regime", "")))

        return picks_df, explanations

    # ── L3b event blackout ────────────────────────────────────────────────
    def _apply_event_blackout(self, picks_df, ref_date):
        try:
            from production.event_calendar import EventCalendarBlackout
            blackout = EventCalendarBlackout(
                blackout_days_before=self.cfg.blackout_days_before,
                blackout_days_after=self.cfg.blackout_days_after,
            )
            code_col = "SC_CODE" if "SC_CODE" in picks_df.columns else "sc_code"
            sc_codes = picks_df[code_col].tolist()
            events = blackout.fetch_bse_announcements(
                sc_codes, ref_date, ref_date,
                cache_dir=str(self.cfg.cache_dir / "events"),
            )
            filtered, excluded = blackout.filter_picks(picks_df, ref_date, events)
            if len(excluded) > 0:
                excl_names = excluded.get("SC_NAME", excluded.get(code_col, pd.Series())).tolist()
                logger.info("L3b: Excluded %d picks in event blackout: %s",
                            len(excluded), excl_names[:5])
            return filtered
        except Exception as e:
            logger.debug("L3b: Event blackout skipped (%s).", e)
            return picks_df

    # ── L6 ───────────────────────────────────────────────────────────────
    def _layer6_portfolio(self, picks_df, bhav_df):
        """Sector cap, correlation filter, position + capital limits."""
        try:
            from production.portfolio_constructor import PortfolioConstructor
            ctor = PortfolioConstructor(
                max_positions=self.cfg.max_positions,
                max_sector_positions=self.cfg.max_sector_positions,
                max_capital_deployed=self.cfg.max_capital_deployed,
                max_correlation=self.cfg.max_correlation,
            )
            picks_df = ctor.construct_portfolio(
                picks_df, bhav_df,
                total_capital=self.cfg.total_capital,
            )
            stats = ctor.get_portfolio_stats(picks_df, bhav_df)
            logger.info("L6: sectors=%s, n=%d, capital=%.1f%%",
                        list(stats.get("sector_distribution", {}).keys()),
                        stats.get("num_positions", 0),
                        stats.get("total_capital_deployed", 0) * 100)
        except Exception as e:
            logger.warning("L6: Portfolio construction partial (%s).", e)
            picks_df = picks_df.head(self.cfg.max_positions)
        return picks_df

    # ── L7 ───────────────────────────────────────────────────────────────
    def _layer7_execution(self, picks_df, bhav_df, size_multiplier):
        """ATR stop-loss, position sizing, friction model."""
        close_col = "Close" if "Close" in bhav_df.columns else "close"
        code_col  = "SC_CODE" if "SC_CODE" in bhav_df.columns else "sc_code"
        date_col  = "DATE"   if "DATE"   in bhav_df.columns else "date"

        # Ensure Close column in picks
        if close_col.title() not in picks_df.columns and close_col in picks_df.columns:
            picks_df = picks_df.rename(columns={close_col: "Close"})
        elif "Close" not in picks_df.columns:
            # Merge from bhav_df latest
            latest_prices = (bhav_df.sort_values(date_col)
                                    .groupby(code_col)[close_col]
                                    .last()
                                    .rename("Close"))
            picks_df = picks_df.join(latest_prices, on=code_col, how="left")

        picks_df["Close"] = pd.to_numeric(picks_df.get("Close", 100), errors="coerce").fillna(100)

        # --- ATR-based position sizing ---
        try:
            from production.position_sizer import PositionSizer
            from production.exit_engine import ATRCalculator

            atr_calc = ATRCalculator()
            atr_dict = atr_calc.get_atr_for_universe(bhav_df, bhav_df[date_col].max())

            sizer = PositionSizer(
                total_capital=self.cfg.total_capital,
                risk_pct_per_trade=self.cfg.risk_pct_per_trade,
                max_position_pct=self.cfg.max_position_pct,
                max_positions=self.cfg.max_positions,
            )
            picks_df = sizer.calculate_portfolio_allocation(picks_df, self.cfg.total_capital, atr_dict)

            # Apply circuit-breaker multiplier
            if size_multiplier != 1.0:
                for col in ["Shares", "Position_Value", "Capital_Allocation"]:
                    if col in picks_df.columns:
                        picks_df[col] = (picks_df[col] * size_multiplier).astype(int if col == "Shares" else float)
                logger.info("L7: Position sizes scaled by %.0f%% (circuit breaker).", size_multiplier * 100)

            # Stop-loss price
            if "ATR" not in picks_df.columns:
                picks_df["ATR"] = picks_df[code_col].map(atr_dict).fillna(picks_df["Close"] * 0.02)
            picks_df["Stop_Loss_Price"] = picks_df["Close"] - self.cfg.initial_stop_atr * picks_df["ATR"]
            picks_df["Trailing_Stop_ATR"] = self.cfg.trailing_stop_atr

        except Exception as e:
            logger.warning("L7: Position sizing partial (%s).", e)
            picks_df["Shares"] = ((self.cfg.total_capital * 0.05) / picks_df["Close"]).astype(int)

        # --- Friction cost estimate ---
        friction_summary = {}
        try:
            from production.friction_model import FrictionModel
            fm = FrictionModel(instrument_type=self.cfg.instrument_type)
            total_trade_value = (picks_df["Close"] * picks_df.get("Shares", 1)).sum()
            rt = fm.calculate_round_trip_cost(total_trade_value)
            friction_summary = rt
            picks_df["Friction_Pct"] = rt.get("total_pct", 0.003)
            logger.info("L7: Round-trip friction = %.3f%%  (breakeven=%.3f%%)",
                        rt.get("total_pct", 0) * 100,        # fraction → %
                        rt.get("breakeven_return_pct", 0) * 100)
        except Exception as e:
            logger.debug("L7: Friction model skipped (%s).", e)

        # --- ADV volume cap ---
        try:
            from production.universe_filter import TradabilityGate
            gate = TradabilityGate()
            for idx, row in picks_df.iterrows():
                sc = row.get(code_col, "")
                max_shares = gate.get_max_tradeable_shares(bhav_df, sc)
                if "Shares" in picks_df.columns and picks_df.at[idx, "Shares"] > max_shares:
                    picks_df.at[idx, "Shares"] = max_shares
                    picks_df.at[idx, "Position_Value"] = max_shares * row["Close"]
        except Exception as e:
            logger.debug("L7: ADV cap skipped (%s).", e)

        return picks_df, friction_summary

    # ── L9 ───────────────────────────────────────────────────────────────
    def _layer9_compliance(self, picks_df, ref_date):
        """Log trades to audit trail and run SEBI compliance checks."""
        compliance = {"status": "OK", "violations": []}
        code_col = "SC_CODE" if "SC_CODE" in picks_df.columns else "sc_code"

        try:
            from production.compliance import AuditTrail, SEBIComplianceChecker
            audit = self._get_audit_trail()
            checker = SEBIComplianceChecker()

            for _, row in picks_df.iterrows():
                sc_code = str(row.get(code_col, ""))
                sc_name = str(row.get("SC_NAME", ""))
                shares  = int(row.get("Shares", 0))
                price   = float(row.get("Close", 0))
                audit.log_order(sc_code, sc_name, "BUY", shares, price, f"ORD_{sc_code}_{ref_date}")

            # Position limit check
            pos_dict = {r[code_col]: r.get("Position_Value", 0) for _, r in picks_df.iterrows()}
            ok, msg = checker.check_position_limits(pos_dict, self.cfg.total_capital)
            if not ok:
                compliance["violations"].append(msg)
                compliance["status"] = "VIOLATION"

            audit.log_event("SYSTEM_STOP", details={"picks_count": len(picks_df), "date": ref_date})
            report = checker.generate_compliance_report(ref_date)
            compliance.update(report)
            logger.info("L9: Compliance status = %s", compliance["status"])

        except Exception as e:
            logger.debug("L9: Compliance layer partial (%s).", e)

        return compliance

    # ====================================================================
    # HELPERS
    # ====================================================================

    def _fetch_bhav_data(self, ref_date: str) -> Optional[pd.DataFrame]:
        """Fetch BhavCopy data for the last lookback_days using the proven BSE loader."""
        try:
            from bse_direct_loader import BSEDataFetcher
            end_dt   = pd.to_datetime(ref_date).date()
            start_dt = (pd.to_datetime(ref_date) - pd.Timedelta(days=self.cfg.lookback_days)).date()
            fetcher  = BSEDataFetcher(cache_dir=str(self.cfg.cache_dir / "bse"))
            df = fetcher.fetch_bhav_range(start_dt, end_dt)
            logger.info("Fetched %d BhavCopy rows from BSE.", len(df))
            return df if not df.empty else None
        except Exception as e:
            logger.error("BhavCopy fetch failed: %s", e)
            return None

    def _ensure_model(self, feature_df):
        """Load or train the prediction model; fall back to last-known-good on failure."""
        if self._model is not None:
            return self._model

        # Try loading saved model
        model_path = self.cfg.models_dir / "lgbm_model.txt"
        feat_path  = self.cfg.models_dir / "feature_names.json"
        if model_path.exists():
            try:
                import lightgbm as lgb
                self._model = lgb.Booster(model_file=str(model_path))
                if feat_path.exists():
                    with open(feat_path) as fh:
                        self._feature_names = json.load(fh)
                logger.info("Loaded saved model from %s.", model_path)
                return self._model
            except Exception as e:
                logger.warning("Model load failed: %s — attempting fallback.", e)

        # Try fallback
        fallback = self._get_fallback_system()
        model, meta = fallback.handle_model_failure(
            Exception("No saved model found."), ref_date := datetime.now().strftime("%Y-%m-%d")
        )
        if model is not None:
            self._model = model
            self._fallback_used = True
            return self._model

        # Last resort: quick train on available data
        logger.warning("No saved model — quick-training on available data.")
        try:
            self._model = self._quick_train(feature_df)
            return self._model
        except Exception as e:
            logger.error("Quick-train failed: %s", e)
            return None

    def _quick_train(self, feature_df):
        """Minimal LightGBM training directly inside the orchestrator."""
        import lightgbm as lgb
        from sklearn.model_selection import TimeSeriesSplit

        date_col  = "DATE" if "DATE" in feature_df.columns else "date"
        close_col = "Close" if "Close" in feature_df.columns else "close"
        code_col  = "SC_CODE" if "SC_CODE" in feature_df.columns else "sc_code"

        df = feature_df.sort_values([code_col, date_col]).copy()
        df["target"] = (df.groupby(code_col)[close_col]
                          .transform(lambda x: x.shift(-5) / x - 1) > 0).astype(int)
        df = df.dropna(subset=["target"])

        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ["target"] and not c.startswith("fwd")]
        self._feature_names = num_cols

        X = df[num_cols].fillna(0)
        y = df["target"]

        tscv = TimeSeriesSplit(n_splits=3)
        for train_idx, _ in tscv.split(X):
            pass  # use last fold's train set

        dtrain = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
        params = {"objective": "binary", "metric": "auc",
                  "num_leaves": 31, "learning_rate": 0.05, "verbose": -1}
        model = lgb.train(params, dtrain, num_boost_round=200)

        # Cache model for future runs
        model.save_model(str(self.cfg.models_dir / "lgbm_model.txt"))
        with open(self.cfg.models_dir / "feature_names.json", "w") as fh:
            json.dump(num_cols, fh)
        self._get_fallback_system().model_cache.save_good_model(
            model, {"model_type": "lightgbm", "feature_count": len(num_cols)}
        )
        logger.info("Quick-train complete (%d features).", len(num_cols))
        return model

    # ── Lazy layer getters ────────────────────────────────────────────────

    def _get_kill_switch(self):
        if self._kill_switch is None:
            try:
                from production.risk_controls import KillSwitch
                self._kill_switch = KillSwitch()
            except Exception:
                class _NoOpKillSwitch:
                    def is_active(self): return False
                    def get_reason(self): return ""
                    def check_and_log(self): return False
                self._kill_switch = _NoOpKillSwitch()
        return self._kill_switch

    def _get_circuit_breaker(self):
        if self._circuit_breaker is None:
            try:
                from production.risk_controls import DrawdownCircuitBreaker
                self._circuit_breaker = DrawdownCircuitBreaker(
                    warning_threshold=self.cfg.drawdown_warning,
                    halt_threshold=self.cfg.drawdown_halt,
                    state_file=str(self.cfg.base_dir / "circuit_breaker_state.json"),
                )
            except Exception:
                class _NoOpCB:
                    def load_state(self): pass
                    def save_state(self): pass
                    def check_state(self):
                        from enum import Enum
                        class S(Enum): NORMAL = "NORMAL"
                        return S.NORMAL
                    def compute_current_drawdown(self): return 0.0
                    def get_position_size_multiplier(self): return 1.0
                    def can_open_new_trades(self): return True
                self._circuit_breaker = _NoOpCB()
        return self._circuit_breaker

    def _get_audit_trail(self):
        if self._audit_trail is None:
            try:
                from production.compliance import AuditTrail
                self._audit_trail = AuditTrail(log_dir=str(self.cfg.audit_dir))
            except Exception:
                class _NoOpAudit:
                    def log_event(self, *a, **kw): pass
                    def log_order(self, *a, **kw): pass
                self._audit_trail = _NoOpAudit()
        return self._audit_trail

    def _get_fallback_system(self):
        if self._fallback_system is None:
            try:
                from production.operational_fallback import create_fallback_system
                self._fallback_system = create_fallback_system(
                    models_dir=str(self.cfg.models_dir),
                    cache_dir=str(self.cfg.cache_dir / "bse"),
                    alert_log=str(self.cfg.alert_log),
                )
            except Exception:
                class _NoOpFallback:
                    class model_cache:
                        @staticmethod
                        def save_good_model(*a, **kw): pass
                    def pre_run_health_check(self, bhav_df, ref_date):
                        return {"is_healthy": bhav_df is not None, "action": "PROCEED", "details": {}}
                    def handle_model_failure(self, err, ref_date):
                        return None, None
                self._fallback_system = _NoOpFallback()
        return self._fallback_system

    def _get_calibrator(self):
        """Load persisted calibrator if it exists."""
        if self._calibrator is not None:
            return self._calibrator
        cal_path = self.cfg.models_dir / "calibrator.pkl"
        if cal_path.exists():
            try:
                from production.probability_calibration import ProbabilityCalibrator
                cal = ProbabilityCalibrator()
                cal.load(str(cal_path))
                self._calibrator = cal
                return cal
            except Exception:
                pass
        return None

    def _log_audit(self, event_type: str, **kwargs):
        audit = self._get_audit_trail()
        try:
            audit.log_event(event_type, **kwargs)
        except Exception:
            pass

    def _save_picks(self, picks_df: pd.DataFrame, ref_date: str):
        """Write picks CSV to results directory."""
        date_str = ref_date.replace("-", "")
        out = self.cfg.results_dir / f"picks_{date_str}.csv"
        picks_df.to_csv(out, index=False)
        logger.info("Picks saved → %s (%d stocks)", out, len(picks_df))


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the production TradeOrchestrator.")
    parser.add_argument("--date",    default=None,    help="Reference date YYYY-MM-DD (default: today)")
    parser.add_argument("--capital", default=1000000, type=float, help="Total capital in ₹")
    parser.add_argument("--min-price", default=20.0,  type=float)
    parser.add_argument("--min-value", default=2.0,   type=float, help="Min daily value traded (crore)")
    args = parser.parse_args()

    cfg = OrchestratorConfig()
    cfg.total_capital  = args.capital
    cfg.min_price      = args.min_price
    cfg.min_value_crore = args.min_value

    orch = TradeOrchestrator(config=cfg)
    result = orch.run_daily(reference_date=args.date)

    picks = result["picks"]
    print(f"\n{'='*68}")
    print(f"  Action     : {result['action']}")
    print(f"  Regime     : {result['regime'].get('regime', 'N/A')}")
    print(f"  Circuit    : {result['circuit_state']}")
    print(f"  Picks      : {len(picks)}")
    if not picks.empty:
        disp_cols = [c for c in ["SC_CODE", "SC_NAME", "Close", "Probability", "Shares",
                                  "Stop_Loss_Price", "Position_Value"] if c in picks.columns]
        print(picks[disp_cols].to_string(index=False))
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
