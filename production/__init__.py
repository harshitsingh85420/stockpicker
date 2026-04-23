"""
production — complete 9-layer production trading system for Indian equities.

Layers implemented
──────────────────
1  Data Integrity          data_integrity.py          (Gaps 1)
2  Universe & Tradability  universe_filter.py          (Gap 2)
3  Context & Events        regime_filter.py            (Gaps 3, 14)
                           event_calendar.py           (Gap 4)
4  Feature Engineering     (existing momentum_features + new modules)
                           feature_leak_audit.py       (Gap 12)
5  Signal & Calibration    probability_calibration.py  (Gap 6)
                           shap_explainability.py      (Gap 15)
                           monitoring.py               (Gap 13)
6  Portfolio Construction  portfolio_constructor.py    (Gaps 9, 16)
7  Execution & Sizing      exit_engine.py              (Gap 7)
                           position_sizer.py           (Gap 8)
                           friction_model.py           (Gaps 10, 20)
8  Risk & Failsafe         risk_controls.py            (Gaps 11, 18)
                           operational_fallback.py     (Gap 17)
9  Compliance & Monitoring compliance.py               (Gap 19)

Entry point
───────────
    from production.trade_orchestrator import TradeOrchestrator
    orch = TradeOrchestrator()
    picks = orch.run_daily()
"""

__version__ = "1.0.0"
__author__ = "stockpicker production team"
