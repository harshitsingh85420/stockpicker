# 5-Session Stock Picker

**ML-powered stock prediction system for Indian markets (BSE). Predicts stocks likely to close positive after 5 trading sessions.**

## Quick Start

```bash
pip install -r requirements-pinned.txt
python run_full_cycle.py
```

**First run**: 15–20 minutes (downloads data, trains model)  
**Subsequent runs**: 2–3 minutes (uses cache)

---

## Walk-Forward Backtest Results (as of April 2026)

| Metric | Value |
|--------|-------|
| Gross win rate | 58.1% (3,551 picks, 4 OOS blocks) |
| Net win rate after friction | 45.8% |
| Average net 5-day return | +0.69% |
| Median net return | -0.34% |
| CV AUC | 0.593 |

> These are **out-of-sample** walk-forward results — each block's model was trained on data strictly before that block. No look-ahead.

---

## Universe & Filters (Honest Numbers)

BSE has ~9,200 listed scrips. The production pipeline applies these liquidity gates before prediction:

| Filter | Value | Reason |
|--------|-------|--------|
| Min price | ₹20 | Sub-₹20 stocks are typically penny/suspended |
| Min avg daily volume | 10,000 shares | Minimum executability |
| Min traded value | ₹2 crore/day | Avoids illiquid traps |
| ADV participation cap | ≤1% ADV per order | Realistic fill assumption |

After these gates ~600–800 stocks are scored on any given day. All stocks above the ML probability threshold are shown — no top-N cap.

---

## Architecture

- 9-layer production pipeline in `production/`
- LightGBM + XGBoost ensemble (60/40 weighted, high-conviction filter)
- 3-layer EMA regime gate (swing/medium/long) + 2-state HMM + India VIX gate
- ATR-based position sizing with half-Kelly
- Walk-forward OOS validation (no look-ahead)
- Historical VaR + stress tests via `risk_analytics.py`
- Telegram alerting + kill switch via `alerting.py`

---

## How It Works

```
BSE BhavCopy (Official Data)
    ↓
Universe filter: ~600–800 tradable stocks
    ↓
Compute 36 features (EMA, ATR, RSI, ADX, FracDiff, FII/DII …)
    ↓
Regime gate: 3-layer EMA + 2-state HMM + VIX
    ↓
LightGBM × 0.60 + XGBoost × 0.40 ensemble
    ↓
High-conviction filter: both models must exceed 0.55
    ↓
Portfolio construction: sector cap, correlation gate, ADV cap
    ↓
ATR stop-loss sizing + friction model
    ↓
Output CSV + Telegram alert
```

---

## Installation

### Requirements

- Python 3.11+
- 8 GB RAM (16 GB recommended for full BSE universe)
- Internet connection

```bash
pip install -r requirements-pinned.txt
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-pinned.txt
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-pinned.txt
```

---

## Daily Usage

```bash
# After market close each day
python run_full_cycle.py

# Output: stock_picker_data/results/picks_YYYYMMDD.csv
```

**Output columns:**
`Rank, SC_CODE, SC_NAME, Close, Probability, ATR14, Stop_Loss_Price, Shares, Position_Value, Sector, Regime, Net_Expected_Return, Rationale`

---

## Walk-Forward Backtest

```bash
python production/walk_forward_backtest.py --months-per-block 3
```

Each block trains on data before the block and tests on the block — genuinely out-of-sample.

---

## Project Structure

```
stockpicker/
├── run_full_cycle.py              # Main entry: train → picks → report
├── train_model.py                 # Standalone model training
├── momentum_features.py           # Feature engineering (36 features)
├── requirements-pinned.txt        # Pinned production dependencies
├── Dockerfile                     # Docker deployment
├── exit_all_positions.py          # Emergency kill switch
├── production/
│   ├── trade_orchestrator.py      # 9-layer pipeline orchestrator
│   ├── signal_generator.py        # ML signal generation
│   ├── regime_filter.py           # EMA + HMM + VIX regime gate
│   ├── universe_filter.py         # Liquidity & tradability gates
│   ├── risk_analytics.py          # VaR, stress tests, correlation gate
│   ├── alerting.py                # Telegram alerts + kill switch
│   ├── broker_adapter.py          # Zerodha Kite Connect (paper/live)
│   ├── walk_forward_backtest.py   # OOS walk-forward validation
│   ├── historical_effectiveness.py# In-sample retrospective report
│   ├── benchmark.py               # Benchmark comparison (P44)
│   └── …                          # 16 more modules
├── tests/                         # pytest suite (5 modules)
└── stock_picker_data/
    ├── cache/                     # BSE data cache
    ├── models/                    # Trained models + feature list
    └── results/                   # Daily picks + summaries
```

---

## Configuration

Edit `OrchestratorConfig` in `production/trade_orchestrator.py`:

```python
self.total_capital       = 1_000_000   # Rs. 10 lakh default
self.risk_pct_per_trade  = 0.01        # 1% capital at risk per trade
self.max_positions       = 15
self.max_capital_deployed = 0.70       # Deploy ≤70% of capital
self.base_threshold      = 0.62        # ML probability threshold
```

---

## Telegram Alerts (Optional)

Set in `.env`:

```
TELEGRAM_BOT_TOKEN=<your bot token>
TELEGRAM_CHAT_ID=<your chat id>
```

Alerts fire on: picks ready, circuit breaker warning/halt, data source failure, kill switch activation.

---

## Zerodha Live Trading (Optional)

Set in `.env`:

```
KITE_API_KEY=<your api key>
KITE_ACCESS_TOKEN=<your daily access token>
ALGO_ID=STOCKPICKER_V1
```

Default is **paper mode** (`paper_mode=True`). Real orders only when you explicitly pass `paper_mode=False` to `BrokerAdapter`.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Running with Docker

```bash
docker build -t stockpicker .
docker run --env-file .env -v $(pwd)/stock_picker_data:/app/stock_picker_data stockpicker
```

---

## Performance Expectations

| Scenario | Expectation |
|----------|------------|
| Win rate (gross OOS) | ~58% |
| Win rate (net after friction) | ~46% |
| Average net 5-day return | ~+0.7% |
| Median net return | ~-0.3% |

Past OOS performance does not guarantee future results. Run at least 3 months of paper trading before committing real capital.

---

## Risk Disclaimer

**FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY**

- Backtest results are walk-forward OOS but still represent historical performance
- No guarantee of future profitability
- Always validate with paper trading before live deployment
- Consult a SEBI-registered financial advisor before trading

---

## License

MIT License — Educational and research use only.

---

*Built with Claude Code.*
