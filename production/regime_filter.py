"""
regime_filter.py
================
Gap 3 — Market Regime Awareness
Gap 14 — Advanced HMM Regime Detection

Two-layer regime system for Indian equity markets:

1. ``IndexRegimeFilter``  — fast, always-on EMA-based filter using Nifty 50.
   No external library dependencies beyond yfinance + pandas/numpy.

2. ``HMMRegimeDetector`` — 4-state Hidden Markov Model that classifies the
   market into BULL_TRENDING, BEAR_TRENDING, HIGH_VOLATILITY, or SIDEWAYS.
   Requires hmmlearn; degrades gracefully if not installed.

3. ``get_combined_regime()`` — module-level helper that runs both layers and
   returns a single consolidated assessment dict.

Usage
-----
>>> from production.regime_filter import get_combined_regime
>>> ctx = get_combined_regime()
>>> print(ctx["combined_label"], ctx["is_tradeable"])
"""

from __future__ import annotations

import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: hmmlearn
# ---------------------------------------------------------------------------
try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore
    HMM_AVAILABLE = True
    logger.debug("hmmlearn loaded successfully.")
except ImportError:
    HMM_AVAILABLE = False
    logger.warning(
        "hmmlearn not available — HMMRegimeDetector will use fallback logic. "
        "Install with: pip install hmmlearn"
    )

# ---------------------------------------------------------------------------
# Optional dependency: yfinance
# ---------------------------------------------------------------------------
try:
    import yfinance as yf  # type: ignore
    YFINANCE_AVAILABLE = True
    logger.debug("yfinance loaded successfully.")
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning(
        "yfinance not available — fetch_nifty_data will use synthetic fallback. "
        "Install with: pip install yfinance"
    )


# ===========================================================================
# Constants
# ===========================================================================
NIFTY_TICKER = "^NSEI"

# Regime labels (canonical strings used across both classes)
BULL_TRENDING = "BULL_TRENDING"
BEAR_TRENDING = "BEAR_TRENDING"
HIGH_VOLATILITY = "HIGH_VOLATILITY"
SIDEWAYS = "SIDEWAYS"


# ===========================================================================
# 1.  IndexRegimeFilter  (EMA-based, always-on)
# ===========================================================================

class IndexRegimeFilter:
    """
    Simple, fast market-regime filter based on Nifty 50 EMA crossover.

    Classifies the market into three regimes:
    - BULL    : close > EMA50 > EMA200
    - BEAR    : close < EMA50 < EMA200
    - SIDEWAYS: mixed / transitional

    The filter also returns a *probability threshold* that the downstream
    model should use when picking stocks — higher threshold in hostile regimes,
    lower in benign ones.

    Parameters
    ----------
    fast_ema : int
        Period for the fast EMA (default 50).
    slow_ema : int
        Period for the slow EMA (default 200).
    bearish_threshold_override : float
        Probability threshold to require in BEAR regime.  Should be higher
        than the base threshold to enforce a stricter quality bar.
    """

    def __init__(
        self,
        fast_ema: int = 50,
        slow_ema: int = 200,
        bearish_threshold_override: float = 0.72,
    ) -> None:
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.bearish_threshold_override = bearish_threshold_override
        logger.info(
            "IndexRegimeFilter init: fast_ema=%d, slow_ema=%d, "
            "bearish_override=%.2f",
            fast_ema, slow_ema, bearish_threshold_override,
        )

    # ------------------------------------------------------------------
    # Data acquisition
    # ------------------------------------------------------------------

    def fetch_nifty_data(self, lookback_days: int = 250) -> pd.DataFrame:
        """
        Download Nifty 50 daily OHLCV data.

        Tries yfinance first; falls back to synthetic data if unavailable or
        the download fails.

        Parameters
        ----------
        lookback_days : int
            Number of calendar days to look back (default 250 ≈ 1 trading year).

        Returns
        -------
        pd.DataFrame
            Columns: DATE (datetime), Close (float).
            Sorted ascending by DATE.
        """
        end_date = datetime.today()
        start_date = end_date - timedelta(days=lookback_days)

        if YFINANCE_AVAILABLE:
            try:
                logger.info(
                    "Fetching Nifty data from yfinance (%s → %s) …",
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                )
                ticker = yf.Ticker(NIFTY_TICKER)
                raw = ticker.history(
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    interval="1d",
                )
                if raw.empty:
                    raise ValueError("Empty response from yfinance.")

                raw = raw.reset_index()
                # yfinance may return 'Date' or 'Datetime'
                date_col = "Date" if "Date" in raw.columns else "Datetime"
                df = pd.DataFrame(
                    {
                        "DATE": pd.to_datetime(raw[date_col]).dt.normalize(),
                        "Close": raw["Close"].astype(float),
                    }
                )
                df = df.dropna().sort_values("DATE").reset_index(drop=True)
                logger.info(
                    "Nifty data fetched: %d rows (%s → %s).",
                    len(df),
                    df["DATE"].iloc[0].date(),
                    df["DATE"].iloc[-1].date(),
                )
                return df

            except Exception as exc:
                logger.warning(
                    "yfinance fetch failed (%s) — using synthetic data.", exc
                )

        # Fallback: deterministic synthetic Nifty series (random-walk seeded)
        return self._synthetic_nifty(lookback_days)

    def _synthetic_nifty(self, lookback_days: int) -> pd.DataFrame:
        """Generate a synthetic Nifty series for offline / testing use."""
        logger.warning("Using synthetic Nifty data — results are illustrative only.")
        rng = np.random.default_rng(seed=42)
        dates = pd.bdate_range(
            end=datetime.today(), periods=min(lookback_days, 260)
        )
        # Start near 22 000 with realistic daily drift/vol
        log_returns = rng.normal(loc=0.0004, scale=0.0095, size=len(dates))
        prices = 22_000.0 * np.exp(np.cumsum(log_returns))
        return pd.DataFrame({"DATE": dates, "Close": prices})

    # ------------------------------------------------------------------
    # EMA computation
    # ------------------------------------------------------------------

    def compute_emas(self, nifty_df: pd.DataFrame) -> pd.DataFrame:
        """
        Append EMA50 and EMA200 columns to *nifty_df*.

        Parameters
        ----------
        nifty_df : pd.DataFrame
            Must contain a ``Close`` column.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with ``EMA{fast}`` and ``EMA{slow}`` columns added.
        """
        df = nifty_df.copy()
        df[f"EMA{self.fast_ema}"] = (
            df["Close"].ewm(span=self.fast_ema, adjust=False).mean()
        )
        df[f"EMA{self.slow_ema}"] = (
            df["Close"].ewm(span=self.slow_ema, adjust=False).mean()
        )
        logger.debug(
            "EMAs computed: EMA%d=%.2f, EMA%d=%.2f (latest row).",
            self.fast_ema, df[f"EMA{self.fast_ema}"].iloc[-1],
            self.slow_ema, df[f"EMA{self.slow_ema}"].iloc[-1],
        )
        return df

    # ------------------------------------------------------------------
    # Regime classification
    # ------------------------------------------------------------------

    def get_regime(self, nifty_df: pd.DataFrame) -> str:
        """
        Classify the current market regime from the latest Nifty row.

        Decision rules (in order):
        - BULL    : close > EMA50 > EMA200
        - BEAR    : close < EMA50 < EMA200
        - SIDEWAYS: everything else (mixed / crossing)

        Parameters
        ----------
        nifty_df : pd.DataFrame
            Must have Close, EMA{fast}, EMA{slow} columns.

        Returns
        -------
        str
            'BULL', 'BEAR', or 'SIDEWAYS'.
        """
        ema_fast_col = f"EMA{self.fast_ema}"
        ema_slow_col = f"EMA{self.slow_ema}"

        for col in ("Close", ema_fast_col, ema_slow_col):
            if col not in nifty_df.columns:
                raise KeyError(
                    f"Column '{col}' missing — call compute_emas() first."
                )

        latest = nifty_df.iloc[-1]
        close = float(latest["Close"])
        ema_fast = float(latest[ema_fast_col])
        ema_slow = float(latest[ema_slow_col])

        if close > ema_fast > ema_slow:
            regime = "BULL"
        elif close < ema_fast < ema_slow:
            regime = "BEAR"
        else:
            regime = "SIDEWAYS"

        logger.info(
            "Regime: %s | Close=%.2f, EMA%d=%.2f, EMA%d=%.2f",
            regime, close, self.fast_ema, ema_fast, self.slow_ema, ema_slow,
        )
        return regime

    # ------------------------------------------------------------------
    # Probability threshold
    # ------------------------------------------------------------------

    def get_probability_threshold(
        self, regime: str, base_threshold: float = 0.62
    ) -> float:
        """
        Return the minimum model-probability required to accept a pick,
        adjusted for the current regime.

        Parameters
        ----------
        regime : str
            'BULL', 'BEAR', or 'SIDEWAYS'.
        base_threshold : float
            Baseline probability threshold (default 0.62).

        Returns
        -------
        float
            Adjusted probability threshold.
        """
        mapping = {
            "BULL": base_threshold,                        # relaxed in bull
            "BEAR": self.bearish_threshold_override,       # strict in bear
            "SIDEWAYS": round(base_threshold * 1.05, 4),  # slightly tighter
        }
        threshold = mapping.get(regime, base_threshold)
        logger.info(
            "Probability threshold for regime=%s: %.4f", regime, threshold
        )
        return threshold

    # ------------------------------------------------------------------
    # Tradability gate
    # ------------------------------------------------------------------

    def is_tradeable_regime(self, regime: str) -> bool:
        """
        Determine whether the strategy should take new positions.

        Returns True for BULL and SIDEWAYS; False for BEAR.

        Parameters
        ----------
        regime : str

        Returns
        -------
        bool
        """
        tradeable = regime in ("BULL", "SIDEWAYS")
        logger.info("is_tradeable_regime(%s) = %s", regime, tradeable)
        return tradeable

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------

    def get_regime_report(self, nifty_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Build a consolidated regime assessment dict from the latest Nifty data.

        Parameters
        ----------
        nifty_df : pd.DataFrame
            Must have Close, EMA{fast}, EMA{slow} columns (use compute_emas).

        Returns
        -------
        dict with keys:
            regime, nifty_close, ema50, ema200, trend_strength,
            is_tradeable, recommended_threshold
        """
        ema_fast_col = f"EMA{self.fast_ema}"
        ema_slow_col = f"EMA{self.slow_ema}"

        latest = nifty_df.iloc[-1]
        close = float(latest["Close"])
        ema_fast = float(latest[ema_fast_col])
        ema_slow = float(latest[ema_slow_col])

        regime = self.get_regime(nifty_df)

        # Trend strength: normalised distance between fast and slow EMA
        # Positive = fast above slow (bullish alignment), negative = bearish
        trend_strength = (ema_fast - ema_slow) / ema_slow if ema_slow != 0 else 0.0

        report: Dict[str, Any] = {
            "regime": regime,
            "nifty_close": round(close, 2),
            "ema50": round(ema_fast, 2),
            "ema200": round(ema_slow, 2),
            "trend_strength": round(trend_strength, 6),
            "is_tradeable": self.is_tradeable_regime(regime),
            "recommended_threshold": self.get_probability_threshold(regime),
        }
        logger.info("Regime report: %s", report)
        return report


# ===========================================================================
# 2.  HMMRegimeDetector  (4-state Hidden Markov Model)
# ===========================================================================

class HMMRegimeDetector:
    """
    4-state Hidden Markov Model for granular market regime detection.

    States (labels assigned post-fit based on regime statistics):
    - BULL_TRENDING   : high positive mean return, low-moderate volatility
    - BEAR_TRENDING   : negative mean return
    - HIGH_VOLATILITY : large absolute swings regardless of direction
    - SIDEWAYS        : near-zero return, low volatility

    Parameters
    ----------
    n_states : int
        Number of HMM states (default 4).
    feature_window : int
        Rolling window (in days) used to compute volatility and z-score
        features (default 20).
    """

    # Mean-return thresholds for regime labelling
    BULL_RETURN_THRESHOLD = 0.0005
    BEAR_RETURN_THRESHOLD = -0.0005
    HIGH_VOL_THRESHOLD = 0.015

    def __init__(self, n_states: int = 4, feature_window: int = 20) -> None:
        self.n_states = n_states
        self.feature_window = feature_window
        self.model: Optional[Any] = None
        self.regime_stats: Optional[Dict[int, Dict]] = None
        self._label_cache: Dict[int, str] = {}
        logger.info(
            "HMMRegimeDetector init: n_states=%d, feature_window=%d",
            n_states, feature_window,
        )

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def prepare_features(
        self,
        returns_series: pd.Series,
        volume_series: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """
        Build a feature matrix for HMM fitting / prediction.

        Features (columns):
        0. Daily return
        1. 5-day rolling volatility
        2. 20-day rolling volatility  (= feature_window)
        3. 20-day return z-score
        [4. Volume change — only if volume_series is provided]

        Parameters
        ----------
        returns_series : pd.Series
            Daily returns (fraction, not percent).
        volume_series : pd.Series, optional
            Daily volume; used to add a volume-change feature.

        Returns
        -------
        np.ndarray, shape (n_rows, n_features)
            Rows with any NaN dropped.  The row count may be slightly less
            than len(returns_series) due to rolling window warm-up.
        """
        df = pd.DataFrame({"ret": returns_series})

        # Rolling volatility
        df["vol_5d"] = df["ret"].rolling(5).std()
        df["vol_20d"] = df["ret"].rolling(self.feature_window).std()

        # Z-score of return relative to trailing 20-day distribution
        roll_mean = df["ret"].rolling(self.feature_window).mean()
        roll_std = df["ret"].rolling(self.feature_window).std().replace(0, np.nan)
        df["ret_z_20d"] = (df["ret"] - roll_mean) / roll_std

        feature_cols = ["ret", "vol_5d", "vol_20d", "ret_z_20d"]

        if volume_series is not None:
            vol_df = volume_series.rename("volume")
            df = df.join(vol_df, how="left")
            df["vol_change"] = df["volume"].pct_change().fillna(0)
            feature_cols.append("vol_change")

        df = df[feature_cols].dropna()
        X = df.values.astype(float)
        logger.debug(
            "prepare_features: %d rows x %d features.", X.shape[0], X.shape[1]
        )
        return X

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        returns_series: pd.Series,
        volume_series: Optional[pd.Series] = None,
    ) -> None:
        """
        Fit the HMM on historical return data.

        Falls back to a simple k-means-style state assignment when hmmlearn
        is not installed so that downstream code keeps working.

        Parameters
        ----------
        returns_series : pd.Series
            Daily returns.
        volume_series : pd.Series, optional
            Daily volumes for extra features.
        """
        X = self.prepare_features(returns_series, volume_series)
        n_obs = len(X)

        if HMM_AVAILABLE:
            logger.info(
                "Fitting GaussianHMM (n_components=%d) on %d observations …",
                self.n_states, n_obs,
            )
            hmm_fitted = False
            # Try progressively larger regularisation if covariance is singular
            for reg_covar in (1e-3, 1e-2, 1e-1):
                try:
                    self.model = GaussianHMM(
                        n_components=self.n_states,
                        covariance_type="full",
                        n_iter=1000,
                        random_state=42,
                    )
                    self.model.fit(X)
                    state_seq = self.model.predict(X)
                    hmm_fitted = True
                    break
                except Exception as hmm_exc:
                    logger.warning(
                        "GaussianHMM fit failed (reg_covar=%.0e): %s — retrying "
                        "with diagonal covariance …", reg_covar, hmm_exc
                    )
                    try:
                        self.model = GaussianHMM(
                            n_components=self.n_states,
                            covariance_type="diag",
                            n_iter=1000,
                            random_state=42,
                        )
                        self.model.fit(X)
                        state_seq = self.model.predict(X)
                        hmm_fitted = True
                        break
                    except Exception as diag_exc:
                        logger.warning(
                            "GaussianHMM diag fit also failed: %s", diag_exc
                        )

            if not hmm_fitted:
                logger.warning(
                    "All HMM fit attempts failed — falling back to quantile method."
                )
                self.model = None
                state_seq = self._quantile_fallback(X[:, 0])
        else:
            logger.warning(
                "hmmlearn unavailable — using quantile-based fallback for fit()."
            )
            state_seq = self._quantile_fallback(X[:, 0])  # use raw return col

        # Compute per-state statistics from the training data
        ret_col = X[:, 0]
        vol_col = X[:, 2]  # vol_20d
        self.regime_stats = {}
        for s in range(self.n_states):
            mask = state_seq == s
            if mask.sum() == 0:
                self.regime_stats[s] = {
                    "mean_return": 0.0,
                    "std_return": 0.0,
                    "frequency": 0.0,
                }
                continue
            self.regime_stats[s] = {
                "mean_return": float(ret_col[mask].mean()),
                "std_return": float(vol_col[mask].mean()),
                "frequency": float(mask.sum()) / len(state_seq),
            }

        # Pre-populate label cache
        self._label_cache = {
            s: self.label_regime(s) for s in range(self.n_states)
        }
        logger.info("HMM fit complete. State labels: %s", self._label_cache)

    # ------------------------------------------------------------------
    # Fallback (no hmmlearn)
    # ------------------------------------------------------------------

    def _quantile_fallback(self, returns: np.ndarray) -> np.ndarray:
        """
        Assign states by quantile buckets when hmmlearn is not available.
        This is a crude approximation — install hmmlearn for proper HMM.
        """
        q25, q50, q75 = np.percentile(returns, [25, 50, 75])
        states = np.zeros(len(returns), dtype=int)
        states[returns >= q75] = 0   # high return → state 0 (bull-ish)
        states[(returns >= q50) & (returns < q75)] = 3  # moderate → sideways
        states[(returns >= q25) & (returns < q50)] = 3
        states[returns < q25] = 1   # low return → bear-ish
        # assign high-vol state (2) based on absolute return magnitude
        abs_r = np.abs(returns)
        high_vol_thresh = np.percentile(abs_r, 85)
        states[abs_r >= high_vol_thresh] = 2
        return states

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_current_regime(
        self,
        returns_series: pd.Series,
        volume_series: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Predict the most likely regime for the current observation.

        Parameters
        ----------
        returns_series : pd.Series
            Recent daily returns (at least ``feature_window`` rows required).
        volume_series : pd.Series, optional

        Returns
        -------
        dict with keys:
            regime_id (int), regime_label (str), confidence (float),
            regime_probs (list[float] — posterior probability for each state)
        """
        if self.model is None and self.regime_stats is None:
            raise RuntimeError("Call fit() before predict_current_regime().")

        X = self.prepare_features(returns_series, volume_series)
        if len(X) == 0:
            raise ValueError(
                "Not enough data to compute features — need at least "
                f"{self.feature_window + 5} observations."
            )

        if HMM_AVAILABLE and self.model is not None:
            state_seq = self.model.predict(X)
            # Posterior state probabilities for the last observation
            posteriors = self.model.predict_proba(X)
            last_probs = posteriors[-1].tolist()
            regime_id = int(state_seq[-1])
        else:
            # Fallback: use last feature row and compute nearest state
            state_seq = self._quantile_fallback(X[:, 0])
            regime_id = int(state_seq[-1])
            # Uniform probability with winner boosted
            last_probs = [0.05] * self.n_states
            last_probs[regime_id] = 1.0 - 0.05 * (self.n_states - 1)

        regime_label = self._label_cache.get(regime_id, self.label_regime(regime_id))
        confidence = float(max(last_probs))

        result = {
            "regime_id": regime_id,
            "regime_label": regime_label,
            "confidence": round(confidence, 4),
            "regime_probs": [round(p, 4) for p in last_probs],
        }
        logger.info("HMM prediction: %s", result)
        return result

    # ------------------------------------------------------------------
    # Regime labelling
    # ------------------------------------------------------------------

    def label_regime(self, regime_id: int) -> str:
        """
        Assign a human-readable label to a state based on its training statistics.

        Classification rules (applied in order):
        1. mean_return > BULL_RETURN_THRESHOLD  → BULL_TRENDING
        2. mean_return < BEAR_RETURN_THRESHOLD  → BEAR_TRENDING
        3. std_return  > HIGH_VOL_THRESHOLD     → HIGH_VOLATILITY
        4. Otherwise                            → SIDEWAYS

        Parameters
        ----------
        regime_id : int

        Returns
        -------
        str
        """
        if self.regime_stats is None or regime_id not in self.regime_stats:
            return SIDEWAYS

        stats = self.regime_stats[regime_id]
        mean_ret = stats["mean_return"]
        std_ret = stats["std_return"]

        if mean_ret > self.BULL_RETURN_THRESHOLD:
            return BULL_TRENDING
        if mean_ret < self.BEAR_RETURN_THRESHOLD:
            return BEAR_TRENDING
        if std_ret > self.HIGH_VOL_THRESHOLD:
            return HIGH_VOLATILITY
        return SIDEWAYS

    # ------------------------------------------------------------------
    # Regime-adjusted trading parameters
    # ------------------------------------------------------------------

    def get_regime_adjusted_params(self, regime_label: str) -> Dict[str, Any]:
        """
        Return risk and position parameters tuned for the given regime.

        Parameters
        ----------
        regime_label : str
            One of BULL_TRENDING, BEAR_TRENDING, HIGH_VOLATILITY, SIDEWAYS.

        Returns
        -------
        dict with keys:
            probability_threshold (float),
            position_size_multiplier (float),
            max_positions (int)
        """
        params_map: Dict[str, Dict[str, Any]] = {
            BULL_TRENDING: {
                "probability_threshold": 0.60,
                "position_size_multiplier": 1.20,
                "max_positions": 15,
            },
            BEAR_TRENDING: {
                "probability_threshold": 0.72,
                "position_size_multiplier": 0.60,
                "max_positions": 8,
            },
            HIGH_VOLATILITY: {
                "probability_threshold": 0.68,
                "position_size_multiplier": 0.70,
                "max_positions": 10,
            },
            SIDEWAYS: {
                "probability_threshold": 0.64,
                "position_size_multiplier": 1.00,
                "max_positions": 12,
            },
        }
        params = params_map.get(regime_label, params_map[SIDEWAYS])
        logger.info(
            "Regime-adjusted params for %s: %s", regime_label, params
        )
        return dict(params)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Pickle the fitted detector to *path*.

        Parameters
        ----------
        path : str
            File path (will be created / overwritten).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "model": self.model,
                    "regime_stats": self.regime_stats,
                    "label_cache": self._label_cache,
                    "n_states": self.n_states,
                    "feature_window": self.feature_window,
                },
                fh,
            )
        logger.info("HMMRegimeDetector saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "HMMRegimeDetector":
        """
        Load a pickled detector from *path*.

        Parameters
        ----------
        path : str

        Returns
        -------
        HMMRegimeDetector
        """
        with open(path, "rb") as fh:
            state = pickle.load(fh)

        detector = cls(
            n_states=state["n_states"],
            feature_window=state["feature_window"],
        )
        detector.model = state["model"]
        detector.regime_stats = state["regime_stats"]
        detector._label_cache = state["label_cache"]
        logger.info("HMMRegimeDetector loaded from %s", path)
        return detector


# ===========================================================================
# 3.  Combined regime assessment
# ===========================================================================

def get_combined_regime(nifty_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Run both ``IndexRegimeFilter`` and ``HMMRegimeDetector`` and return a
    merged regime assessment.

    The combined label is determined by the following priority:
    - If the EMA filter says BEAR → combined is BEAR_TRENDING regardless of HMM.
    - If the EMA filter says BULL and HMM agrees (BULL_TRENDING) → BULL_TRENDING.
    - Otherwise the HMM label is used.

    Parameters
    ----------
    nifty_data : pd.DataFrame, optional
        Pre-fetched Nifty data with DATE and Close columns.  If None the
        function fetches it automatically.

    Returns
    -------
    dict with keys:
        ema_regime (str), hmm_regime (str), combined_label (str),
        is_tradeable (bool), recommended_threshold (float),
        position_size_multiplier (float), max_positions (int),
        nifty_close (float), ema50 (float), ema200 (float),
        trend_strength (float), hmm_confidence (float),
        timestamp (str)
    """
    ema_filter = IndexRegimeFilter()

    # --- EMA layer -------------------------------------------------------
    if nifty_data is None:
        nifty_data = ema_filter.fetch_nifty_data()

    nifty_data = ema_filter.compute_emas(nifty_data)
    ema_report = ema_filter.get_regime_report(nifty_data)
    ema_regime = ema_report["regime"]  # 'BULL', 'BEAR', 'SIDEWAYS'

    # --- HMM layer -------------------------------------------------------
    returns = nifty_data["Close"].pct_change().dropna()

    hmm_detector = HMMRegimeDetector()
    hmm_result: Dict[str, Any] = {}
    try:
        hmm_detector.fit(returns)
        hmm_result = hmm_detector.predict_current_regime(returns)
    except Exception as exc:
        logger.warning("HMM layer failed (%s) — using EMA regime only.", exc)
        hmm_result = {
            "regime_id": -1,
            "regime_label": SIDEWAYS,
            "confidence": 0.5,
            "regime_probs": [],
        }

    hmm_label = hmm_result.get("regime_label", SIDEWAYS)

    # --- Combine ---------------------------------------------------------
    if ema_regime == "BEAR":
        combined_label = BEAR_TRENDING
    elif ema_regime == "BULL" and hmm_label == BULL_TRENDING:
        combined_label = BULL_TRENDING
    else:
        combined_label = hmm_label

    params = hmm_detector.get_regime_adjusted_params(combined_label)

    # is_tradeable: not in hard bear regime
    is_tradeable = combined_label != BEAR_TRENDING

    result: Dict[str, Any] = {
        "ema_regime": ema_regime,
        "hmm_regime": hmm_label,
        "combined_label": combined_label,
        "is_tradeable": is_tradeable,
        "recommended_threshold": params["probability_threshold"],
        "position_size_multiplier": params["position_size_multiplier"],
        "max_positions": params["max_positions"],
        "nifty_close": ema_report["nifty_close"],
        "ema50": ema_report["ema50"],
        "ema200": ema_report["ema200"],
        "trend_strength": ema_report["trend_strength"],
        "hmm_confidence": hmm_result.get("confidence", 0.0),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    logger.info("Combined regime: %s", result)
    return result


# ===========================================================================
# Smoke test
# ===========================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    print("=" * 70)
    print("regime_filter.py — self-test")
    print("=" * 70)

    # --- IndexRegimeFilter -----------------------------------------------
    print("\n[1] IndexRegimeFilter")
    rf = IndexRegimeFilter()
    nifty = rf.fetch_nifty_data(lookback_days=300)
    nifty = rf.compute_emas(nifty)
    report = rf.get_regime_report(nifty)
    for k, v in report.items():
        print(f"  {k:<28}: {v}")

    # --- HMMRegimeDetector -----------------------------------------------
    print("\n[2] HMMRegimeDetector")
    returns_s = nifty["Close"].pct_change().dropna()
    det = HMMRegimeDetector(n_states=4, feature_window=20)
    det.fit(returns_s)
    pred = det.predict_current_regime(returns_s)
    for k, v in pred.items():
        print(f"  {k:<28}: {v}")
    params_out = det.get_regime_adjusted_params(pred["regime_label"])
    print("  Adjusted params:", params_out)

    # --- Combined --------------------------------------------------------
    print("\n[3] get_combined_regime()")
    ctx = get_combined_regime(nifty_data=nifty)
    for k, v in ctx.items():
        print(f"  {k:<28}: {v}")

    print("\nSelf-test complete.")
