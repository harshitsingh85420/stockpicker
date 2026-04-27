"""
production/meta_labeler.py -- P13: Meta-labeling secondary confidence filter.

Implements the Lopez de Prado meta-labeling technique:

  1. The primary model (LightGBM ensemble) generates direction signals.
  2. A secondary classifier is trained ONLY on rows where the primary model
     predicted positive (prob >= primary_threshold).
  3. The meta-label target is whether that trade was ACTUALLY profitable.
  4. At inference time, each primary pick must also pass the meta-labeler's
     confidence threshold to be retained.

This catches systematic false positives that the primary model misses --
e.g., breakout signals in bearish regime, or low-volume gap-ups.

Training features:
  - primary_prob        : primary model's raw probability
  - plus a subset of the technical features from the pick row

Saved to: stock_picker_data/models/meta_labeler.pkl
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROOT    = Path(__file__).parent.parent
MODEL_DIR = _ROOT / "stock_picker_data" / "models"

# Features the meta-labeler uses from the pick row.  These must already be
# present in the feature_df passed at training time.
META_FEATURES = [
    "primary_prob",   # primary model's raw probability (injected at train time)
    "ATR14",
    "ATRpct",
    "VolMult",
    "RS_Composite",
    "ADX14",
    "RSI14",
    "BBWidth",
    "DistTo52W",
    "Break63_Today",
    "RET21D",
    "RET63D",
]


class MetaLabeler:
    """
    P13 -- Secondary meta-labeling classifier.

    Parameters
    ----------
    model_path : Path
        Where to save / load the meta-labeler.
    primary_threshold : float
        Minimum primary model probability to be considered a primary positive
        during training (default 0.55 -- looser than live threshold to get
        enough training samples).
    meta_threshold : float
        Minimum meta-labeler probability required to pass at inference time
        (default 0.50).
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        primary_threshold: float = 0.55,
        meta_threshold: float = 0.50,
    ) -> None:
        self.model_path       = model_path or (MODEL_DIR / "meta_labeler.pkl")
        self.primary_threshold = primary_threshold
        self.meta_threshold   = meta_threshold
        self._model           = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        feature_df: pd.DataFrame,
        primary_probs: np.ndarray,
        label_col: str = "Label_fwd5_positive",
    ) -> dict:
        """
        Train the meta-labeler.

        Parameters
        ----------
        feature_df    : Full feature DataFrame (all rows, chronological order).
                        Must contain label_col and the columns in META_FEATURES.
        primary_probs : Array of primary model probabilities, same length as
                        feature_df (row-aligned).
        label_col     : True binary outcome column.

        Returns
        -------
        dict with keys: meta_auc, n_training_samples, meta_positive_rate.
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score

        df = feature_df.copy()
        df["primary_prob"] = primary_probs

        # Keep only primary positives and rows with known outcome
        mask = (df["primary_prob"] >= self.primary_threshold) & df[label_col].notna()
        df_pos = df[mask].copy()

        if len(df_pos) < 30:
            logger.warning(
                "P13: Only %d primary positives found — meta-labeler requires >= 30. "
                "Skipping fit.",
                len(df_pos),
            )
            return {"meta_auc": None, "n_training_samples": len(df_pos)}

        avail_feats = [c for c in META_FEATURES if c in df_pos.columns]
        X = df_pos[avail_feats].fillna(0).values
        y = df_pos[label_col].astype(int).values

        logger.info(
            "P13 MetaLabeler: training on %d primary-positive rows, %d features, "
            "%.1f%% positive.",
            len(X), len(avail_feats), y.mean() * 100,
        )

        tscv = TimeSeriesSplit(n_splits=min(3, max(2, len(X) // 50)))
        cv_aucs = []

        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
            if len(va_idx) < 5:
                continue
            clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42,
            )
            clf.fit(X[tr_idx], y[tr_idx])
            auc = roc_auc_score(y[va_idx], clf.predict_proba(X[va_idx])[:, 1])
            cv_aucs.append(auc)
            logger.info("  Meta fold %d: AUC=%.4f", fold, auc)

        # Final model on all primary-positive data
        final_clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
        final_clf.fit(X, y)

        self._model = {"clf": final_clf, "features": avail_feats}
        self._save()

        mean_auc = float(np.mean(cv_aucs)) if cv_aucs else None
        logger.info(
            "P13 MetaLabeler trained. CV AUC=%.4f, saved to %s",
            mean_auc or 0.0, self.model_path,
        )
        return {
            "meta_auc":             mean_auc,
            "n_training_samples":   len(X),
            "meta_positive_rate":   float(y.mean()),
            "meta_features_used":   avail_feats,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def filter_picks(
        self,
        picks_df: pd.DataFrame,
        primary_prob_col: str = "Probability_Raw",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply meta-labeler to a picks DataFrame.

        Parameters
        ----------
        picks_df         : Primary picks (must have primary_prob_col + META_FEATURES
                           columns where available).
        primary_prob_col : Column in picks_df holding the primary model probability.

        Returns
        -------
        (passed_df, rejected_df)
        """
        model_pkg = self._load()
        if model_pkg is None:
            logger.debug("P13: No meta-labeler available — passing all picks through.")
            return picks_df, pd.DataFrame(columns=picks_df.columns)

        clf     = model_pkg["clf"]
        features = model_pkg["features"]

        df = picks_df.copy()
        if primary_prob_col in df.columns and "primary_prob" not in df.columns:
            df["primary_prob"] = df[primary_prob_col]

        avail = [c for c in features if c in df.columns]
        if not avail:
            logger.warning("P13: No meta features available in picks_df — skipping filter.")
            return picks_df, pd.DataFrame(columns=picks_df.columns)

        X = df.reindex(columns=features, fill_value=0).fillna(0).values
        meta_probs = clf.predict_proba(X)[:, 1]
        picks_df = picks_df.copy()
        picks_df["Meta_Prob"] = meta_probs

        passed   = picks_df[meta_probs >= self.meta_threshold].copy()
        rejected = picks_df[meta_probs <  self.meta_threshold].copy()

        logger.info(
            "P13 MetaLabeler: %d -> %d picks (rejected %d, meta_thresh=%.2f)",
            len(picks_df), len(passed), len(rejected), self.meta_threshold,
        )
        return passed, rejected

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, "wb") as fh:
            pickle.dump(self._model, fh)

    def _load(self) -> Optional[dict]:
        if self._model is not None:
            return self._model
        if not self.model_path.exists():
            return None
        try:
            with open(self.model_path, "rb") as fh:
                self._model = pickle.load(fh)
            logger.info("P13: MetaLabeler loaded from %s", self.model_path)
            return self._model
        except Exception as exc:
            logger.warning("P13: MetaLabeler load failed: %s", exc)
            return None
