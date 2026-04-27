"""
train.py
Trains one XGBoost classifier per prediction horizon (1d, 5d, 20d).

Walk-forward split: train on first TRAIN_FRAC of dates, evaluate on remainder.
Models are saved to data/models/ as .json files for fast daily reload.
"""

import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              classification_report)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from config import (
    HORIZONS, MODELS_DIR, TRAIN_FRAC, XGBOOST_PARAMS,
    CONFIDENCE_THRESHOLD,
)
from features import get_feature_cols, get_target_col

log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def encode_tickers(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    le = LabelEncoder()
    df = df.copy()
    df["ticker_enc"] = le.fit_transform(df["ticker"])
    return df, le


def walk_forward_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split on date: first TRAIN_FRAC of unique dates → train, rest → test.
    This prevents look-ahead bias.
    """
    dates      = sorted(df.index.unique())
    cutoff_idx = int(len(dates) * TRAIN_FRAC)
    cutoff     = dates[cutoff_idx]
    train = df[df.index < cutoff]
    test  = df[df.index >= cutoff]
    log.info("Train: %s → %s (%d rows) | Test: %s → %s (%d rows)",
             dates[0].date(), cutoff.date(), len(train),
             cutoff.date(), dates[-1].date(), len(test))
    return train, test


def clean_for_training(df: pd.DataFrame, feature_cols: list[str],
                        target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows with NaN in features or target."""
    cols = feature_cols + [target_col]
    sub  = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    X    = sub[feature_cols]
    y    = sub[target_col].astype(int)
    return X, y


# ── Train one model ────────────────────────────────────────────────────────────

def train_horizon(df_encoded: pd.DataFrame, feature_cols: list[str],
                   horizon: str) -> XGBClassifier:
    """Train and evaluate a model for one prediction horizon."""
    target_col = get_target_col(horizon)
    train_df, test_df = walk_forward_split(df_encoded)

    X_train, y_train = clean_for_training(train_df, feature_cols, target_col)
    X_test,  y_test  = clean_for_training(test_df,  feature_cols, target_col)

    if X_train.empty:
        raise ValueError(f"Empty training set for horizon {horizon}")

    model = XGBClassifier(**XGBOOST_PARAMS)
    fit_kwargs: dict = {"verbose": False}
    if not X_test.empty:
        fit_kwargs["eval_set"] = [(X_test, y_test)]
    model.fit(X_train, y_train, **fit_kwargs)

    # ── Evaluation ────────────────────────────────────────────────────────────
    if not X_test.empty:
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        acc   = accuracy_score(y_test, preds)
        try:
            auc = roc_auc_score(y_test, probs)
        except Exception:
            auc = float("nan")

        # High-confidence accuracy
        high_conf_mask = (probs >= CONFIDENCE_THRESHOLD) | (probs <= (1 - CONFIDENCE_THRESHOLD))
        if high_conf_mask.sum() > 0:
            hc_acc = accuracy_score(y_test[high_conf_mask], preds[high_conf_mask])
            hc_n   = high_conf_mask.sum()
        else:
            hc_acc, hc_n = float("nan"), 0

        log.info(
            "Horizon %s → Accuracy: %.3f | AUC: %.3f | "
            "High-conf accuracy: %.3f (n=%d, threshold=%.2f)",
            horizon, acc, auc, hc_acc, hc_n, CONFIDENCE_THRESHOLD,
        )
        log.info("\n%s", classification_report(y_test, preds,
                                               target_names=["Down", "Up"],
                                               zero_division=0))
    return model


# ── Save / load models ─────────────────────────────────────────────────────────

def save_model(model: XGBClassifier, le: LabelEncoder, horizon: str,
               feature_cols: list[str]) -> None:
    model.save_model(str(MODELS_DIR / f"xgb_{horizon}.json"))
    with open(MODELS_DIR / f"meta_{horizon}.pkl", "wb") as fh:
        pickle.dump({"label_encoder": le, "feature_cols": feature_cols}, fh)
    log.info("Saved model for horizon %s", horizon)


def load_model(horizon: str) -> tuple[XGBClassifier, LabelEncoder, list[str]]:
    model = XGBClassifier()
    model.load_model(str(MODELS_DIR / f"xgb_{horizon}.json"))
    with open(MODELS_DIR / f"meta_{horizon}.pkl", "rb") as fh:
        meta = pickle.load(fh)
    return model, meta["label_encoder"], meta["feature_cols"]


# ── Full training run ──────────────────────────────────────────────────────────

def train_all(feature_matrix: pd.DataFrame) -> None:
    """Train and save models for all horizons."""
    df_encoded, le = encode_tickers(feature_matrix)
    feature_cols   = get_feature_cols(df_encoded)
    # Include encoded ticker as a feature
    if "ticker_enc" not in feature_cols:
        feature_cols = feature_cols + ["ticker_enc"]

    for horizon in HORIZONS:
        log.info("═" * 60)
        log.info("Training horizon: %s", horizon)
        model = train_horizon(df_encoded, feature_cols, horizon)
        save_model(model, le, horizon, feature_cols)

    log.info("═" * 60)
    log.info("All models trained and saved to %s", MODELS_DIR)


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    from cache_manager import refresh_all
    from features import build_feature_matrix

    log.info("Refreshing cache...")
    all_data = refresh_all()

    log.info("Building feature matrix...")
    matrix = build_feature_matrix(all_data)

    log.info("Training models...")
    train_all(matrix)
