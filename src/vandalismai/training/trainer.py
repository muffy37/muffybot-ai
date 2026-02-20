# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 muffydu37
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from muffybot.env import get_env, get_int_env

from ..features import FEATURE_COLUMNS, enrich_dataframe


def _timestamp_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_auc(y_true: pd.Series, y_prob: list[float]) -> float:
    try:
        value = float(roc_auc_score(y_true, y_prob))
        if value != value:  # NaN
            return 0.0
        return value
    except Exception:
        return 0.0


def _model_dirs() -> tuple[Path, Path, Path]:
    model_root = Path(get_env("ML_MODEL_DIR", "/home/ubuntu/vandalismai/model") or "/home/ubuntu/vandalismai/model")
    data_dir = Path(get_env("ML_DATA_DIR", "/home/ubuntu/vandalismai/data") or "/home/ubuntu/vandalismai/data")
    latest_dir = model_root / "latest"
    archive_dir = model_root / "archive" / _timestamp_key()
    latest_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, latest_dir, archive_dir


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "word_tfidf",
                TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=60000, min_df=2),
                "text",
            ),
            (
                "char_tfidf",
                TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=40000, min_df=2),
                "text",
            ),
            ("num", "passthrough", [column for column in FEATURE_COLUMNS if column != "text"]),
        ],
        sparse_threshold=0.3,
    )


def _build_model() -> CalibratedClassifierCV:
    base = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor()),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear")),
        ]
    )
    try:
        model = CalibratedClassifierCV(estimator=base, cv=3, method="sigmoid")
    except TypeError:
        model = CalibratedClassifierCV(base_estimator=base, cv=3, method="sigmoid")  # type: ignore[call-arg]
    return model


def _temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.copy()
    ordered["parsed_ts"] = pd.to_datetime(ordered["timestamp"], errors="coerce", utc=True)
    train_parts: list[pd.DataFrame] = []
    holdout_parts: list[pd.DataFrame] = []
    for _lang, group in ordered.groupby("lang", dropna=False):
        current = group.sort_values(by="parsed_ts", na_position="last")
        split_index = max(int(len(current) * 0.8), 1)
        train_parts.append(current.iloc[:split_index])
        holdout_parts.append(current.iloc[split_index:])
    train = pd.concat(train_parts, ignore_index=True) if train_parts else ordered.copy()
    holdout = pd.concat(holdout_parts, ignore_index=True) if holdout_parts else pd.DataFrame(columns=ordered.columns)
    train = train.drop(columns=["parsed_ts"], errors="ignore")
    holdout = holdout.drop(columns=["parsed_ts"], errors="ignore")
    if holdout.empty:
        holdout = train.tail(min(len(train), 200)).copy()
    if train["label"].nunique() < 2 or holdout["label"].nunique() < 2:
        fallback_train, fallback_holdout = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["label"] if df["label"].nunique() >= 2 else None,
        )
        train = fallback_train.copy()
        holdout = fallback_holdout.copy()
    return train, holdout


def _metrics(y_true: pd.Series, y_prob: list[float], threshold: float = 0.5) -> dict[str, float]:
    y_pred = [1 if score >= threshold else 0 for score in y_prob]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": _safe_auc(y_true, y_prob),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def _select_threshold(y_true: pd.Series, y_prob: list[float], *, target_precision: float) -> tuple[float, list[dict[str, float]]]:
    best_thr = 0.5
    best_f1 = -1.0
    best_prec_thr = None
    best_prec_recall = -1.0
    report: list[dict[str, float]] = []
    for raw in range(5, 96, 5):
        thr = raw / 100.0
        y_pred = [1 if score >= thr else 0 for score in y_prob]
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        current = float(f1_score(y_true, y_pred, zero_division=0))
        report.append(
            {
                "threshold": float(thr),
                "precision": precision,
                "recall": recall,
                "f1": current,
            }
        )
        if precision >= target_precision and recall > best_prec_recall:
            best_prec_recall = recall
            best_prec_thr = thr
        if current > best_f1:
            best_f1 = current
            best_thr = thr
    return (float(best_prec_thr) if best_prec_thr is not None else float(best_thr), report)


def _rebalance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    max_ratio = max(get_int_env("ML_MAX_NEG_POS_RATIO", 6), 1)
    positives = df[df["label"] == 1]
    negatives = df[df["label"] == 0]
    if positives.empty or negatives.empty:
        return df

    result_parts: list[pd.DataFrame] = [positives]
    for lang, neg_group in negatives.groupby("lang", dropna=False):
        pos_lang = positives[positives["lang"] == lang]
        cap = len(pos_lang) * max_ratio
        if len(pos_lang) == 0:
            cap = min(len(neg_group), max_ratio * 50)
        if cap <= 0 or len(neg_group) <= cap:
            result_parts.append(neg_group)
        else:
            result_parts.append(neg_group.sample(n=cap, random_state=42))

    return pd.concat(result_parts, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)


def train_and_export(df: pd.DataFrame) -> dict[str, Any]:
    min_train = max(get_int_env("ML_MIN_TRAIN_SAMPLES", 1000), 50)
    if len(df) < min_train:
        raise RuntimeError(f"Dataset trop petit pour entraînement: {len(df)} < {min_train}")

    enriched = enrich_dataframe(df)
    deduped = enriched.drop_duplicates(subset=["lang", "title", "user", "comment", "added_text", "removed_text", "label"])
    deduped = deduped.dropna(subset=["label"])
    deduped["label"] = deduped["label"].astype(int)
    deduped = _rebalance_dataset(deduped)

    train_df, holdout_df = _temporal_split(deduped)
    feature_cols = FEATURE_COLUMNS

    x_train = train_df[feature_cols]
    y_train = train_df["label"]
    x_hold = holdout_df[feature_cols]
    y_hold = holdout_df["label"]

    model = _build_model()
    model.fit(x_train, y_train)
    hold_probs = model.predict_proba(x_hold)[:, 1].tolist()
    try:
        target_precision = float(get_env("ML_TARGET_PRECISION", "0.90") or "0.90")
    except Exception:
        target_precision = 0.90
    target_precision = min(max(target_precision, 0.0), 1.0)
    threshold, threshold_report = _select_threshold(y_hold, hold_probs, target_precision=target_precision)
    metrics = _metrics(y_hold, hold_probs, threshold=threshold)
    metrics["threshold"] = float(threshold)

    data_dir, latest_dir, archive_dir = _model_dirs()
    train_path = data_dir / "train_dataset.parquet"
    eval_path = data_dir / "eval_dataset.parquet"
    train_df.to_parquet(train_path, index=False)
    holdout_df.to_parquet(eval_path, index=False)

    model_path = latest_dir / "model.joblib"
    vectorizer_path = latest_dir / "vectorizer.joblib"
    metadata_path = latest_dir / "metadata.json"
    metrics_path = latest_dir / "metrics.json"
    thresholds_path = latest_dir / "thresholds.json"

    joblib.dump(model, model_path)
    # Keep a dedicated vectorizer artifact (preprocessor from first calibrated estimator).
    preprocessor = None
    if getattr(model, "calibrated_classifiers_", None):
        estimator = model.calibrated_classifiers_[0].estimator
        preprocessor = estimator.named_steps.get("preprocessor")
    joblib.dump(preprocessor, vectorizer_path)

    metadata = {
        "model_version": _timestamp_key(),
        "rows_total": int(len(deduped)),
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "labels_positive": int(deduped["label"].sum()),
        "labels_negative": int((1 - deduped["label"]).sum()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "features": feature_cols,
        "threshold": float(threshold),
        "target_precision": float(target_precision),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    thresholds_path.write_text(json.dumps(threshold_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    for path in (model_path, vectorizer_path, metadata_path, metrics_path, thresholds_path):
        shutil.copy2(path, archive_dir / path.name)

    return {
        "metrics": metrics,
        "metadata": metadata,
        "paths": {
            "train_dataset": str(train_path),
            "eval_dataset": str(eval_path),
            "model": str(model_path),
            "vectorizer": str(vectorizer_path),
            "metadata": str(metadata_path),
            "metrics": str(metrics_path),
            "thresholds": str(thresholds_path),
            "archive": str(archive_dir),
        },
    }


def evaluate_saved_model(*, model_path: Path, eval_dataset_path: Path) -> dict[str, float]:
    model = joblib.load(model_path)
    df = pd.read_parquet(eval_dataset_path)
    enriched = enrich_dataframe(df)
    feature_cols = list(FEATURE_COLUMNS)
    metadata_path = model_path.with_name("metadata.json")
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            raw_cols = payload.get("features")
            if isinstance(raw_cols, list) and raw_cols:
                feature_cols = [str(item) for item in raw_cols]
        except Exception:
            feature_cols = list(FEATURE_COLUMNS)
    x_eval = enriched[feature_cols]
    y_eval = enriched["label"].astype(int)
    probs = model.predict_proba(x_eval)[:, 1].tolist()
    threshold = 0.5
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            threshold = float(payload.get("threshold", 0.5))
        except Exception:
            threshold = 0.5
    metrics = _metrics(y_eval, probs, threshold=threshold)
    metrics["threshold"] = float(threshold)
    return metrics
