# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 muffydu37
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from muffybot.env import get_env

from ..features import FEATURE_COLUMNS, enrich_dataframe, top_tokens
from ..schemas import PredictionResult


class MLPredictor:
    def __init__(self, *, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.model_path = model_dir / "model.joblib"
        self.metadata_path = model_dir / "metadata.json"
        self.model = None
        self.metadata: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            self.model = None
            return
        self.model = joblib.load(self.model_path)
        if self.metadata_path.exists():
            try:
                self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except Exception:
                self.metadata = {}

    @property
    def enabled(self) -> bool:
        return self.model is not None

    def predict(self, *, lang: str, title: str, user: str, comment: str, added_text: str, removed_text: str) -> PredictionResult:
        if self.model is None:
            return PredictionResult(score=0.0, label="UNKNOWN", explanations=[], model_version="unavailable")

        frame = pd.DataFrame(
            [
                {
                    "lang": lang,
                    "title": title,
                    "user": user,
                    "comment": comment,
                    "added_text": added_text,
                    "removed_text": removed_text,
                    "label": 0,
                    "source": "runtime",
                    "timestamp": "",
                }
            ]
        )
        enriched = enrich_dataframe(frame)
        feature_cols = self.metadata.get("features")
        if not isinstance(feature_cols, list) or not feature_cols:
            feature_cols = FEATURE_COLUMNS
        x = enriched[[str(column) for column in feature_cols]]
        score = float(self.model.predict_proba(x)[0][1])
        threshold = float(self.metadata.get("threshold", 0.65) or 0.65)
        label = "VANDALISM" if score >= threshold else "LEGIT"
        explanations = top_tokens(str(enriched.iloc[0]["text"]), max_items=4)
        version = str(self.metadata.get("model_version") or "unknown")
        return PredictionResult(score=score, label=label, explanations=explanations, model_version=version)


_CACHE: MLPredictor | None = None


def load_predictor() -> MLPredictor:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    model_root = Path(get_env("ML_MODEL_DIR", "/home/ubuntu/vandalismai/model") or "/home/ubuntu/vandalismai/model")
    _CACHE = MLPredictor(model_dir=model_root / "latest")
    return _CACHE
