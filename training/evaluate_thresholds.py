#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from vandalismai.features import FEATURE_COLUMNS, enrich_dataframe


def _metrics(y_true: pd.Series, probs: list[float], threshold: float) -> dict[str, float]:
    y_pred = [1 if score >= threshold else 0 for score in probs]
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Évalue précision/rappel/F1 selon plusieurs seuils.")
    parser.add_argument("--model", default="model/latest/model.joblib", help="Chemin du modèle joblib")
    parser.add_argument("--eval", default="data/eval_dataset.parquet", help="Jeu d'évaluation parquet")
    parser.add_argument("--metadata", default="model/latest/metadata.json", help="Metadata json")
    parser.add_argument("--output", default="model/latest/threshold_eval.json", help="Sortie JSON du rapport")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    eval_path = Path(args.eval)
    metadata_path = Path(args.metadata)
    output_path = Path(args.output)

    model = joblib.load(model_path)
    df = pd.read_parquet(eval_path)
    enriched = enrich_dataframe(df)

    feature_cols = list(FEATURE_COLUMNS)
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            raw_cols = payload.get("features")
            if isinstance(raw_cols, list) and raw_cols:
                feature_cols = [str(item) for item in raw_cols]
        except Exception:
            pass

    x_eval = enriched[feature_cols]
    y_eval = enriched["label"].astype(int)
    probs = model.predict_proba(x_eval)[:, 1].tolist()

    rows = [_metrics(y_eval, probs, raw / 100.0) for raw in range(5, 96, 5)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] report: {output_path}")
    for row in rows:
        print(
            f"thr={row['threshold']:.2f} "
            f"prec={row['precision']:.3f} rec={row['recall']:.3f} f1={row['f1']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
