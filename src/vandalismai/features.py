# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 muffydu37
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import pandas as pd

from muffybot.tasks.vandalism_shared import normalize_detection_text

URL_RE = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", flags=re.IGNORECASE)
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]", flags=re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9à-öø-ÿ_'-]{3,40}", flags=re.IGNORECASE)
IP_USER_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
REPEATED_CHAR_RE = re.compile(r"(.)\1{4,}")
UPPER_RE = re.compile(r"[A-ZÀ-ÖØ-Þ]")
ALNUM_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]")
DIGIT_RE = re.compile(r"\d")

DEFAULT_SENSITIVE_TITLE_TOKENS = {
    "obama",
    "macron",
    "trump",
    "hitler",
    "israel",
    "palestine",
    "gaza",
    "ukraine",
    "russie",
    "poutine",
    "religion",
    "sexe",
    "porn",
}

FEATURE_COLUMNS = [
    "text",
    "url_count",
    "symbol_ratio",
    "burst_count",
    "title_sensitive",
    "delta_len",
    "is_ip_user",
    "digit_ratio",
    "upper_ratio",
    "repeat_char_flag",
    "comment_revert_hint",
]

REVERT_HINT_WORDS = ("revert", "annulation", "rollback", "undo", "rv", "rvv", "restauration")


def _safe_text(value: object) -> str:
    return str(value or "")


def _symbol_ratio(text: str) -> float:
    if not text:
        return 0.0
    return min(len(NON_ALNUM_RE.findall(text)) / max(len(text), 1), 1.0)


def _title_sensitive(title: str) -> int:
    normalized = normalize_detection_text(title)
    return 1 if any(token in normalized for token in DEFAULT_SENSITIVE_TITLE_TOKENS) else 0


def _ratio_for_regex(text: str, pattern: re.Pattern[str]) -> float:
    if not text:
        return 0.0
    den = len(ALNUM_RE.findall(text))
    if den <= 0:
        return 0.0
    return min(len(pattern.findall(text)) / float(den), 1.0)


def _comment_revert_hint(comment: str) -> int:
    lowered = normalize_detection_text(comment)
    return 1 if any(word in lowered for word in REVERT_HINT_WORDS) else 0


def make_text_blob(row: dict[str, Any]) -> str:
    parts = [
        _safe_text(row.get("title")),
        _safe_text(row.get("comment")),
        _safe_text(row.get("added_text")),
        _safe_text(row.get("removed_text")),
    ]
    return normalize_detection_text(" \n ".join(parts))


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "user" not in out.columns:
        out["user"] = ""
    out["title"] = out["title"].map(_safe_text)
    out["user"] = out["user"].map(_safe_text)
    out["comment"] = out["comment"].map(_safe_text)
    out["added_text"] = out["added_text"].map(_safe_text)
    out["removed_text"] = out["removed_text"].map(_safe_text)
    out["text"] = out.apply(lambda row: make_text_blob(row.to_dict()), axis=1)
    out["url_count"] = out["text"].map(lambda text: int(len(URL_RE.findall(text))))
    out["symbol_ratio"] = out["text"].map(_symbol_ratio)
    out["delta_len"] = (out["added_text"].str.len() - out["removed_text"].str.len()).abs()
    out["title_sensitive"] = out["title"].map(_title_sensitive)
    out["burst_count"] = _compute_burst_count(out)
    out["is_ip_user"] = out["user"].map(lambda user: 1 if IP_USER_RE.match(str(user or "").strip()) else 0)
    out["digit_ratio"] = out["text"].map(lambda text: _ratio_for_regex(text, DIGIT_RE))
    out["upper_ratio"] = out["added_text"].map(lambda text: _ratio_for_regex(str(text), UPPER_RE))
    out["repeat_char_flag"] = out["text"].map(lambda text: 1 if REPEATED_CHAR_RE.search(text) else 0)
    out["comment_revert_hint"] = out["comment"].map(_comment_revert_hint)
    return out


def _compute_burst_count(df: pd.DataFrame) -> pd.Series:
    counts = Counter(str(item or "") for item in df.get("user", []))
    return df["user"].map(lambda user: int(counts.get(str(user or ""), 1)))


def top_tokens(text: str, max_items: int = 5) -> list[str]:
    tokens = [match.group(0) for match in TOKEN_RE.finditer(normalize_detection_text(text))]
    if not tokens:
        return []
    freq = Counter(tokens)
    return [token for token, _count in freq.most_common(max_items)]
