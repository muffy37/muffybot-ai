# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 muffydu37
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TrainingSample:
    lang: str
    title: str
    user: str
    comment: str
    added_text: str
    removed_text: str
    label: int
    source: str
    timestamp: str


@dataclass(slots=True)
class PredictionResult:
    score: float
    label: str
    explanations: list[str] = field(default_factory=list)
    model_version: str = "unknown"
