# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MPL-2.0
# Copyright (c) 2026 muffydu37
"""Vandalism AI package."""

from .inference.predictor import MLPredictor, load_predictor
from .schemas import PredictionResult, TrainingSample

__all__ = ["TrainingSample", "PredictionResult", "MLPredictor", "load_predictor"]
