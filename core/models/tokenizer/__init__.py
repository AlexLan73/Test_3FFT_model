"""Объёмный токенизатор (гл.4 + гл.4-бис) -- публичный API пакета.

`SquareView`/`SquareToken` (`core/graphics/square_view.py`) остаются нетронутыми
(контрольный вид reduce+argmax) -- это полноценный OS-CFAR-детектор поверх них.
"""
from .cfar import OsCfarDetector
from .features import FeatureExtractor, FeatureVector
from .tokenizer import VolumeTokenizer, assemble_range
from .tokens import (
    BARRAGE,
    COMB,
    TARGET,
    PeakInfo,
    RangeVerdict,
    SliceToken,
)
from .triage import NOISE, SMEARED, SOURCE, RuleBasedTriage, SliceTriage

__all__ = [
    "FeatureVector", "FeatureExtractor",
    "OsCfarDetector",
    "PeakInfo", "SliceToken", "RangeVerdict",
    "NOISE", "SOURCE", "SMEARED",
    "TARGET", "COMB", "BARRAGE",
    "SliceTriage", "RuleBasedTriage",
    "VolumeTokenizer", "assemble_range",
]
