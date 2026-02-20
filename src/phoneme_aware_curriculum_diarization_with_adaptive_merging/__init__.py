"""Phoneme-Aware Curriculum Diarization with Adaptive Merging.

A novel speaker diarization system that jointly learns speaker-discriminative
and phoneme-aware representations using curriculum learning.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from .models.model import DualEncoderDiarizationModel
from .models.components import PhonemeAwareLoss, AdaptiveMergingModule

__all__ = [
    "DualEncoderDiarizationModel",
    "PhonemeAwareLoss",
    "AdaptiveMergingModule",
]
