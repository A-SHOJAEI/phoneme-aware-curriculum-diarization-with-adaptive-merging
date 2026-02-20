"""Model architecture and custom components."""

from .model import DualEncoderDiarizationModel
from .components import PhonemeAwareLoss, AdaptiveMergingModule, CurriculumScheduler

__all__ = [
    "DualEncoderDiarizationModel",
    "PhonemeAwareLoss",
    "AdaptiveMergingModule",
    "CurriculumScheduler",
]
