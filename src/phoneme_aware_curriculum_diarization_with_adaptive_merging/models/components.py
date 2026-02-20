"""Custom loss functions, layers, and training components for phoneme-aware diarization."""

import logging
import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PhonemeAwareLoss(nn.Module):
    """Custom loss combining speaker discrimination and phoneme-level alignment.

    This is a novel component that combines:
    1. Speaker classification loss (cross-entropy)
    2. Phoneme-aware contrastive loss (pull same-speaker embeddings together)
    3. Boundary alignment loss (ensure boundaries align with phoneme transitions)
    """

    def __init__(
        self,
        speaker_weight: float = 1.0,
        phoneme_weight: float = 0.5,
        boundary_weight: float = 0.3,
        temperature: float = 0.07,
    ):
        """Initialize phoneme-aware loss.

        Args:
            speaker_weight: Weight for speaker classification loss.
            phoneme_weight: Weight for phoneme-aware contrastive loss.
            boundary_weight: Weight for boundary alignment loss.
            temperature: Temperature parameter for contrastive loss.
        """
        super().__init__()
        self.speaker_weight = speaker_weight
        self.phoneme_weight = phoneme_weight
        self.boundary_weight = boundary_weight
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        speaker_logits: torch.Tensor,
        speaker_embeddings: torch.Tensor,
        phoneme_embeddings: torch.Tensor,
        speaker_labels: torch.Tensor,
        phoneme_boundaries: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss.

        Args:
            speaker_logits: Speaker classification logits [batch, num_speakers].
            speaker_embeddings: Speaker embeddings [batch, embed_dim].
            phoneme_embeddings: Phoneme-level embeddings [batch, embed_dim].
            speaker_labels: Ground truth speaker labels [batch].
            phoneme_boundaries: Optional boundary prediction [batch, seq_len].

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        # 1. Speaker classification loss
        speaker_loss = self.ce_loss(speaker_logits, speaker_labels)

        # 2. Phoneme-aware contrastive loss
        # Normalize embeddings
        speaker_embeddings = F.normalize(speaker_embeddings, dim=1)
        phoneme_embeddings = F.normalize(phoneme_embeddings, dim=1)

        # Compute similarity matrix weighted by phoneme similarity
        similarity = torch.matmul(speaker_embeddings, speaker_embeddings.T) / self.temperature
        phoneme_similarity = torch.matmul(phoneme_embeddings, phoneme_embeddings.T)

        # Weight contrastive loss by phoneme similarity
        weighted_similarity = similarity * torch.sigmoid(phoneme_similarity)

        # Create label matrix (1 for same speaker, 0 otherwise)
        label_matrix = (speaker_labels.unsqueeze(0) == speaker_labels.unsqueeze(1)).float()

        # Compute contrastive loss
        exp_sim = torch.exp(weighted_similarity)
        pos_sim = (exp_sim * label_matrix).sum(dim=1)
        all_sim = exp_sim.sum(dim=1)
        contrastive_loss = -torch.log(pos_sim / (all_sim + 1e-8) + 1e-8).mean()

        # 3. Boundary alignment loss (if provided)
        boundary_loss = torch.tensor(0.0, device=speaker_logits.device)
        if phoneme_boundaries is not None:
            # Encourage sharp boundaries (binary cross-entropy with soft targets)
            boundary_probs = torch.sigmoid(phoneme_boundaries)
            # Penalize uncertain boundaries (entropy regularization)
            entropy = -boundary_probs * torch.log(boundary_probs + 1e-8)
            entropy -= (1 - boundary_probs) * torch.log(1 - boundary_probs + 1e-8)
            boundary_loss = entropy.mean()

        # Combine losses
        total_loss = (
            self.speaker_weight * speaker_loss +
            self.phoneme_weight * contrastive_loss +
            self.boundary_weight * boundary_loss
        )

        loss_dict = {
            'speaker_loss': speaker_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'boundary_loss': boundary_loss.item(),
            'total_loss': total_loss.item(),
        }

        return total_loss, loss_dict


class AdaptiveMergingModule(nn.Module):
    """Adaptive cluster merging based on phonetic context.

    This module refines speaker boundaries by considering phonetic similarity
    when deciding whether to merge adjacent speaker segments.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dim: int = 128,
        threshold: float = 0.5,
    ):
        """Initialize adaptive merging module.

        Args:
            embed_dim: Embedding dimension.
            hidden_dim: Hidden layer dimension.
            threshold: Merging threshold (0-1).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.threshold = threshold

        # MLP to predict merge probability
        self.merge_predictor = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),  # speaker1, speaker2, phoneme context
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        segment_embeddings: torch.Tensor,
        phoneme_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Predict merge probabilities for adjacent segments.

        Args:
            segment_embeddings: Speaker segment embeddings [batch, num_segments, embed_dim].
            phoneme_embeddings: Phoneme context embeddings [batch, num_segments, embed_dim].

        Returns:
            Merge probabilities [batch, num_segments - 1].
        """
        batch_size, num_segments, _ = segment_embeddings.shape

        if num_segments < 2:
            return torch.zeros(batch_size, 0, device=segment_embeddings.device)

        # Concatenate adjacent segment pairs with phoneme context
        seg1 = segment_embeddings[:, :-1, :]
        seg2 = segment_embeddings[:, 1:, :]
        phoneme = (phoneme_embeddings[:, :-1, :] + phoneme_embeddings[:, 1:, :]) / 2

        merge_input = torch.cat([seg1, seg2, phoneme], dim=-1)

        # Predict merge probability
        merge_probs = self.merge_predictor(merge_input).squeeze(-1)

        return merge_probs

    def apply_merging(
        self,
        segment_labels: torch.Tensor,
        merge_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Apply merging decisions to segment labels.

        Args:
            segment_labels: Original segment labels [batch, num_segments].
            merge_probs: Merge probabilities [batch, num_segments - 1].

        Returns:
            Merged segment labels [batch, num_segments].
        """
        batch_size, num_segments = segment_labels.shape
        merged_labels = segment_labels.clone()

        for b in range(batch_size):
            for i in range(num_segments - 1):
                if merge_probs[b, i] > self.threshold:
                    # Merge segment i+1 into segment i
                    merged_labels[b, i + 1] = merged_labels[b, i]

        return merged_labels


class CurriculumScheduler:
    """Curriculum learning scheduler for progressively harder training.

    Gradually increases difficulty by:
    1. Starting with clear single-speaker segments
    2. Introducing overlapping speech
    3. Adding challenging multilingual/code-switched examples
    """

    def __init__(
        self,
        num_stages: int = 3,
        stage_epochs: list = None,
        overlap_ratios: list = None,
    ):
        """Initialize curriculum scheduler.

        Args:
            num_stages: Number of curriculum stages.
            stage_epochs: Epochs per stage.
            overlap_ratios: Speech overlap ratios per stage.
        """
        self.num_stages = num_stages
        self.stage_epochs = stage_epochs or [10, 15, 25]
        self.overlap_ratios = overlap_ratios or [0.0, 0.2, 0.4]
        self.current_stage = 0
        self.current_epoch = 0

    def step(self, epoch: int) -> int:
        """Update curriculum stage based on epoch.

        Args:
            epoch: Current training epoch.

        Returns:
            Current curriculum stage (0 to num_stages - 1).
        """
        self.current_epoch = epoch

        cumulative_epochs = 0
        for stage, stage_epoch in enumerate(self.stage_epochs):
            cumulative_epochs += stage_epoch
            if epoch < cumulative_epochs:
                self.current_stage = stage
                break
        else:
            self.current_stage = self.num_stages - 1

        return self.current_stage

    def get_difficulty(self) -> Dict[str, float]:
        """Get current difficulty parameters.

        Returns:
            Dictionary with difficulty settings.
        """
        return {
            'stage': self.current_stage,
            'overlap_ratio': self.overlap_ratios[min(self.current_stage, len(self.overlap_ratios) - 1)],
            'augmentation_strength': 0.5 + 0.5 * (self.current_stage / self.num_stages),
        }


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-based models."""

    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoding.

        Args:
            d_model: Embedding dimension.
            max_len: Maximum sequence length.
        """
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Args:
            x: Input tensor [seq_len, batch, d_model].

        Returns:
            Encoded tensor.
        """
        return x + self.pe[:x.size(0)]
