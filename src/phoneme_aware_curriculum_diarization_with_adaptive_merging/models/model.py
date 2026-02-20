"""Core dual-encoder model for phoneme-aware speaker diarization."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import AdaptiveMergingModule, PositionalEncoding

logger = logging.getLogger(__name__)


class SpeakerEncoder(nn.Module):
    """Encoder for speaker-discriminative embeddings."""

    def __init__(
        self,
        input_dim: int = 80,
        embed_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """Initialize speaker encoder.

        Args:
            input_dim: Input feature dimension (e.g., mel bins).
            embed_dim: Embedding dimension.
            num_layers: Number of transformer layers.
            dropout: Dropout rate.
        """
        super().__init__()

        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input features to speaker embeddings.

        Args:
            x: Input features [batch, n_mels, time].

        Returns:
            Speaker embeddings [batch, embed_dim].
        """
        # Transpose to [time, batch, n_mels]
        x = x.permute(2, 0, 1)

        # Project to embed_dim
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x)

        # Pool over time: [time, batch, embed_dim] -> [batch, embed_dim, time]
        x = x.permute(1, 2, 0)
        x = self.pooling(x).squeeze(-1)

        return x


class PhonemeEncoder(nn.Module):
    """Encoder for phoneme-aware acoustic embeddings."""

    def __init__(
        self,
        input_dim: int = 80,
        embed_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize phoneme encoder.

        Args:
            input_dim: Input feature dimension.
            embed_dim: Embedding dimension.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate.
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=embed_dim // 2,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True,
        )

        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input features to phoneme embeddings.

        Args:
            x: Input features [batch, n_mels, time].

        Returns:
            Phoneme embeddings [batch, embed_dim].
        """
        # Transpose to [batch, time, n_mels]
        x = x.permute(0, 2, 1)

        # LSTM encoding
        x, _ = self.lstm(x)

        # Pool over time: [batch, time, embed_dim] -> [batch, embed_dim]
        x = x.permute(0, 2, 1)
        x = self.pooling(x).squeeze(-1)

        return x


class DualEncoderDiarizationModel(nn.Module):
    """Dual-encoder model for phoneme-aware speaker diarization.

    This model jointly learns:
    1. Speaker-discriminative representations via transformer encoder
    2. Phoneme-aware acoustic representations via LSTM encoder
    3. Adaptive merging of speaker segments using phonetic context
    """

    def __init__(
        self,
        input_dim: int = 80,
        embed_dim: int = 256,
        num_speakers: int = 100,
        speaker_layers: int = 3,
        phoneme_layers: int = 2,
        dropout: float = 0.1,
        use_adaptive_merging: bool = True,
    ):
        """Initialize dual-encoder diarization model.

        Args:
            input_dim: Input feature dimension (mel bins).
            embed_dim: Embedding dimension.
            num_speakers: Number of speaker classes.
            speaker_layers: Number of layers in speaker encoder.
            phoneme_layers: Number of layers in phoneme encoder.
            dropout: Dropout rate.
            use_adaptive_merging: Whether to use adaptive merging module.
        """
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_speakers = num_speakers
        self.use_adaptive_merging = use_adaptive_merging

        # Dual encoders
        self.speaker_encoder = SpeakerEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_layers=speaker_layers,
            dropout=dropout,
        )

        self.phoneme_encoder = PhonemeEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_layers=phoneme_layers,
            dropout=dropout,
        )

        # Speaker classification head
        self.speaker_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_speakers),
        )

        # Adaptive merging module
        if use_adaptive_merging:
            self.adaptive_merging = AdaptiveMergingModule(
                embed_dim=embed_dim,
                hidden_dim=128,
            )

        logger.info(f"Initialized DualEncoderDiarizationModel with {self.count_parameters()} parameters")

    def forward(
        self,
        features: torch.Tensor,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through dual encoder.

        Args:
            features: Input mel spectrograms [batch, n_mels, time].
            return_embeddings: Whether to return intermediate embeddings.

        Returns:
            Dictionary with model outputs.
        """
        # Encode speaker and phoneme representations
        speaker_embeddings = self.speaker_encoder(features)
        phoneme_embeddings = self.phoneme_encoder(features)

        # Speaker classification
        speaker_logits = self.speaker_classifier(speaker_embeddings)

        outputs = {
            'speaker_logits': speaker_logits,
            'speaker_embeddings': speaker_embeddings,
            'phoneme_embeddings': phoneme_embeddings,
        }

        if return_embeddings:
            outputs['raw_speaker_embeddings'] = speaker_embeddings
            outputs['raw_phoneme_embeddings'] = phoneme_embeddings

        return outputs

    def predict_speakers(
        self,
        features: torch.Tensor,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict speaker labels for input features.

        Args:
            features: Input mel spectrograms [batch, n_mels, time].
            threshold: Confidence threshold for predictions.

        Returns:
            Tuple of (predicted_labels, confidence_scores).
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(features)
            logits = outputs['speaker_logits']

            # Get probabilities and predictions
            probs = F.softmax(logits, dim=-1)
            confidence, predicted = torch.max(probs, dim=-1)

            # Filter by threshold
            predicted[confidence < threshold] = -1  # Unknown speaker

        return predicted, confidence

    def extract_embeddings(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract speaker and phoneme embeddings.

        Args:
            features: Input mel spectrograms [batch, n_mels, time].

        Returns:
            Dictionary with embeddings.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(features, return_embeddings=True)

        return {
            'speaker': outputs['speaker_embeddings'],
            'phoneme': outputs['phoneme_embeddings'],
        }

    def count_parameters(self) -> int:
        """Count trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict:
        """Get model configuration.

        Returns:
            Dictionary with model configuration.
        """
        return {
            'input_dim': self.input_dim,
            'embed_dim': self.embed_dim,
            'num_speakers': self.num_speakers,
            'use_adaptive_merging': self.use_adaptive_merging,
        }
