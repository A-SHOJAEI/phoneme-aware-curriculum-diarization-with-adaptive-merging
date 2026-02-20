"""Tests for model architecture and components."""

import pytest
import torch

from phoneme_aware_curriculum_diarization_with_adaptive_merging.models.components import (
    AdaptiveMergingModule,
    CurriculumScheduler,
    PhonemeAwareLoss,
)
from phoneme_aware_curriculum_diarization_with_adaptive_merging.models.model import (
    DualEncoderDiarizationModel,
    PhonemeEncoder,
    SpeakerEncoder,
)


class TestSpeakerEncoder:
    """Tests for SpeakerEncoder."""

    def test_forward(self, sample_features, embed_dim):
        """Test forward pass."""
        encoder = SpeakerEncoder(
            input_dim=80,
            embed_dim=embed_dim,
            num_layers=2,
        )

        embeddings = encoder(sample_features)

        assert embeddings.shape == (sample_features.shape[0], embed_dim)

    def test_output_range(self, sample_features, embed_dim):
        """Test output is in reasonable range."""
        encoder = SpeakerEncoder(input_dim=80, embed_dim=embed_dim)

        embeddings = encoder(sample_features)

        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()


class TestPhonemeEncoder:
    """Tests for PhonemeEncoder."""

    def test_forward(self, sample_features, embed_dim):
        """Test forward pass."""
        encoder = PhonemeEncoder(
            input_dim=80,
            embed_dim=embed_dim,
            num_layers=2,
        )

        embeddings = encoder(sample_features)

        assert embeddings.shape == (sample_features.shape[0], embed_dim)


class TestDualEncoderDiarizationModel:
    """Tests for DualEncoderDiarizationModel."""

    def test_initialization(self, input_dim, embed_dim, num_speakers):
        """Test model initialization."""
        model = DualEncoderDiarizationModel(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_speakers=num_speakers,
        )

        assert model.embed_dim == embed_dim
        assert model.num_speakers == num_speakers

    def test_forward(self, sample_features, input_dim, embed_dim, num_speakers):
        """Test forward pass."""
        model = DualEncoderDiarizationModel(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_speakers=num_speakers,
        )

        outputs = model(sample_features)

        assert 'speaker_logits' in outputs
        assert 'speaker_embeddings' in outputs
        assert 'phoneme_embeddings' in outputs

        assert outputs['speaker_logits'].shape == (sample_features.shape[0], num_speakers)
        assert outputs['speaker_embeddings'].shape == (sample_features.shape[0], embed_dim)

    def test_predict_speakers(self, sample_features, input_dim, embed_dim, num_speakers):
        """Test speaker prediction."""
        model = DualEncoderDiarizationModel(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_speakers=num_speakers,
        )

        predictions, confidences = model.predict_speakers(sample_features)

        assert predictions.shape == (sample_features.shape[0],)
        assert confidences.shape == (sample_features.shape[0],)
        assert (confidences >= 0).all() and (confidences <= 1).all()

    def test_extract_embeddings(self, sample_features, input_dim, embed_dim, num_speakers):
        """Test embedding extraction."""
        model = DualEncoderDiarizationModel(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_speakers=num_speakers,
        )

        embeddings = model.extract_embeddings(sample_features)

        assert 'speaker' in embeddings
        assert 'phoneme' in embeddings
        assert embeddings['speaker'].shape == (sample_features.shape[0], embed_dim)

    def test_count_parameters(self, input_dim, embed_dim, num_speakers):
        """Test parameter counting."""
        model = DualEncoderDiarizationModel(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_speakers=num_speakers,
        )

        num_params = model.count_parameters()

        assert num_params > 0
        assert isinstance(num_params, int)


class TestPhonemeAwareLoss:
    """Tests for PhonemeAwareLoss."""

    def test_forward(self, batch_size, embed_dim, num_speakers, sample_labels):
        """Test loss computation."""
        criterion = PhonemeAwareLoss()

        speaker_logits = torch.randn(batch_size, num_speakers)
        speaker_embeddings = torch.randn(batch_size, embed_dim)
        phoneme_embeddings = torch.randn(batch_size, embed_dim)

        loss, loss_dict = criterion(
            speaker_logits=speaker_logits,
            speaker_embeddings=speaker_embeddings,
            phoneme_embeddings=phoneme_embeddings,
            speaker_labels=sample_labels,
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert 'speaker_loss' in loss_dict
        assert 'contrastive_loss' in loss_dict
        assert not torch.isnan(loss)


class TestAdaptiveMergingModule:
    """Tests for AdaptiveMergingModule."""

    def test_forward(self, batch_size, embed_dim):
        """Test merge prediction."""
        module = AdaptiveMergingModule(embed_dim=embed_dim)

        num_segments = 10
        segment_embeddings = torch.randn(batch_size, num_segments, embed_dim)
        phoneme_embeddings = torch.randn(batch_size, num_segments, embed_dim)

        merge_probs = module(segment_embeddings, phoneme_embeddings)

        assert merge_probs.shape == (batch_size, num_segments - 1)
        assert (merge_probs >= 0).all() and (merge_probs <= 1).all()


class TestCurriculumScheduler:
    """Tests for CurriculumScheduler."""

    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = CurriculumScheduler(
            num_stages=3,
            stage_epochs=[10, 15, 25],
        )

        assert scheduler.num_stages == 3
        assert len(scheduler.stage_epochs) == 3

    def test_step(self):
        """Test curriculum stage progression."""
        scheduler = CurriculumScheduler(
            num_stages=3,
            stage_epochs=[10, 15, 25],
        )

        # Test stage 0
        stage = scheduler.step(5)
        assert stage == 0

        # Test stage 1
        stage = scheduler.step(15)
        assert stage == 1

        # Test stage 2
        stage = scheduler.step(30)
        assert stage == 2

    def test_get_difficulty(self):
        """Test difficulty parameters."""
        scheduler = CurriculumScheduler()

        difficulty = scheduler.get_difficulty()

        assert 'stage' in difficulty
        assert 'overlap_ratio' in difficulty
        assert 'augmentation_strength' in difficulty
