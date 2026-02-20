"""Tests for data loading and preprocessing."""

import pytest
import torch

from phoneme_aware_curriculum_diarization_with_adaptive_merging.data.loader import CommonVoiceDataset
from phoneme_aware_curriculum_diarization_with_adaptive_merging.data.preprocessing import AudioPreprocessor


class TestAudioPreprocessor:
    """Tests for AudioPreprocessor class."""

    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = AudioPreprocessor(
            sample_rate=16000,
            n_mels=80,
            duration=3.0,
        )
        assert preprocessor.sample_rate == 16000
        assert preprocessor.n_mels == 80
        assert preprocessor.duration == 3.0

    def test_extract_features(self, sample_waveform):
        """Test feature extraction."""
        waveform, sr = sample_waveform
        preprocessor = AudioPreprocessor(sample_rate=sr)

        features = preprocessor.extract_features(waveform)

        assert features.dim() == 2
        assert features.shape[0] == preprocessor.n_mels
        assert features.shape[1] > 0

    def test_segment_audio(self, sample_waveform):
        """Test audio segmentation."""
        waveform, sr = sample_waveform
        preprocessor = AudioPreprocessor(sample_rate=sr, duration=1.0)

        segments = preprocessor.segment_audio(waveform, overlap=0.5)

        assert len(segments) > 0
        for segment in segments:
            assert segment.shape[0] == 1
            assert segment.shape[1] == int(sr * 1.0)

    def test_add_noise(self, sample_waveform):
        """Test noise augmentation."""
        waveform, sr = sample_waveform
        preprocessor = AudioPreprocessor(sample_rate=sr)

        noisy_waveform = preprocessor.add_noise(waveform, snr_db=15.0)

        assert noisy_waveform.shape == waveform.shape
        assert not torch.allclose(noisy_waveform, waveform)

    def test_normalize(self, sample_waveform):
        """Test waveform normalization."""
        waveform, sr = sample_waveform
        preprocessor = AudioPreprocessor(sample_rate=sr)

        normalized = preprocessor.normalize(waveform)

        assert normalized.shape == waveform.shape
        assert torch.abs(torch.mean(normalized)) < 1e-5
        assert torch.abs(torch.std(normalized) - 1.0) < 1e-1


class TestCommonVoiceDataset:
    """Tests for CommonVoiceDataset class."""

    def test_initialization(self):
        """Test dataset initialization."""
        dataset = CommonVoiceDataset(
            split='train',
            max_samples=10,
        )
        assert len(dataset) > 0

    def test_getitem(self):
        """Test dataset item retrieval."""
        dataset = CommonVoiceDataset(
            split='train',
            max_samples=10,
        )

        sample = dataset[0]

        assert 'features' in sample
        assert 'speaker_id' in sample
        assert 'phoneme_features' in sample
        assert 'waveform' in sample

        assert isinstance(sample['features'], torch.Tensor)
        assert isinstance(sample['speaker_id'], torch.Tensor)
        assert sample['features'].dim() == 2

    def test_length(self):
        """Test dataset length."""
        max_samples = 50
        dataset = CommonVoiceDataset(
            split='train',
            max_samples=max_samples,
        )

        assert len(dataset) <= max_samples
