"""Tests for training utilities."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from phoneme_aware_curriculum_diarization_with_adaptive_merging.models.model import DualEncoderDiarizationModel
from phoneme_aware_curriculum_diarization_with_adaptive_merging.training.trainer import Trainer


class TestTrainer:
    """Tests for Trainer class."""

    @pytest.fixture
    def mock_dataloader(self, batch_size):
        """Create mock dataloader for testing."""
        num_samples = 32
        features = torch.randn(num_samples, 80, 100)
        speaker_ids = torch.randint(0, 10, (num_samples,))
        phoneme_features = torch.randn(num_samples, 80, 100)
        waveforms = torch.randn(num_samples, 48000)

        # Create dataset with dictionary-like samples
        class MockDataset:
            def __len__(self):
                return num_samples

            def __getitem__(self, idx):
                return {
                    'features': features[idx],
                    'speaker_id': speaker_ids[idx],
                    'phoneme_features': phoneme_features[idx],
                    'waveform': waveforms[idx],
                }

        dataset = MockDataset()
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def test_initialization(self, mock_dataloader, device, sample_config):
        """Test trainer initialization."""
        model = DualEncoderDiarizationModel(
            input_dim=80,
            embed_dim=128,
            num_speakers=10,
        )

        trainer = Trainer(
            model=model,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            config=sample_config['training'],
            device=device,
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None

    def test_train_epoch(self, mock_dataloader, device, sample_config):
        """Test single training epoch."""
        model = DualEncoderDiarizationModel(
            input_dim=80,
            embed_dim=128,
            num_speakers=10,
        )

        config = sample_config['training'].copy()
        config['use_amp'] = False  # Disable AMP for CPU testing

        trainer = Trainer(
            model=model,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            config=config,
            device=device,
        )

        metrics = trainer.train_epoch()

        assert 'total_loss' in metrics
        assert isinstance(metrics['total_loss'], float)
        assert metrics['total_loss'] > 0

    def test_validate(self, mock_dataloader, device, sample_config):
        """Test validation."""
        model = DualEncoderDiarizationModel(
            input_dim=80,
            embed_dim=128,
            num_speakers=10,
        )

        config = sample_config['training'].copy()
        config['use_amp'] = False

        trainer = Trainer(
            model=model,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            config=config,
            device=device,
        )

        metrics = trainer.validate()

        assert 'val_loss' in metrics
        assert 'accuracy' in metrics
        assert isinstance(metrics['val_loss'], float)
        assert 0.0 <= metrics['accuracy'] <= 1.0

    def test_save_load_checkpoint(self, mock_dataloader, device, sample_config, tmp_path):
        """Test checkpoint saving and loading."""
        model = DualEncoderDiarizationModel(
            input_dim=80,
            embed_dim=128,
            num_speakers=10,
        )

        config = sample_config['training'].copy()
        config['use_amp'] = False

        trainer = Trainer(
            model=model,
            train_loader=mock_dataloader,
            val_loader=mock_dataloader,
            config=config,
            device=device,
            checkpoint_dir=str(tmp_path),
        )

        # Save checkpoint
        checkpoint_path = tmp_path / 'test_checkpoint.pt'
        trainer.save_checkpoint('test_checkpoint.pt')

        assert checkpoint_path.exists()

        # Load checkpoint
        trainer.load_checkpoint(str(checkpoint_path))

        assert trainer.current_epoch == 0
