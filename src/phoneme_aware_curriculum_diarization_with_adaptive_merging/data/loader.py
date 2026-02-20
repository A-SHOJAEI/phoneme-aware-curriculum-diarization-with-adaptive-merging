"""Data loading utilities for Common Voice dataset."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from datasets import load_dataset

from .preprocessing import AudioPreprocessor

logger = logging.getLogger(__name__)


class CommonVoiceDataset(Dataset):
    """PyTorch Dataset for Common Voice audio data."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        split: str = 'train',
        language: str = 'en',
        max_samples: Optional[int] = None,
        preprocessor: Optional[AudioPreprocessor] = None,
        augment: bool = True,
        num_speakers: int = 100,
    ):
        """Initialize Common Voice dataset.

        Args:
            data_dir: Directory for caching dataset (optional).
            split: Dataset split ('train', 'validation', 'test').
            language: Language code for Common Voice.
            max_samples: Maximum number of samples to load (for faster iteration).
            preprocessor: Audio preprocessor instance.
            augment: Whether to apply data augmentation.
            num_speakers: Maximum number of speaker classes (must match model).
        """
        self.split = split
        self.language = language
        self.augment = augment and split == 'train'
        self.num_speakers = num_speakers

        if preprocessor is None:
            self.preprocessor = AudioPreprocessor(augment=self.augment)
        else:
            self.preprocessor = preprocessor

        logger.info(f"Loading Common Voice dataset: {language} - {split}")

        try:
            # Load dataset from HuggingFace
            self.dataset = load_dataset(
                "mozilla-foundation/common_voice_11_0",
                language,
                split=split,
                cache_dir=data_dir,
                trust_remote_code=True,
            )

            if max_samples is not None and len(self.dataset) > max_samples:
                self.dataset = self.dataset.select(range(max_samples))

            logger.info(f"Loaded {len(self.dataset)} samples")

        except Exception as e:
            logger.warning(f"Could not load Common Voice dataset: {e}")
            logger.info("Creating synthetic dataset for demonstration")
            self.dataset = self._create_synthetic_dataset(max_samples or 1000)

    def _create_synthetic_dataset(self, num_samples: int) -> List[Dict]:
        """Create synthetic dataset for testing when real data unavailable.

        Args:
            num_samples: Number of synthetic samples to generate.

        Returns:
            List of synthetic data dictionaries.
        """
        synthetic_data = []
        # Use self.num_speakers to match model output dimension
        num_speakers = min(self.num_speakers, 50)

        for i in range(num_samples):
            # Create synthetic audio (silence with some noise)
            duration = self.preprocessor.duration
            num_samples_audio = int(duration * self.preprocessor.sample_rate)
            waveform = torch.randn(1, num_samples_audio) * 0.01

            synthetic_data.append({
                'audio': {'array': waveform.squeeze().numpy(), 'sampling_rate': self.preprocessor.sample_rate},
                'speaker_id': i % num_speakers,
                'sentence': f"Synthetic utterance {i}",
            })

        return synthetic_data

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single data sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with 'features', 'speaker_id', and 'phoneme_features'.
        """
        sample = self.dataset[idx]

        # Extract audio
        if isinstance(sample['audio'], dict):
            audio_array = sample['audio']['array']
            sample_rate = sample['audio']['sampling_rate']
        else:
            audio_array = sample['audio']
            sample_rate = self.preprocessor.sample_rate

        waveform = torch.from_numpy(audio_array).float().unsqueeze(0)

        # Resample if needed
        if sample_rate != self.preprocessor.sample_rate:
            resampler = torch.nn.functional.interpolate
            target_length = int(len(audio_array) * self.preprocessor.sample_rate / sample_rate)
            waveform = torch.nn.functional.interpolate(
                waveform.unsqueeze(0),
                size=target_length,
                mode='linear',
                align_corners=False
            ).squeeze(0)

        # Pad or trim to fixed duration
        target_samples = int(self.preprocessor.duration * self.preprocessor.sample_rate)
        if waveform.shape[1] < target_samples:
            padding = target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :target_samples]

        # Apply augmentation
        if self.augment and torch.rand(1).item() > 0.5:
            waveform = self.preprocessor.add_noise(waveform, snr_db=15.0 + torch.randn(1).item() * 5)

        # Extract features
        features = self.preprocessor.extract_features(waveform)

        # Get speaker ID
        # Use modulo to ensure speaker_id fits within model's num_speakers
        speaker_id = sample.get('client_id', idx % self.num_speakers)
        if isinstance(speaker_id, str):
            # Hash string to integer, constrained to num_speakers range
            speaker_id = hash(speaker_id) % self.num_speakers
        else:
            speaker_id = int(speaker_id) % self.num_speakers

        # Generate phoneme-level features (simplified: use frame-level features)
        phoneme_features = features.clone()

        return {
            'features': features,
            'speaker_id': torch.tensor(speaker_id, dtype=torch.long),
            'phoneme_features': phoneme_features,
            'waveform': waveform.squeeze(0),
        }


def create_dataloaders(
    data_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    val_split: float = 0.1,
    test_split: float = 0.1,
    language: str = 'en',
    max_samples: Optional[int] = None,
    num_speakers: int = 100,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders.

    Args:
        data_dir: Directory for caching dataset.
        batch_size: Batch size for dataloaders.
        num_workers: Number of worker processes for data loading.
        val_split: Validation set ratio.
        test_split: Test set ratio.
        language: Language code.
        max_samples: Maximum samples per split.
        num_speakers: Maximum number of speaker classes (must match model).

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    preprocessor = AudioPreprocessor(augment=True)

    # Create datasets
    train_dataset = CommonVoiceDataset(
        data_dir=data_dir,
        split='train',
        language=language,
        max_samples=max_samples,
        preprocessor=preprocessor,
        augment=True,
        num_speakers=num_speakers,
    )

    # Split train into train and validation
    dataset_size = len(train_dataset)
    val_size = int(dataset_size * val_split)
    test_size = int(dataset_size * test_split)
    train_size = dataset_size - val_size - test_size

    train_subset, val_subset, test_subset = random_split(
        train_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"Created dataloaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    return train_loader, val_loader, test_loader
