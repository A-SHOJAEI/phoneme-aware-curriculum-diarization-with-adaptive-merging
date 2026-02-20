"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

import pytest
import torch
import numpy as np

# Add src to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device('cpu')


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 8


@pytest.fixture
def input_dim():
    """Input feature dimension."""
    return 80


@pytest.fixture
def embed_dim():
    """Embedding dimension."""
    return 128


@pytest.fixture
def num_speakers():
    """Number of speaker classes."""
    return 10


@pytest.fixture
def sample_features(batch_size, input_dim):
    """Generate sample feature tensors."""
    time_steps = 100
    return torch.randn(batch_size, input_dim, time_steps)


@pytest.fixture
def sample_labels(batch_size, num_speakers):
    """Generate sample speaker labels."""
    return torch.randint(0, num_speakers, (batch_size,))


@pytest.fixture
def sample_waveform():
    """Generate sample audio waveform."""
    sample_rate = 16000
    duration = 3.0
    num_samples = int(sample_rate * duration)
    return torch.randn(1, num_samples), sample_rate


@pytest.fixture
def sample_config():
    """Sample configuration dictionary."""
    return {
        'seed': 42,
        'data': {
            'batch_size': 8,
            'num_workers': 0,
            'max_samples': 50,
        },
        'model': {
            'input_dim': 80,
            'embed_dim': 128,
            'num_speakers': 10,
            'speaker_layers': 2,
            'phoneme_layers': 1,
            'dropout': 0.1,
        },
        'training': {
            'num_epochs': 2,
            'learning_rate': 0.001,
            'gradient_clip': 1.0,
            'early_stopping_patience': 5,
            'use_amp': False,
        },
    }
