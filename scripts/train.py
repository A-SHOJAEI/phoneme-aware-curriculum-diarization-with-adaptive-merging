#!/usr/bin/env python
"""Training script for phoneme-aware curriculum diarization model."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import yaml

from phoneme_aware_curriculum_diarization_with_adaptive_merging.data.loader import create_dataloaders
from phoneme_aware_curriculum_diarization_with_adaptive_merging.models.model import DualEncoderDiarizationModel
from phoneme_aware_curriculum_diarization_with_adaptive_merging.training.trainer import Trainer
from phoneme_aware_curriculum_diarization_with_adaptive_merging.utils.config import get_device, set_random_seeds

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log'),
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train phoneme-aware speaker diarization model"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file',
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume training',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for training',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with reduced dataset',
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file.

    Returns:
        Configuration dictionary.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Set random seeds
    seed = config.get('seed', 42)
    set_random_seeds(seed)

    # Determine device
    if args.device == 'auto':
        device = get_device(prefer_cuda=True)
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    # Debug mode - reduce dataset size
    if args.debug:
        logger.info("Debug mode enabled - using reduced dataset")
        config['data']['max_samples'] = 100
        config['training']['num_epochs'] = 5

    try:
        # Create dataloaders
        logger.info("Creating dataloaders...")
        num_speakers = config.get('model', {}).get('num_speakers', 100)
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=config.get('data', {}).get('data_dir', None),
            batch_size=config.get('data', {}).get('batch_size', 32),
            num_workers=config.get('data', {}).get('num_workers', 4),
            val_split=config.get('data', {}).get('val_split', 0.1),
            test_split=config.get('data', {}).get('test_split', 0.1),
            language=config.get('data', {}).get('language', 'en'),
            max_samples=config.get('data', {}).get('max_samples', None),
            num_speakers=num_speakers,
        )

        logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        # Create model
        logger.info("Creating model...")
        model_config = config.get('model', {})
        model = DualEncoderDiarizationModel(
            input_dim=model_config.get('input_dim', 80),
            embed_dim=model_config.get('embed_dim', 256),
            num_speakers=model_config.get('num_speakers', 100),
            speaker_layers=model_config.get('speaker_layers', 3),
            phoneme_layers=model_config.get('phoneme_layers', 2),
            dropout=model_config.get('dropout', 0.1),
            use_adaptive_merging=model_config.get('use_adaptive_merging', True),
        )

        logger.info(f"Model has {model.count_parameters():,} parameters")

        # Create trainer
        logger.info("Creating trainer...")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config.get('training', {}),
            device=device,
            checkpoint_dir=config.get('checkpoint_dir', 'checkpoints'),
        )

        # Load checkpoint if provided
        if args.checkpoint:
            logger.info(f"Loading checkpoint from {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint)

        # Train model
        logger.info("Starting training...")
        history = trainer.train()

        # Save final results
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")

        # Save training history
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)

        import json
        history_path = results_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")

        logger.info("=" * 60)
        logger.info("Training Summary:")
        logger.info(f"  Total epochs: {len(history['train_loss'])}")
        logger.info(f"  Best val loss: {trainer.best_val_loss:.4f}")
        logger.info(f"  Final train loss: {history['train_loss'][-1]:.4f}")
        logger.info(f"  Final val loss: {history['val_loss'][-1]:.4f}")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
