#!/usr/bin/env python
"""Evaluation script for phoneme-aware diarization model."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import yaml

from phoneme_aware_curriculum_diarization_with_adaptive_merging.data.loader import create_dataloaders
from phoneme_aware_curriculum_diarization_with_adaptive_merging.evaluation.analysis import ResultsAnalyzer
from phoneme_aware_curriculum_diarization_with_adaptive_merging.evaluation.metrics import DiarizationMetrics
from phoneme_aware_curriculum_diarization_with_adaptive_merging.models.model import DualEncoderDiarizationModel
from phoneme_aware_curriculum_diarization_with_adaptive_merging.utils.config import get_device, set_random_seeds

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate phoneme-aware speaker diarization model"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/best_model.pt',
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save evaluation results',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for evaluation',
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to evaluate on',
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> DualEncoderDiarizationModel:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model on.

    Returns:
        Loaded model.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config from checkpoint
    model_config = checkpoint.get('model_config', {})

    # Create model
    model = DualEncoderDiarizationModel(
        input_dim=model_config.get('input_dim', 80),
        embed_dim=model_config.get('embed_dim', 256),
        num_speakers=model_config.get('num_speakers', 100),
        use_adaptive_merging=model_config.get('use_adaptive_merging', True),
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Model has {model.count_parameters():,} parameters")

    return model


def evaluate_model(
    model: DualEncoderDiarizationModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model on dataset.

    Args:
        model: Model to evaluate.
        dataloader: Data loader.
        device: Device to run evaluation on.

    Returns:
        Dictionary of evaluation metrics.
    """
    metrics_calculator = DiarizationMetrics()

    all_predictions = []
    all_targets = []
    all_confidences = []

    logger.info("Running evaluation...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            features = batch['features'].to(device)
            speaker_labels = batch['speaker_id'].to(device)

            # Get predictions
            predictions, confidences = model.predict_speakers(features)

            # Update metrics
            metrics_calculator.update(predictions, speaker_labels, confidences)

            # Store for detailed analysis
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(speaker_labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")

    # Compute all metrics
    metrics = metrics_calculator.compute()

    # Add additional statistics
    metrics['num_samples'] = len(all_predictions)
    metrics['num_unique_speakers_pred'] = len(np.unique(all_predictions))
    metrics['num_unique_speakers_true'] = len(np.unique(all_targets))
    metrics['avg_confidence'] = float(np.mean(all_confidences))

    return metrics, np.array(all_predictions), np.array(all_targets)


def main():
    """Main evaluation function."""
    args = parse_args()

    # Set random seeds
    set_random_seeds(42)

    # Determine device
    if args.device == 'auto':
        device = get_device(prefer_cuda=True)
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")

    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

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

        # Select dataloader based on split
        if args.split == 'train':
            dataloader = train_loader
        elif args.split == 'val':
            dataloader = val_loader
        else:
            dataloader = test_loader

        logger.info(f"Evaluating on {args.split} split with {len(dataloader)} batches")

        # Load model
        if not Path(args.checkpoint).exists():
            logger.error(f"Checkpoint not found: {args.checkpoint}")
            logger.info("Please train the model first using: python scripts/train.py")
            sys.exit(1)

        model = load_model(args.checkpoint, device)

        # Evaluate
        metrics, predictions, targets = evaluate_model(model, dataloader, device)

        # Create results analyzer
        analyzer = ResultsAnalyzer(results_dir=args.output_dir)

        # Save metrics
        analyzer.save_metrics(metrics, filename='evaluation_metrics.json')
        analyzer.save_metrics_csv(metrics, filename='evaluation_metrics.csv')

        # Generate report
        report = analyzer.generate_report(metrics)
        print("\n" + report)

        # Create visualizations
        try:
            logger.info("Creating visualizations...")
            analyzer.plot_confusion_matrix(predictions, targets)
            logger.info(f"Saved visualizations to {args.output_dir}/")
        except Exception as e:
            logger.warning(f"Could not create visualizations: {e}")

        # Print summary table
        logger.info("\n" + "=" * 60)
        logger.info("EVALUATION RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Dataset Split: {args.split}")
        logger.info(f"Number of Samples: {metrics['num_samples']}")
        logger.info("")
        logger.info("Diarization Metrics:")
        logger.info(f"  DER (Diarization Error Rate): {metrics['DER']:.2f}%")
        logger.info(f"  JER (Jaccard Error Rate): {metrics['JER']:.2f}%")
        logger.info(f"  Speaker Purity: {metrics['speaker_purity']:.4f}")
        logger.info(f"  Phoneme Boundary F1: {metrics['phoneme_boundary_f1']:.4f}")
        logger.info("")
        logger.info("Classification Metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1_score']:.4f}")
        logger.info("")
        logger.info(f"Results saved to: {args.output_dir}/")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
