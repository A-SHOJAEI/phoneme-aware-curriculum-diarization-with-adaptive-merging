#!/usr/bin/env python
"""Inference script for phoneme-aware diarization model."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torchaudio

from phoneme_aware_curriculum_diarization_with_adaptive_merging.data.preprocessing import AudioPreprocessor
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
        description="Run inference with phoneme-aware diarization model"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='models/best_model.pt',
        help='Path to model checkpoint',
    )
    parser.add_argument(
        '--audio',
        type=str,
        required=True,
        help='Path to input audio file',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save predictions (optional)',
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for predictions',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use for inference',
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
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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

    return model


def predict_audio(
    audio_path: str,
    model: DualEncoderDiarizationModel,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Run diarization on audio file.

    Args:
        audio_path: Path to audio file.
        model: Trained model.
        device: Device to run on.
        threshold: Confidence threshold.

    Returns:
        Dictionary with predictions.
    """
    # Initialize preprocessor
    preprocessor = AudioPreprocessor(
        sample_rate=16000,
        duration=3.0,
        augment=False,
    )

    # Load and preprocess audio
    logger.info(f"Loading audio from {audio_path}")
    waveform, sr = preprocessor.load_audio(audio_path)

    # Segment audio
    segments = preprocessor.segment_audio(waveform, overlap=0.5)
    logger.info(f"Created {len(segments)} segments from audio")

    if len(segments) == 0:
        logger.warning("No segments created from audio")
        return {'segments': [], 'speakers': []}

    # Process each segment
    predictions = []
    confidences = []
    embeddings = []

    with torch.no_grad():
        for i, segment in enumerate(segments):
            # Extract features
            features = preprocessor.extract_features(segment)
            features = features.unsqueeze(0).to(device)

            # Get prediction
            pred, conf = model.predict_speakers(features, threshold=threshold)

            # Extract embeddings
            emb = model.extract_embeddings(features)

            predictions.append(pred.item())
            confidences.append(conf.item())
            embeddings.append(emb['speaker'].cpu().numpy())

            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(segments)} segments")

    # Create time-aligned results
    results = {
        'audio_path': audio_path,
        'total_segments': len(segments),
        'unique_speakers': len(set([p for p in predictions if p >= 0])),
        'segments': [],
    }

    segment_duration = preprocessor.duration
    hop_duration = segment_duration * 0.5  # 50% overlap

    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        start_time = i * hop_duration
        end_time = start_time + segment_duration

        results['segments'].append({
            'segment_id': i,
            'start_time': round(start_time, 2),
            'end_time': round(end_time, 2),
            'speaker_id': int(pred) if pred >= 0 else -1,
            'confidence': round(conf, 4),
        })

    return results


def format_predictions(results: dict) -> str:
    """Format predictions for display.

    Args:
        results: Prediction results.

    Returns:
        Formatted string.
    """
    lines = [
        "=" * 70,
        "SPEAKER DIARIZATION RESULTS",
        "=" * 70,
        f"Audio: {results['audio_path']}",
        f"Total Segments: {results['total_segments']}",
        f"Unique Speakers Detected: {results['unique_speakers']}",
        "",
        "Time-Aligned Speaker Segments:",
        "-" * 70,
        f"{'Segment':<10} {'Start':<10} {'End':<10} {'Speaker':<15} {'Confidence':<12}",
        "-" * 70,
    ]

    for seg in results['segments']:
        speaker_label = f"Speaker {seg['speaker_id']}" if seg['speaker_id'] >= 0 else "Unknown"
        lines.append(
            f"{seg['segment_id']:<10} "
            f"{seg['start_time']:<10.2f} "
            f"{seg['end_time']:<10.2f} "
            f"{speaker_label:<15} "
            f"{seg['confidence']:<12.4f}"
        )

    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    """Main inference function."""
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
        # Check if audio file exists
        if not Path(args.audio).exists():
            logger.error(f"Audio file not found: {args.audio}")
            sys.exit(1)

        # Load model
        model = load_model(args.checkpoint, device)

        # Run prediction
        logger.info("Running speaker diarization...")
        results = predict_audio(
            audio_path=args.audio,
            model=model,
            device=device,
            threshold=args.threshold,
        )

        # Display results
        output = format_predictions(results)
        print("\n" + output)

        # Save to file if requested
        if args.output:
            import json
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

            logger.info(f"Saved predictions to {output_path}")

    except Exception as e:
        logger.error(f"Prediction failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
