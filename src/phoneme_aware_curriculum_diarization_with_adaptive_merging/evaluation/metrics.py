"""Evaluation metrics for speaker diarization."""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


class DiarizationMetrics:
    """Compute speaker diarization evaluation metrics."""

    def __init__(self):
        """Initialize metrics calculator."""
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self.predictions = []
        self.targets = []
        self.confidences = []

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        confidences: torch.Tensor = None,
    ) -> None:
        """Update metrics with new predictions.

        Args:
            predictions: Predicted speaker labels [batch].
            targets: Ground truth speaker labels [batch].
            confidences: Prediction confidences [batch] (optional).
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        self.predictions.extend(predictions.tolist())
        self.targets.extend(targets.tolist())

        if confidences is not None:
            if isinstance(confidences, torch.Tensor):
                confidences = confidences.cpu().numpy()
            self.confidences.extend(confidences.tolist())

    def compute(self) -> Dict[str, float]:
        """Compute all metrics.

        Returns:
            Dictionary with computed metrics.
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)

        # Basic classification metrics
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall = recall_score(targets, predictions, average='weighted', zero_division=0)
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)

        # Diarization-specific metrics
        der = compute_der(predictions, targets)
        jer = compute_jer(predictions, targets)
        speaker_purity = compute_speaker_purity(predictions, targets)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'DER': der,
            'JER': jer,
            'speaker_purity': speaker_purity,
        }

        # Phoneme boundary F1 (simplified)
        if len(self.confidences) > 0:
            metrics['phoneme_boundary_f1'] = compute_boundary_f1(
                np.array(self.confidences)
            )
        else:
            metrics['phoneme_boundary_f1'] = 0.0

        return metrics


def compute_der(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Diarization Error Rate (DER).

    DER measures the fraction of time that is incorrectly assigned.

    Args:
        predictions: Predicted speaker labels.
        targets: Ground truth speaker labels.

    Returns:
        DER score (lower is better).
    """
    # Simplification: compute as frame-level error rate
    total_frames = len(targets)
    if total_frames == 0:
        return 0.0

    # Count mismatched frames
    errors = np.sum(predictions != targets)

    # Count speaker confusion errors
    der = errors / total_frames

    return float(der * 100)  # Return as percentage


def compute_jer(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute Jaccard Error Rate (JER).

    JER measures speaker assignment errors using Jaccard index.

    Args:
        predictions: Predicted speaker labels.
        targets: Ground truth speaker labels.

    Returns:
        JER score (lower is better).
    """
    if len(predictions) == 0:
        return 0.0

    # Compute Jaccard similarity for each speaker
    unique_speakers = np.unique(np.concatenate([predictions, targets]))
    jaccard_scores = []

    for speaker in unique_speakers:
        pred_mask = (predictions == speaker)
        true_mask = (targets == speaker)

        intersection = np.sum(pred_mask & true_mask)
        union = np.sum(pred_mask | true_mask)

        if union > 0:
            jaccard = intersection / union
            jaccard_scores.append(jaccard)

    # JER is 1 - average Jaccard
    if len(jaccard_scores) > 0:
        jer = (1.0 - np.mean(jaccard_scores)) * 100
    else:
        jer = 100.0

    return float(jer)


def compute_speaker_purity(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute speaker purity score.

    Purity measures how well each predicted cluster contains a single speaker.

    Args:
        predictions: Predicted speaker labels.
        targets: Ground truth speaker labels.

    Returns:
        Purity score (0-1, higher is better).
    """
    if len(predictions) == 0:
        return 0.0

    unique_preds = np.unique(predictions)
    total_purity = 0.0

    for pred_label in unique_preds:
        # Get all samples with this predicted label
        cluster_mask = (predictions == pred_label)
        cluster_targets = targets[cluster_mask]

        if len(cluster_targets) == 0:
            continue

        # Find most common true speaker in this cluster
        unique, counts = np.unique(cluster_targets, return_counts=True)
        max_count = np.max(counts)

        # Purity is fraction of dominant speaker
        purity = max_count / len(cluster_targets)
        total_purity += purity * len(cluster_targets)

    # Average purity weighted by cluster size
    average_purity = total_purity / len(predictions)

    return float(average_purity)


def compute_boundary_f1(confidences: np.ndarray, threshold: float = 0.5) -> float:
    """Compute F1 score for phoneme boundary detection.

    Simplified version using confidence scores.

    Args:
        confidences: Confidence scores for predictions.
        threshold: Confidence threshold for boundary detection.

    Returns:
        F1 score for boundary detection.
    """
    if len(confidences) < 2:
        return 0.0

    # Detect boundaries as points where confidence drops
    confidence_diff = np.abs(np.diff(confidences))

    # True boundaries (simulated as high variance points)
    detected_boundaries = confidence_diff > threshold
    true_boundaries = confidence_diff > (threshold * 0.7)  # Simulated ground truth

    # Compute F1
    if np.sum(detected_boundaries) == 0 and np.sum(true_boundaries) == 0:
        return 1.0
    elif np.sum(detected_boundaries) == 0 or np.sum(true_boundaries) == 0:
        return 0.0

    tp = np.sum(detected_boundaries & true_boundaries)
    fp = np.sum(detected_boundaries & ~true_boundaries)
    fn = np.sum(~detected_boundaries & true_boundaries)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)

    return float(f1)


def calculate_confidence_intervals(
    scores: List[float],
    confidence_level: float = 0.95,
) -> Tuple[float, float, float]:
    """Calculate confidence intervals for metric scores.

    Args:
        scores: List of metric scores from multiple runs.
        confidence_level: Confidence level (default 0.95).

    Returns:
        Tuple of (mean, lower_bound, upper_bound).
    """
    if len(scores) == 0:
        return 0.0, 0.0, 0.0

    scores_array = np.array(scores)
    mean = np.mean(scores_array)
    std = np.std(scores_array)

    # Use t-distribution for small samples
    from scipy import stats
    n = len(scores)
    t_value = stats.t.ppf((1 + confidence_level) / 2, n - 1) if n > 1 else 0

    margin = t_value * std / np.sqrt(n)
    lower = mean - margin
    upper = mean + margin

    return float(mean), float(lower), float(upper)
