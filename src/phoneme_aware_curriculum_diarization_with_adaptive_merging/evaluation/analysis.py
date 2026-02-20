"""Results analysis and visualization utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyze and visualize diarization results."""

    def __init__(self, results_dir: str = 'results'):
        """Initialize results analyzer.

        Args:
            results_dir: Directory to save results and plots.
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_metrics(
        self,
        metrics: Dict[str, float],
        filename: str = 'metrics.json',
    ) -> None:
        """Save metrics to JSON file.

        Args:
            metrics: Dictionary of metrics.
            filename: Output filename.
        """
        output_path = self.results_dir / filename

        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved metrics to {output_path}")

    def save_metrics_csv(
        self,
        metrics: Dict[str, float],
        filename: str = 'metrics.csv',
    ) -> None:
        """Save metrics to CSV file.

        Args:
            metrics: Dictionary of metrics.
            filename: Output filename.
        """
        output_path = self.results_dir / filename

        df = pd.DataFrame([metrics])
        df.to_csv(output_path, index=False)

        logger.info(f"Saved metrics to {output_path}")

    def create_summary_table(
        self,
        metrics: Dict[str, float],
    ) -> pd.DataFrame:
        """Create formatted summary table.

        Args:
            metrics: Dictionary of metrics.

        Returns:
            Formatted DataFrame.
        """
        df = pd.DataFrame([metrics])
        df = df.round(4)

        return df

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot training and validation loss curves.

        Args:
            history: Training history dictionary.
            save_path: Path to save plot (optional).
        """
        if save_path is None:
            save_path = self.results_dir / 'training_curves.png'

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curves
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Learning rate
        axes[1].plot(history['learning_rate'], linewidth=2, color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved training curves to {save_path}")

    def plot_confusion_matrix(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot confusion matrix for speaker predictions.

        Args:
            predictions: Predicted labels.
            targets: True labels.
            save_path: Path to save plot (optional).
        """
        if save_path is None:
            save_path = self.results_dir / 'confusion_matrix.png'

        from sklearn.metrics import confusion_matrix

        # Limit to top speakers for visualization
        unique_speakers = np.unique(np.concatenate([predictions, targets]))
        if len(unique_speakers) > 20:
            # Show only top 20 most frequent speakers
            speaker_counts = np.bincount(targets)
            top_speakers = np.argsort(speaker_counts)[-20:]
            mask = np.isin(targets, top_speakers) & np.isin(predictions, top_speakers)
            predictions = predictions[mask]
            targets = targets[mask]

        cm = confusion_matrix(targets, predictions)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True)
        plt.xlabel('Predicted Speaker')
        plt.ylabel('True Speaker')
        plt.title('Speaker Confusion Matrix')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved confusion matrix to {save_path}")

    def plot_metric_comparison(
        self,
        baseline_metrics: Dict[str, float],
        model_metrics: Dict[str, float],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot comparison between baseline and model metrics.

        Args:
            baseline_metrics: Baseline model metrics.
            model_metrics: Current model metrics.
            save_path: Path to save plot (optional).
        """
        if save_path is None:
            save_path = self.results_dir / 'metric_comparison.png'

        # Select key metrics
        key_metrics = ['DER', 'JER', 'speaker_purity', 'f1_score']
        baseline_values = [baseline_metrics.get(m, 0) for m in key_metrics]
        model_values = [model_metrics.get(m, 0) for m in key_metrics]

        x = np.arange(len(key_metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8)
        ax.bar(x + width/2, model_values, width, label='Our Model', alpha=0.8)

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Baseline vs. Phoneme-Aware Model')
        ax.set_xticks(x)
        ax.set_xticklabels(key_metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved metric comparison to {save_path}")

    def generate_report(
        self,
        metrics: Dict[str, float],
        history: Optional[Dict[str, List[float]]] = None,
    ) -> str:
        """Generate text report of results.

        Args:
            metrics: Evaluation metrics.
            history: Training history (optional).

        Returns:
            Report as string.
        """
        report_lines = [
            "=" * 60,
            "PHONEME-AWARE DIARIZATION EVALUATION REPORT",
            "=" * 60,
            "",
            "Diarization Metrics:",
            f"  DER (Diarization Error Rate): {metrics.get('DER', 0):.2f}%",
            f"  JER (Jaccard Error Rate): {metrics.get('JER', 0):.2f}%",
            f"  Speaker Purity: {metrics.get('speaker_purity', 0):.4f}",
            "",
            "Classification Metrics:",
            f"  Accuracy: {metrics.get('accuracy', 0):.4f}",
            f"  Precision: {metrics.get('precision', 0):.4f}",
            f"  Recall: {metrics.get('recall', 0):.4f}",
            f"  F1 Score: {metrics.get('f1_score', 0):.4f}",
            "",
            "Phoneme-Level Metrics:",
            f"  Boundary F1: {metrics.get('phoneme_boundary_f1', 0):.4f}",
            "",
        ]

        if history:
            report_lines.extend([
                "Training Summary:",
                f"  Total Epochs: {len(history.get('train_loss', []))}",
                f"  Final Train Loss: {history['train_loss'][-1]:.4f}" if history.get('train_loss') else "",
                f"  Final Val Loss: {history['val_loss'][-1]:.4f}" if history.get('val_loss') else "",
                "",
            ])

        report_lines.append("=" * 60)

        report = "\n".join(report_lines)

        # Save report
        report_path = self.results_dir / 'evaluation_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Saved evaluation report to {report_path}")

        return report
