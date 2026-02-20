"""Trainer class for phoneme-aware diarization model with curriculum learning."""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.components import CurriculumScheduler, PhonemeAwareLoss
from ..models.model import DualEncoderDiarizationModel

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for dual-encoder diarization model with curriculum learning."""

    def __init__(
        self,
        model: DualEncoderDiarizationModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: torch.device,
        checkpoint_dir: str = 'checkpoints',
    ):
        """Initialize trainer.

        Args:
            model: Model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            config: Training configuration dictionary.
            device: Device to train on.
            checkpoint_dir: Directory to save checkpoints.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training parameters
        self.num_epochs = config.get('num_epochs', 50)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 0.0001)
        self.gradient_clip = config.get('gradient_clip', 1.0)
        self.patience = config.get('early_stopping_patience', 10)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Learning rate scheduler
        scheduler_type = config.get('scheduler', 'cosine')
        if scheduler_type == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True,
            )
        else:
            self.scheduler = None

        # Loss function
        self.criterion = PhonemeAwareLoss(
            speaker_weight=config.get('speaker_weight', 1.0),
            phoneme_weight=config.get('phoneme_weight', 0.5),
            boundary_weight=config.get('boundary_weight', 0.3),
        )

        # Curriculum scheduler
        self.curriculum = CurriculumScheduler(
            num_stages=config.get('curriculum_stages', 3),
            stage_epochs=config.get('stage_epochs', [10, 15, 25]),
        )

        # Mixed precision training
        self.use_amp = config.get('use_amp', True) and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            logger.info("Using mixed precision training")

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }

        # MLflow tracking
        self.use_mlflow = config.get('use_mlflow', False)
        if self.use_mlflow:
            try:
                import mlflow
                self.mlflow = mlflow
                mlflow.set_experiment(config.get('experiment_name', 'phoneme_diarization'))
                logger.info("MLflow tracking enabled")
            except ImportError:
                logger.warning("MLflow not available, disabling tracking")
                self.use_mlflow = False
            except Exception as e:
                logger.warning(f"MLflow setup failed: {e}")
                self.use_mlflow = False

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary with training metrics.
        """
        self.model.train()
        total_loss = 0.0
        loss_components = {'speaker_loss': 0.0, 'contrastive_loss': 0.0, 'boundary_loss': 0.0}

        # Update curriculum stage
        curriculum_stage = self.curriculum.step(self.current_epoch)
        difficulty = self.curriculum.get_difficulty()

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} (Stage {curriculum_stage})")

        for batch_idx, batch in enumerate(pbar):
            features = batch['features'].to(self.device)
            speaker_labels = batch['speaker_id'].to(self.device)
            phoneme_features = batch['phoneme_features'].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(features)
                    loss, loss_dict = self.criterion(
                        speaker_logits=outputs['speaker_logits'],
                        speaker_embeddings=outputs['speaker_embeddings'],
                        phoneme_embeddings=outputs['phoneme_embeddings'],
                        speaker_labels=speaker_labels,
                    )

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(features)
                loss, loss_dict = self.criterion(
                    speaker_logits=outputs['speaker_logits'],
                    speaker_embeddings=outputs['speaker_embeddings'],
                    phoneme_embeddings=outputs['phoneme_embeddings'],
                    speaker_labels=speaker_labels,
                )

                loss.backward()

                if self.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                self.optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key in loss_components:
                    loss_components[key] += value

            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'stage': curriculum_stage})

        # Average losses
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches

        return {'total_loss': avg_loss, **loss_components}

    def validate(self) -> Dict[str, float]:
        """Validate model on validation set.

        Returns:
            Dictionary with validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                features = batch['features'].to(self.device)
                speaker_labels = batch['speaker_id'].to(self.device)
                phoneme_features = batch['phoneme_features'].to(self.device)

                outputs = self.model(features)
                loss, _ = self.criterion(
                    speaker_logits=outputs['speaker_logits'],
                    speaker_embeddings=outputs['speaker_embeddings'],
                    phoneme_embeddings=outputs['phoneme_embeddings'],
                    speaker_labels=speaker_labels,
                )

                total_loss += loss.item()

                # Calculate accuracy
                predictions = torch.argmax(outputs['speaker_logits'], dim=1)
                correct += (predictions == speaker_labels).sum().item()
                total += speaker_labels.size(0)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0.0

        return {'val_loss': avg_loss, 'accuracy': accuracy}

    def train(self) -> Dict[str, list]:
        """Full training loop with early stopping.

        Returns:
            Training history dictionary.
        """
        logger.info("Starting training...")

        if self.use_mlflow:
            try:
                self.mlflow.start_run()
                self.mlflow.log_params({
                    'learning_rate': self.learning_rate,
                    'num_epochs': self.num_epochs,
                    'batch_size': self.train_loader.batch_size,
                    'model_params': self.model.count_parameters(),
                })
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")

        try:
            for epoch in range(self.num_epochs):
                self.current_epoch = epoch

                # Train epoch
                train_metrics = self.train_epoch()

                # Validate
                val_metrics = self.validate()

                # Update learning rate
                if self.scheduler is not None:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['val_loss'])
                    else:
                        self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]['lr']

                # Log metrics
                logger.info(
                    f"Epoch {epoch + 1}/{self.num_epochs} - "
                    f"Train Loss: {train_metrics['total_loss']:.4f}, "
                    f"Val Loss: {val_metrics['val_loss']:.4f}, "
                    f"Accuracy: {val_metrics['accuracy']:.4f}, "
                    f"LR: {current_lr:.6f}"
                )

                # Update history
                self.history['train_loss'].append(train_metrics['total_loss'])
                self.history['val_loss'].append(val_metrics['val_loss'])
                self.history['learning_rate'].append(current_lr)

                # MLflow logging
                if self.use_mlflow:
                    try:
                        self.mlflow.log_metrics({
                            'train_loss': train_metrics['total_loss'],
                            'val_loss': val_metrics['val_loss'],
                            'accuracy': val_metrics['accuracy'],
                            'learning_rate': current_lr,
                        }, step=epoch)
                    except Exception as e:
                        logger.warning(f"MLflow logging failed: {e}")

                # Save best model
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.patience_counter = 0
                    self.save_checkpoint('best_model.pt', is_best=True)
                    logger.info(f"Saved best model with val_loss: {self.best_val_loss:.4f}")
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

                # Save periodic checkpoint
                if (epoch + 1) % 10 == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')

        finally:
            if self.use_mlflow:
                try:
                    self.mlflow.end_run()
                except Exception:
                    pass

        logger.info("Training completed!")
        return self.history

    def save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename.
            is_best: Whether this is the best model so far.
        """
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config,
            'model_config': self.model.get_config(),
        }

        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Also save to models/ directory if best
        if is_best:
            best_path = Path('models') / 'best_model.pt'
            best_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.use_amp and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
