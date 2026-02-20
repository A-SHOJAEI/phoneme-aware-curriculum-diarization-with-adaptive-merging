# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Training

### Train with default config (phoneme-aware + adaptive merging):
```bash
python scripts/train.py
```

### Train with ablation config (baseline without phoneme awareness):
```bash
python scripts/train.py --config configs/ablation.yaml
```

### Quick debug training (reduced dataset):
```bash
python scripts/train.py --debug
```

## Evaluation

### Evaluate on test set:
```bash
python scripts/evaluate.py --checkpoint models/best_model.pt --split test
```

### Evaluate on validation set:
```bash
python scripts/evaluate.py --checkpoint models/best_model.pt --split val
```

## Prediction

### Run inference on audio file:
```bash
python scripts/predict.py --checkpoint models/best_model.pt --audio path/to/audio.wav
```

### Save predictions to file:
```bash
python scripts/predict.py --checkpoint models/best_model.pt --audio path/to/audio.wav --output predictions.json
```

## Testing

### Run all tests:
```bash
python -m pytest tests/ -v
```

### Run specific test file:
```bash
python -m pytest tests/test_model.py -v
```

## Project Structure

```
.
├── configs/
│   ├── default.yaml      # Full model with phoneme awareness
│   └── ablation.yaml     # Baseline without adaptive merging
├── scripts/
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   └── predict.py        # Inference script
├── src/
│   └── phoneme_aware_curriculum_diarization_with_adaptive_merging/
│       ├── data/         # Data loading and preprocessing
│       ├── models/       # Model architecture
│       │   ├── model.py       # Main model
│       │   └── components.py  # Custom components (losses, layers)
│       ├── training/     # Training utilities
│       ├── evaluation/   # Metrics and analysis
│       └── utils/        # Configuration utilities
└── tests/                # Unit tests
```

## Key Features

1. **Dual-Encoder Architecture**
   - Speaker encoder (Transformer-based)
   - Phoneme encoder (LSTM-based)

2. **Custom Components**
   - PhonemeAwareLoss: Multi-objective loss function
   - AdaptiveMergingModule: Boundary refinement
   - CurriculumScheduler: Progressive difficulty

3. **Ablation Studies**
   - Compare full model vs baseline
   - Configurable via YAML files

## Outputs

- **Checkpoints**: Saved to `checkpoints/` or `models/`
- **Training history**: `results/training_history.json`
- **Evaluation metrics**: `results/evaluation_metrics.{json,csv}`
- **Evaluation report**: `results/evaluation_report.txt`
- **Predictions**: JSON format with segment-level diarization
