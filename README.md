# Sleep Transformer for Automated Sleep Staging

A PyTorch implementation of a transformer-based model for automatic sleep stage classification from EEG/EOG signals.

## Overview

This project implements the SleepTransformer architecture and compares it against the state-of-the-art U-Sleep model on a large-scale dataset of **19,000+ sleep recordings** from **15,000+ subjects** across multiple sleep centers.

**Key Results:**
- Achieves Cohen's Kappa of **0.79** (comparable to U-Sleep's 0.81)
- Strong generalization on **completely unseen datasets** (κ = 0.702)
- Maintains consistent performance across **12 different sleep study datasets**

## Architecture

The model has three main components:
1. **Epoch Transformer** - Processes 30-second spectrograms using self-attention
2. **Sequence Transformer** - Models temporal dependencies between epochs  
3. **Classification Module** - Outputs sleep stage predictions (Wake, N1, N2, N3, REM)

## Quick Start

```python
# Train the model
from sleep_transformer_main.model.SleepTransformerLightning import train
train()

# Or load a trained model
from sleep_transformer_main.model.SleepTransformerLightning import SleepTransformerLightning
model = SleepTransformerLightning.load_from_checkpoint('path/to/checkpoint.ckpt')
```

## Dataset

Expects HDF5 files with sleep recordings containing:
- EEG and EOG channels as spectrograms (29 x 129 time-frequency matrices)
- Hypnogram labels (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM)

## Performance

Tested on 12 training datasets + 8 hold-out datasets:

| Model | Cohen's Kappa | Performance |
|-------|---------------|-------------|
| SleepTransformer | 0.79 | Comparable to state-of-the-art |
| U-Sleep | 0.81 | Current state-of-the-art |

The model achieves strong performance across all sleep stages:
- **Wake detection**: 94% accuracy  
- **REM sleep**: 90.6% accuracy
- **Deep sleep (N2)**: 88.4% accuracy
- Robust performance with κ > 0.8 on multiple clinical datasets

## Files

- `SleepTransformerModel.py` - Core transformer architecture
- `SleepTransformerLightning.py` - Training and evaluation with PyTorch Lightning
- `config.py` - Model hyperparameters
- `modules.py` - Transformer building blocks (attention, feedforward, etc.)
- `preprocessing/` - Data preprocessing utilities

## Dependencies

```
torch>=1.9.0
pytorch-lightning>=1.5.0
numpy
scipy
h5py
pandas
scikit-learn
torchmetrics
```

## Research

This implementation is based on comparative research evaluating transformer architectures for sleep staging against established methods. The work demonstrates that transformers can achieve competitive performance while offering interpretability through attention mechanisms.
