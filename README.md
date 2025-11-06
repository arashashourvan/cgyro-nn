# cgyro-nn
Physics informed neural-net modeling of gyrokinetic turbulence based on  CGYRO 

# Phi2Flux: Deep Learning Surrogate for Gyrokinetic Flux Prediction

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](link-when-ready)

## Overview

Phi2Flux is a deep learning surrogate model that predicts ion/electron heat and particle 
fluxes from gyrokinetic turbulence potential fields. By combining 3D convolutional encoders 
with temporal causal networks (TCN), we achieve fast, accurate flux predictions from 
δf-CGYRO simulations.

**Key Features:**
- ⚡ XXX× speedup over traditional CGYRO post-processing
- 📊 RMSE ~0.5 across all flux channels (Qi, Qe, Γ)
- 🔄 Causal temporal modeling via TCN
- 🎯 Multi-output regression for simultaneous flux prediction
- 🚀 Trained in 30 minutes on single A100 GPU

## Results

![Training Curves](results/training_curves.png)
![Predictions](results/predictions_vs_actual.png)

| Flux Channel | RMSE | R² |
|--------------|------|-----|
| Qi (ion heat) | 0.50 | X.XX |
| Qe (electron heat) | 0.52 | X.XX |
| Γ (particle) | 0.49 | X.XX |

## Architecture

- **Spatial Encoder:** 3D CNN (3 layers) on φ(kx, ky, θ) with real/imaginary channels
- **Temporal Model:** TCN with dilations [1,2,4,8] for causal φ(t) evolution
- **Output Heads:** 3 linear regressors for Qi, Qe, Γ

See [docs/architecture.md](docs/architecture.md) for details.

## Installation
```bash
conda env create -f environment.yml
conda activate phi2flux
```

## Quick Start
```python
from model import Phi2Flux
from data_loader import load_cgyro_data

# Load model
model = Phi2Flux.load_pretrained('checkpoints/best_model.pt')

# Predict fluxes
phi_field = load_cgyro_data('path/to/data')
qi, qe, gamma = model.predict(phi_field)
```

## Training
```bash
python train.py --config configs/default.yaml --gpu 0
```

## Citation
```bibtex
@article{ashourvan2024phi2flux,
  title={Phi2Flux: Deep Learning Surrogate for Gyrokinetic Flux Prediction},
  author={Ashourvan, Arash},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## Applications Beyond Fusion

This architecture is applicable to any spatiotemporal physics problem:
- Computational fluid dynamics
- Climate modeling
- Plasma physics
- Materials science

## Future Work

- Self-consistent φ(t) → φ(t+Δt) autoregressive model
- Multi-GPU distributed training
- Extension to multi-scale transport (core + pedestal + SOL)

## Contact

Arash Ashourvan - [LinkedIn](your-profile) - [Email]

## Acknowledgments

Simulations performed on Perlmutter at NERSC.
