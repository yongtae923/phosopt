# PhosOpt Code Structure and Design Guide

## 1. Project Overview

This project trains a physics-guided inverse model for cortical visual prosthesis optimization.

Core question:
"If an implant is inserted only once, what implant placement should be used so that many target phosphene images can be reproduced with per-image electrode stimulation?"

### Main outputs

- **4 shared implant parameters** (global, same for all images)
  - `alpha`: rotation angle (-90 to 90 deg)
  - `beta`: tilt angle (-15 to 110 deg)
  - `offset_from_base`: offset from base (0 to 40 mm)
  - `shank_length`: shank length (10 to 40 mm)
- **1000 electrode activations** (predicted per image)

---

## 2. Directory Structure

```text
phosopt/
├── environment.yml
├── README.md
│
├── basecode/                        # Reference legacy code
│   ├── model.py
│   ├── ninimplant.py
│   └── ...
│
├── code/                            # Active training/inference pipeline
│   ├── train.py                     # Entry point
│   ├── trainer.py                   # Train/eval loops
│   ├── dataset.py                   # Data loading + splitting
│   ├── models/
│   │   ├── encoder.py               # E2E-style encoder (adapted to 256x256)
│   │   ├── parameter_head.py        # ElectrodeHead, ContinuousHead, bounds
│   │   ├── inverse_model.py         # Shared-implant inverse model
│   │   └── __init__.py
│   ├── simulator/
│   │   ├── physics_forward.py       # Numpy/physics forward implementation
│   │   ├── physics_forward_torch.py # Differentiable torch simulator
│   │   ├── simulator_wrapper.py     # Differentiable/non-diff adapters
│   │   └── __init__.py
│   ├── loss/
│   │   ├── losses.py                # Loss terms and metrics
│   │   └── __init__.py
│   ├── diagnose.py
│   ├── diag_sim_upper_bound.py
│   └── analyze_letters_structure.py
│
└── docs/
    ├── design_shared_params_ko.md
    ├── design_shared_params_en.md   # This document
    └── current_issues.md
```

---

## 3. End-to-End Data Flow

1. `dataset.py` loads target phosphene maps from `--maps-npz` or `--maps-dir`.
2. `train.py` creates train/val/test splits and `DataLoader`s.
3. `InverseModel` predicts:
   - shared implant parameters `[B, 4]` (identical rows),
   - per-image electrode logits `[B, 1000]`.
4. `DifferentiableSimulator` (or `NumpySimulatorAdapter`) reconstructs phosphene maps.
5. `losses.py` computes reconstruction + regularization losses.
6. `trainer.py` backpropagates and updates:
   - network weights (encoder + electrode head),
   - shared implant parameter vector.
7. `train.py` saves:
   - model checkpoint (`*.pt`),
   - report JSON with metrics + learned shared implant params.

---

## 4. Core Model Design

### 4.1 Previous (per-image 4 params)

```text
image -> Encoder -> latent -> ContinuousHead -> 4 params (per-image)
                           -> ElectrodeHead  -> 1000 logits (per-image)
```

### 4.2 Current (shared 4 params + per-image electrodes)

```text
                         +--------------------------------------+
                         |              InverseModel            |
                         |                                      |
                         | _shared_params_raw (nn.Parameter[4]) |
                         |          -> sigmoid + bounds         |
                         |          -> shared_params [4]        |
                         |          -> expand -> [B,4]          |
image [B,1,256,256] ---> | Encoder -> latent [B,256]           |
                         |            -> ElectrodeHead          |
                         |            -> electrode_logits[B,1000]|
                         +------------------+-------------------+
                                            |
                                            v
                                  Simulator(params, logits)
                                            |
                                            v
                                  reconstructed map [B,1,H,W]
```

### 4.3 Why this is correct for your use case

- The implant is physically fixed after surgery -> the 4 placement parameters must be global/shared.
- Different target images still require different stimulation patterns -> keep electrode prediction per image.

---

## 5. File-by-File Responsibilities

### `code/train.py`

- Parses CLI arguments.
- Selects CUDA/CPU runtime.
- Loads data and creates splits/loaders.
- Builds model/simulator/loss/train configs.
- Runs training + evaluation.
- Saves checkpoint and report.
- New argument: `--shared-params-lr`.
- Report now includes `learned_implant_params`.

### `code/trainer.py`

- Defines `TrainConfig` and train/eval routines.
- Uses two optimizer parameter groups:
  - network params -> `lr`
  - shared implant raw params -> `shared_params_lr`
- Keeps refinement image-specific by refining only `electrode_logits`.
- Logs current shared parameters during diagnostics and per epoch.

### `code/models/inverse_model.py`

- Implements shared-implant architecture.
- Stores `_shared_params_raw = nn.Parameter(torch.zeros(4))`.
- Converts raw params to bounded physical ranges with sigmoid scaling.
- Predicts per-image electrode logits from encoder features.

### `code/models/encoder.py`

- E2E-style CNN adapted to 256x256 phosphene maps.
- LeakyReLU + residual blocks + flatten projection to latent vector.

### `code/models/parameter_head.py`

- `ElectrodeHead`: predicts 1000 electrode logits.
- `ParameterBounds`: defines valid physical ranges.
- `ContinuousHead` and `ParameterHead` remain for compatibility, but current `InverseModel` uses `ElectrodeHead` directly.

### `code/loss/losses.py`

- Reconstruction MSE, soft Dice, sparsity, invalid-region penalty, optional param prior.
- With shared params, batch-wise param-prior variance is naturally near zero.

### `code/simulator/*`

- `physics_forward_torch.py`: differentiable simulator for gradient training.
- `simulator_wrapper.py`: converts logits to probabilities and bridges simulator interface.
- `physics_forward.py`: numpy-based forward path.

### `code/dataset.py`

- Normalizes maps to [0,1].
- Supports `.npy` directory or `.npz` keys.
- Deterministic splits with seed control.

---

## 6. Training Configuration

Important CLI options:

- `--maps-npz` or `--maps-dir`: input maps source
- `--retinotopy-dir`: required retinotopy data
- `--hemisphere`: `LH` or `RH`
- `--lr`: learning rate for network weights
- `--shared-params-lr`: learning rate for shared implant params
- `--epochs`, `--batch-size`
- `--valid-electrode-mask`: optional 1000-dim mask

Example:

```bash
python code/train.py \
  --maps-npz data/letters/emnist_letters_phosphenes.npz \
  --retinotopy-dir data/fmri/100610 \
  --hemisphere LH \
  --subject-id s100610 \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-3 \
  --shared-params-lr 1e-2 \
  --save-dir data/output/inverse_training
```

---

## 7. Output Artifacts

- `..._inverse_model.pt`
  - includes encoder/electrode-head weights and shared implant raw params.
- `..._report.json`
  - `history`
  - `test_metrics`
  - `baselines`
  - `config`
  - `learned_implant_params` (alpha, beta, offset_from_base, shank_length)

---

## 8. Design Rationale Summary

1. **Physical realism**: one implant placement shared across all images.
2. **Task decomposition**: global placement + per-image stimulation.
3. **Stable optimization**: separate learning rates for global 4 params vs large network.
4. **Compatibility**: simulator/trainer interfaces stay mostly unchanged.
5. **Interpretability**: final report directly exposes globally optimized implant parameters.

---

## 9. Recommended Next Step

If you want to deploy this model for inference:

1. Load checkpoint.
2. Read `model.shared_params` once as the global implant setup.
3. For each new image, run forward pass and convert logits to probabilities (`sigmoid`) or binary on/off (`threshold`).
