# PhosOpt Inverse Training

Physics-guided inverse estimator for phosphene map reconstruction.

## Quick Start

```bash
conda env remove -n phosopt -y
conda env create -f environment.yml
conda activate phosopt
python -c "import torch, nibabel, trimesh; print('imports_ok'); print('cuda=', torch.cuda.is_available())"
python code/train.py \
  --maps-npz data/letters/emnist_letters_phosphenes.npz \
  --retinotopy-dir data/fmri/100610 \
  --hemisphere LH \
  --subject-id s100610 \
  --max-train-samples 5000 \
  --max-test-samples 1000 \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-3 \
  --save-dir data/output/inverse_training
```

Expected output files:

- `data/output/inverse_training/s100610_inverse_model.pt`
- `data/output/inverse_training/s100610_report.json`

## What this project does

Given a target phosphene map, the model predicts:

- 4 implant parameters: `alpha`, `beta`, `offset_from_base`, `shank_length`
- 1000 electrode activations (continuous during training)

Then the simulator reconstructs a phosphene map, and training minimizes reconstruction + regularization losses.

## Architecture

- `code/dataset.py`: target map dataset and train/val/test split
- `code/models/encoder.py`: small CNN encoder
- `code/models/parameter_head.py`: multi-head output (`4 params + 1000 logits`)
- `code/models/inverse_model.py`: full inverse estimator
- `code/simulator/physics_forward.py`: forward physics simulator module
- `code/simulator/simulator_wrapper.py`: differentiable wrapper + numpy adapter
- `code/loss/losses.py`: loss and metrics
- `code/trainer.py`: training/evaluation loops with `tqdm`
- `code/train.py`: entrypoint script

## Inputs / Outputs

### Inputs

1. Target maps source (required, one of):
   - path: `--maps-dir`
   - format: `.npy` files, each `[H, W]`
   - or path: `--maps-npz`
   - format: `.npz` with phosphene keys (default: `train_phosphenes`, `test_phosphenes`)

2. Retinotopy directory (required):
   - path: `--retinotopy-dir`
   - required files:
     - `inferred_angle.mgz`
     - `inferred_eccen.mgz`
     - `inferred_sigma.mgz`
     - `aparc+aseg.mgz`
     - `inferred_varea.mgz`

3. Optional valid electrode mask:
   - path: `--valid-electrode-mask`
   - format: `.npy`, shape `[1000]`, values in `[0,1]`

### Outputs

- model checkpoint: `<save-dir>/<subject-id>_inverse_model.pt`
- report json: `<save-dir>/<subject-id>_report.json`
  - contains train history, test metrics, baselines, and configs

## Device and performance behavior

- GPU: uses CUDA automatically when available (`torch.cuda.is_available()`).
- CPU fallback:
  - DataLoader workers = `max(1, int(cpu_count * 0.8))`
  - `torch.set_num_threads(max(1, int(cpu_count * 0.8)))`
- Progress bars:
  - training and evaluation loops use `tqdm` in terminal.

## Environment setup (Conda)

### 1) Create environment

```bash
conda env create -f environment.yml
conda activate phosopt
```

This project no longer requires `mathutils`.

### 2) Verify CUDA in PyTorch

```bash
python -c "import torch; print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"
```

### 3) If CUDA is not available

- Check NVIDIA driver installation.
- Run `nvidia-smi`.
- Ensure your installed CUDA runtime matches your GPU driver.

## Run training

```bash
python code/train.py \
  --maps-npz data/letters/emnist_letters_phosphenes.npz \
  --retinotopy-dir data/fmri/100610 \
  --hemisphere LH \
  --subject-id s100610 \
  --max-train-samples 5000 \
  --max-test-samples 1000 \
  --epochs 50 \
  --batch-size 8 \
  --lr 1e-3 \
  --save-dir data/output/inverse_training
```

## Data flow summary

1. `PhospheneDataset` loads target maps and normalizes each map.
2. `InverseModel` predicts bounded parameters and electrode logits.
3. `SimulatorWrapper` applies `sigmoid` to logits and runs simulator.
4. `build_losses` computes:
   - reconstruction MSE
   - sparsity penalty
   - parameter prior
   - invalid-region penalty
5. `trainer` updates model, logs metrics, and evaluates baselines.

## Notes

- The current `NumpySimulatorAdapter` is non-differentiable and intended for baseline/inference/debug path.
- For full gradient-based inverse learning, plug your differentiable simulator into `SimulatorWrapper`.
