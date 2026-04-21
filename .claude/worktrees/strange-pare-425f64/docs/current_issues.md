# Current Issues (2026-03-02)

## 1) GPU compatibility warning (RTX 5070)

- Observed warning:
  - `NVIDIA GeForce RTX 5070 with CUDA capability sm_120 is not compatible with the current PyTorch installation`
- Current environment:
  - `torch==2.3.1`
  - CUDA runtime present, but this PyTorch build does not include support for `sm_120`.
- Impact:
  - CUDA may be detected, but training kernels can fail or run incorrectly on this GPU generation.

## 2) Training stops due to non-differentiable simulator

- Observed error:
  - `RuntimeError: Simulator is non-differentiable. Use differentiable simulator or set allow_nondiff_training=True for debugging only.`
- Root cause:
  - `code/train.py` currently uses `NumpySimulatorAdapter` by default.
  - This adapter is intentionally marked non-differentiable and blocked in training mode.

## Temporary workaround (debug only)

Run on CPU and allow non-differentiable training path:

```powershell
$env:CUDA_VISIBLE_DEVICES=""
python code/train.py --maps-npz data/letters/emnist_letters_phosphenes.npz --retinotopy-dir data/fmri/100610 --hemisphere LH --subject-id letters100610 --max-train-samples 5000 --max-test-samples 1000 --epochs 20 --batch-size 8 --lr 1e-3 --save-dir data/output/inverse_training --allow-nondiff-training
```

## Proper fix plan

1. Upgrade/install a PyTorch build that supports RTX 5070 (`sm_120`).
2. Connect the real differentiable simulator to `SimulatorWrapper`.
3. Keep `NumpySimulatorAdapter` only for baseline/debug/inference.
