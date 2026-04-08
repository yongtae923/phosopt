# D:\yongtae\phosopt\code\cnnopt_exp.py

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
CNNOPT_ROOT = PROJECT_ROOT / "data" / "cnnopt"


@dataclass(frozen=True)
class Variant:
    name: str
    stage_channels: tuple[int, int, int, int]
    num_res_blocks: int


VARIANTS = [
    Variant(
        name="baseline",
        stage_channels=(16, 32, 64, 128),
        num_res_blocks=4,
    ),
    Variant(
        name="shallow-0.5x",
        stage_channels=(16, 32, 64, 128),
        num_res_blocks=2,
    ),
    Variant(
        name="deep-2x",
        stage_channels=(16, 32, 64, 128),
        num_res_blocks=8,
    ),
    Variant(
        name="wide-2x",
        stage_channels=(32, 64, 128, 256),
        num_res_blocks=4,
    ),
]


def _run_command(command: list[str], env: dict[str, str], tag: str) -> None:
    print(f"\n[{tag}] Running: {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)


def _variant_env(base_env: dict[str, str], variant: Variant, variant_root: Path) -> dict[str, str]:
    train_dir = variant_root / "train"
    infer_dir = variant_root / "infer"
    analysis_dir = variant_root / "analysis"

    train_dir.mkdir(parents=True, exist_ok=True)
    infer_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    env = dict(base_env)
    env["PHOSOPT_NO_RESUME"] = "1"
    env["PHOSOPT_SAVE_DIR"] = str(train_dir)
    env["PHOSOPT_MODEL_PATH"] = str(train_dir / "single_subject_inverse_model.pt")
    env["PHOSOPT_INFER_RUN_DIR"] = str(infer_dir)
    env["PHOSOPT_ANALYSIS_INPUT_DIR"] = str(infer_dir)
    env["PHOSOPT_ANALYSIS_OUTPUT_DIR"] = str(analysis_dir)
    env["PHOSOPT_ENCODER_STAGE_CHANNELS"] = ",".join(str(c) for c in variant.stage_channels)
    env["PHOSOPT_ENCODER_RES_BLOCKS"] = str(variant.num_res_blocks)
    return env


def main() -> None:
    CNNOPT_ROOT.mkdir(parents=True, exist_ok=True)
    base_env = os.environ.copy()

    print("[cnnopt] Starting sequential CNN variant experiments", flush=True)
    print(f"[cnnopt] Output root: {CNNOPT_ROOT}", flush=True)

    for idx, variant in enumerate(VARIANTS, start=1):
        variant_root = CNNOPT_ROOT / variant.name
        env = _variant_env(base_env, variant, variant_root)

        print("\n" + "=" * 90, flush=True)
        print(f"[cnnopt] Variant {idx}/{len(VARIANTS)}: {variant.name}", flush=True)
        print(
            f"[cnnopt] Encoder stage_channels={variant.stage_channels}, "
            f"res_blocks={variant.num_res_blocks}",
            flush=True,
        )
        print(f"[cnnopt] Variant output: {variant_root}", flush=True)
        print("=" * 90, flush=True)

        _run_command([sys.executable, str(CODE_DIR / "train.py")], env=env, tag=f"{variant.name}:train")
        _run_command([sys.executable, str(CODE_DIR / "infer.py")], env=env, tag=f"{variant.name}:infer")
        _run_command(
            [sys.executable, str(CODE_DIR / "util" / "v3_infer_result_analysis.py")],
            env=env,
            tag=f"{variant.name}:analysis",
        )

    print("\n[cnnopt] All variants completed successfully.", flush=True)


if __name__ == "__main__":
    main()
