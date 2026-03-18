"""
Standardised experiment logger for PhosOpt.

All methods use the same loss formula and log schema:
  - score S = DC + 0.1*Y - HD  (composite score; loss = 2 - S)
  - dc = Dice coefficient, y = yield, hd = Hellinger distance.

Writes JSON-Lines (one JSON object per line) for easy append and pandas load.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentLog:
    """Single experiment result record (matches exp.md Section 8)."""

    experiment_id: str = ""
    benchmark_type: str = ""        # per_target | generalized | adaptation
    method: str = ""                # bayesian | all_on | random | heuristic_center | ...
    seed: int = 0
    target_id: str = ""

    # --- results (same schema for all methods) ---
    score: float = 0.0   # S = DC + 0.1*Y - HD
    loss: float = 0.0    # Loss = 2 - S
    dc: float = 0.0      # Dice coefficient
    y: float = 0.0       # yield (fraction of contacts in valid cortex)
    hd: float = 0.0      # Hellinger distance
    active_electrode_count: int = 0
    simulator_calls: int = 0
    wall_clock_time: float = 0.0

    # --- cost breakdown ---
    training_cost: float = 0.0      # CNN training wall-clock (0 for per-target)
    solving_cost: float = 0.0       # target reconstruction wall-clock

    # --- optional extras (params, etc.) ---
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


class ExperimentLogger:
    """Append-only JSON-Lines logger backed by a single file."""

    def __init__(self, log_path: str | Path) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: ExperimentLog) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def log_many(self, records: list[ExperimentLog]) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        with open(self.log_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


class SimulatorCallCounter:
    """Transparent wrapper that counts how many times a simulator is called."""

    def __init__(self, simulator: Any) -> None:
        self._sim = simulator
        self.call_count: int = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        return self._sim(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        return self._sim.forward(*args, **kwargs)

    def reset(self) -> None:
        self.call_count = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._sim, name)


class Timer:
    """Simple wall-clock context manager."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self._start
