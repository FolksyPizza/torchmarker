from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List


DEFAULT_MODELS = ["sshleifer/tiny-gpt2"]
DEFAULT_MODES = ["eager", "compile"]

MATRIX_PRESETS: Dict[str, Dict[str, List[int]]] = {
    "quick": {
        "batch_sizes": [1, 4],
        "prompt_lengths": [128, 512],
        "gen_lengths": [32],
    },
    "balanced": {
        "batch_sizes": [1, 4, 16],
        "prompt_lengths": [128, 512, 2048],
        "gen_lengths": [32, 128],
    },
    "exhaustive": {
        "batch_sizes": [1, 2, 4, 8, 16, 32],
        "prompt_lengths": [64, 128, 512, 1024, 2048, 4096],
        "gen_lengths": [16, 32, 64, 128, 256],
    },
}


@dataclass
class BenchmarkConfig:
    models: List[str] = field(default_factory=lambda: DEFAULT_MODELS.copy())
    devices: List[str] = field(default_factory=lambda: ["auto"])
    modes: List[str] = field(default_factory=lambda: DEFAULT_MODES.copy())
    dtypes: List[str] = field(default_factory=lambda: ["auto"])
    matrix: str = "balanced"
    output_dir: Path = field(
        default_factory=lambda: Path("artifacts") / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    num_runs: int = 10
    warmup_runs: int = 3
    max_new_tokens: int = 128
    prompt_lengths: List[int] = field(default_factory=list)
    batch_sizes: List[int] = field(default_factory=list)
    gen_lengths: List[int] = field(default_factory=list)
    enable_int4: bool = True
    enable_kernel_fallback: bool = True
    seed: int = 1337

    def resolved_prompt_lengths(self) -> List[int]:
        if self.prompt_lengths:
            return self.prompt_lengths
        return MATRIX_PRESETS[self.matrix]["prompt_lengths"]

    def resolved_batch_sizes(self) -> List[int]:
        if self.batch_sizes:
            return self.batch_sizes
        return MATRIX_PRESETS[self.matrix]["batch_sizes"]

    def resolved_gen_lengths(self) -> List[int]:
        if self.gen_lengths:
            return self.gen_lengths
        return MATRIX_PRESETS[self.matrix]["gen_lengths"]


def parse_csv_arg(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def parse_csv_int_arg(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]
