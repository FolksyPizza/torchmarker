from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("benchmarks")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def write_supported_types_log(path: Path, payload: Dict[str, Any]) -> None:
    lines = ["# Supported Type Matrix", ""]
    for dev, info in payload.get("devices", {}).items():
        lines.append(f"[{dev}]")
        lines.append("supported_tensor_dtypes=" + ", ".join(info.get("supported_tensor_dtypes", [])))
        if info.get("unsupported_tensor_dtypes"):
            lines.append(
                "unsupported_tensor_dtypes=" + ", ".join(info.get("unsupported_tensor_dtypes", []))
            )
        lines.append("")

    lines.append("# Model DType Support")
    for model, model_info in payload.get("models", {}).items():
        lines.append(f"[{model}]")
        for key, val in model_info.items():
            lines.append(f"{key}={json.dumps(val, sort_keys=True)}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
