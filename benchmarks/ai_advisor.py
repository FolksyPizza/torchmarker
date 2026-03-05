from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, Dict


def _prompt_from_payload(payload: Dict[str, Any], diagnostics: Dict[str, Any]) -> str:
    summary = {
        "suitability": payload.get("suitability", {}),
        "temperature": {
            "cpu_max_temp_c": payload.get("temperature", {}).get("cpu_max_temp_c"),
            "gpu_max_temp_c": payload.get("temperature", {}).get("gpu_max_temp_c", {}),
            "sample_count": payload.get("temperature", {}).get("sample_count", 0),
        },
        "diagnostics": diagnostics,
        "system": payload.get("benchmarks", {}).get("system", {}).get("results", []),
    }
    return (
        "You are a performance engineer. Based on the benchmark summary below, provide concise upgrade and tuning "
        "recommendations for inference, training, and local Torch development. Mention likely bottlenecks, thermal "
        "risk, and next 5 actions prioritized by impact.\n\n"
        f"DATA:\n{json.dumps(summary, indent=2)}"
    )


def generate_ai_diagnosis(
    payload: Dict[str, Any],
    diagnostics: Dict[str, Any],
    model: str = "qwen3.5:2b",
    timeout_sec: int = 90,
) -> Dict[str, Any]:
    if shutil.which("ollama") is None:
        return {
            "enabled": True,
            "model": model,
            "status": "skipped",
            "reason": "ollama_not_found",
            "output": None,
        }

    prompt = _prompt_from_payload(payload, diagnostics)
    try:
        proc = subprocess.run(
            ["ollama", "run", model, prompt],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        if proc.returncode != 0:
            return {
                "enabled": True,
                "model": model,
                "status": "error",
                "reason": proc.stderr.strip() or f"exit_code_{proc.returncode}",
                "output": None,
            }
        return {
            "enabled": True,
            "model": model,
            "status": "ok",
            "reason": None,
            "output": proc.stdout.strip(),
        }
    except subprocess.TimeoutExpired:
        return {
            "enabled": True,
            "model": model,
            "status": "timeout",
            "reason": f"timeout_after_{timeout_sec}s",
            "output": None,
        }
