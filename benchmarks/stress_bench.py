from __future__ import annotations

import time
from typing import Dict, List, Tuple

import torch

from .discovery import DTYPE_TO_TORCH


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def _shape_candidates(device: str) -> List[int]:
    if device.startswith("cuda"):
        return [8192, 6144, 4096, 3072, 2048]
    return [4096, 3072, 2048, 1024]


def _pick_shape(device: str, torch_dtype: torch.dtype) -> Tuple[int, torch.Tensor, torch.Tensor]:
    last_exc: Exception | None = None
    for n in _shape_candidates(device):
        try:
            a = torch.randn((n, n), device=device, dtype=torch_dtype)
            b = torch.randn((n, n), device=device, dtype=torch_dtype)
            _ = a @ b
            _sync(device)
            return n, a, b
        except Exception as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("no candidate shape")


def benchmark_stress_suite(
    devices: List[str],
    dtypes: List[str],
    duration_sec: int,
) -> Dict:
    results = []
    skips = []

    for device in devices:
        stress_dtypes = [d for d in dtypes if d in {"fp32", "fp16", "bf16"}]
        if not stress_dtypes:
            stress_dtypes = [d for d in dtypes if d != "int4"]

        for dtype_key in stress_dtypes:
            torch_dtype = DTYPE_TO_TORCH.get(dtype_key)
            if torch_dtype is None:
                skips.append(
                    {
                        "suite": "stress",
                        "device": device,
                        "dtype": dtype_key,
                        "reason": "unknown dtype",
                    }
                )
                continue

            try:
                n, a, b = _pick_shape(device, torch_dtype)
            except Exception as exc:
                skips.append(
                    {
                        "suite": "stress",
                        "device": device,
                        "dtype": dtype_key,
                        "reason": f"allocation/setup failed: {exc}",
                    }
                )
                continue

            iters = 0
            t0 = time.perf_counter()
            deadline = t0 + duration_sec

            while time.perf_counter() < deadline:
                _ = a @ b
                _sync(device)
                iters += 1

            elapsed = time.perf_counter() - t0
            ops = 2 * (n**3) * max(iters, 1)
            tflops = (ops / elapsed) / 1e12 if elapsed > 0 else 0.0
            iters_per_sec = iters / elapsed if elapsed > 0 else 0.0

            results.append(
                {
                    "suite": "stress",
                    "device": device,
                    "dtype": dtype_key,
                    "shape": f"{n}x{n}",
                    "duration_sec": round(elapsed, 3),
                    "iterations": iters,
                    "iters_per_sec": round(iters_per_sec, 4),
                    "tflops_est": round(tflops, 4),
                }
            )

    return {"suite": "stress", "results": results, "skips": skips}
