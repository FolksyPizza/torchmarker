from __future__ import annotations

import time
from typing import Dict, List

import torch

from .discovery import DTYPE_TO_TORCH


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def benchmark_kernel_suite(
    devices: List[str],
    dtypes: List[str],
    warmup_runs: int,
    num_runs: int,
) -> Dict:
    results = []
    skips = []

    shapes = [(1024, 1024, 1024), (2048, 2048, 2048)]

    for device in devices:
        for dtype_key in dtypes:
            if dtype_key == "int4":
                skips.append(
                    {
                        "suite": "kernel_microbench",
                        "device": device,
                        "dtype": dtype_key,
                        "reason": "native torch int4 tensor dtype unavailable",
                    }
                )
                continue

            torch_dtype = DTYPE_TO_TORCH.get(dtype_key)
            if torch_dtype is None:
                skips.append(
                    {
                        "suite": "kernel_microbench",
                        "device": device,
                        "dtype": dtype_key,
                        "reason": "unknown dtype",
                    }
                )
                continue

            for m, n, k in shapes:
                try:
                    a = torch.randn((m, k), device=device).to(torch_dtype)
                    b = torch.randn((k, n), device=device).to(torch_dtype)
                except Exception as exc:
                    skips.append(
                        {
                            "suite": "kernel_microbench",
                            "device": device,
                            "dtype": dtype_key,
                            "shape": f"{m}x{n}x{k}",
                            "reason": str(exc),
                        }
                    )
                    continue

                for _ in range(warmup_runs):
                    _ = a @ b
                    _sync(device)

                t0 = time.perf_counter()
                for _ in range(num_runs):
                    _ = a @ b
                _sync(device)
                elapsed = time.perf_counter() - t0

                ops = 2 * m * n * k * num_runs
                tflops = (ops / elapsed) / 1e12 if elapsed > 0 else 0.0
                iters_per_sec = num_runs / elapsed if elapsed > 0 else 0.0

                bytes_moved = (a.numel() * a.element_size() + b.numel() * b.element_size()) * num_runs
                gbps = (bytes_moved / elapsed) / 1e9 if elapsed > 0 else 0.0

                results.append(
                    {
                        "suite": "kernel_microbench",
                        "op": "matmul",
                        "device": device,
                        "dtype": dtype_key,
                        "shape": f"{m}x{n}x{k}",
                        "num_runs": num_runs,
                        "iters_per_sec": round(iters_per_sec, 4),
                        "tflops_est": round(tflops, 4),
                        "gbps_est": round(gbps, 4),
                    }
                )

    return {"suite": "kernel_microbench", "results": results, "skips": skips}
