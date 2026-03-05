from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict

import numpy as np


def benchmark_ram_speed(size_mb: int = 512, iters: int = 5) -> Dict:
    n_float32 = (size_mb * 1024 * 1024) // 4
    src = np.random.rand(n_float32).astype(np.float32)

    _ = src.copy()
    start = time.perf_counter()
    for _ in range(iters):
        dst = src.copy()
    elapsed = time.perf_counter() - start
    _ = float(dst[0])

    total_bytes = src.nbytes * iters
    gbps = (total_bytes / elapsed) / 1e9 if elapsed > 0 else 0.0

    return {
        "suite": "system",
        "metric": "ram_copy_bandwidth_gbps",
        "size_mb": size_mb,
        "iters": iters,
        "value": round(gbps, 4),
    }


def benchmark_disk_speed(output_dir: Path, size_mb: int = 512, block_mb: int = 8) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "disk_bench.bin"

    total_bytes = size_mb * 1024 * 1024
    block = os.urandom(block_mb * 1024 * 1024)
    blocks = max(1, total_bytes // len(block))

    t0 = time.perf_counter()
    with path.open("wb") as f:
        for _ in range(blocks):
            f.write(block)
        f.flush()
        os.fsync(f.fileno())
    write_elapsed = time.perf_counter() - t0

    t1 = time.perf_counter()
    with path.open("rb") as f:
        while f.read(len(block)):
            pass
    read_elapsed = time.perf_counter() - t1

    written_bytes = blocks * len(block)
    write_mbps = (written_bytes / write_elapsed) / 1e6 if write_elapsed > 0 else 0.0
    read_mbps = (written_bytes / read_elapsed) / 1e6 if read_elapsed > 0 else 0.0

    try:
        path.unlink()
    except Exception:
        pass

    return {
        "suite": "system",
        "metric": "disk_throughput_mbps",
        "size_mb": size_mb,
        "block_mb": block_mb,
        "write_mbps": round(write_mbps, 4),
        "read_mbps": round(read_mbps, 4),
    }


def benchmark_system_suite(output_dir: Path, ram_size_mb: int, disk_size_mb: int) -> Dict:
    ram = benchmark_ram_speed(size_mb=ram_size_mb)
    disk = benchmark_disk_speed(output_dir=output_dir, size_mb=disk_size_mb)
    return {"suite": "system", "results": [ram, disk], "skips": []}
