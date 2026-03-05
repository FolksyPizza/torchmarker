from __future__ import annotations

from typing import Any, Dict, List


def _tier(score: float) -> str:
    if score >= 85:
        return "excellent"
    if score >= 70:
        return "good"
    if score >= 50:
        return "moderate"
    return "limited"


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def build_suitability_scores(payload: Dict[str, Any]) -> Dict[str, Any]:
    model_runs = payload.get("benchmarks", {}).get("model_inference", [])
    stress_rows = payload.get("benchmarks", {}).get("stress", {}).get("results", [])
    system_rows = payload.get("benchmarks", {}).get("system", {}).get("results", [])
    env_devices = payload.get("environment", {}).get("devices", {})

    model_rows: List[Dict[str, Any]] = []
    for run in model_runs:
        model_rows.extend(run.get("results", []))

    best_total_tps = max((float(r.get("total_tokens_per_sec", 0.0)) for r in model_rows), default=0.0)
    best_tflops = max((float(r.get("tflops_est", 0.0)) for r in stress_rows), default=0.0)

    max_gpu_mem = 0.0
    has_gpu = False
    for _, info in env_devices.items():
        if info.get("type") == "cuda":
            has_gpu = True
            max_gpu_mem = max(max_gpu_mem, float(info.get("total_memory_gb", 0.0)))

    ram_gbps = 0.0
    disk_read = 0.0
    disk_write = 0.0
    for row in system_rows:
        if row.get("metric") == "ram_copy_bandwidth_gbps":
            ram_gbps = float(row.get("value", 0.0))
        if row.get("metric") == "disk_throughput_mbps":
            disk_read = float(row.get("read_mbps", 0.0))
            disk_write = float(row.get("write_mbps", 0.0))

    inference = 0.0
    inference += 25.0 if has_gpu else 5.0
    inference += min(55.0, best_total_tps / 40.0)
    inference += min(20.0, max_gpu_mem * 1.5)
    inference = _clamp(inference)

    training = 0.0
    training += 20.0 if has_gpu else 0.0
    training += min(50.0, best_tflops * 6.0)
    training += min(20.0, max_gpu_mem * 1.5)
    training += min(10.0, ram_gbps / 5.0)
    training = _clamp(training)

    dev = 0.0
    dev += min(40.0, ram_gbps * 1.5)
    dev += min(30.0, disk_read / 100.0)
    dev += min(30.0, disk_write / 100.0)
    dev = _clamp(dev)

    return {
        "inference": {"score": round(inference, 2), "tier": _tier(inference)},
        "training": {"score": round(training, 2), "tier": _tier(training)},
        "torch_playground_dev": {"score": round(dev, 2), "tier": _tier(dev)},
        "inputs": {
            "best_total_tokens_per_sec": round(best_total_tps, 2),
            "best_stress_tflops": round(best_tflops, 4),
            "max_gpu_memory_gb": round(max_gpu_mem, 2),
            "ram_gbps": round(ram_gbps, 4),
            "disk_read_mbps": round(disk_read, 2),
            "disk_write_mbps": round(disk_write, 2),
        },
    }
