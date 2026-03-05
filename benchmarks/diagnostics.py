from __future__ import annotations

from typing import Any, Dict, List


def _system_values(payload: Dict[str, Any]) -> Dict[str, float]:
    ram_gbps = 0.0
    disk_read = 0.0
    disk_write = 0.0
    for row in payload.get("benchmarks", {}).get("system", {}).get("results", []):
        if row.get("metric") == "ram_copy_bandwidth_gbps":
            ram_gbps = float(row.get("value", 0.0))
        elif row.get("metric") == "disk_throughput_mbps":
            disk_read = float(row.get("read_mbps", 0.0))
            disk_write = float(row.get("write_mbps", 0.0))
    return {"ram_gbps": ram_gbps, "disk_read_mbps": disk_read, "disk_write_mbps": disk_write}


def _best_inference_tps(payload: Dict[str, Any]) -> float:
    rows: List[Dict[str, Any]] = []
    for run in payload.get("benchmarks", {}).get("model_inference", []):
        rows.extend(run.get("results", []))
    return max((float(r.get("total_tokens_per_sec", 0.0)) for r in rows), default=0.0)


def _max_gpu_mem(payload: Dict[str, Any]) -> float:
    max_mem = 0.0
    for _, info in payload.get("environment", {}).get("devices", {}).items():
        if info.get("type") == "cuda":
            max_mem = max(max_mem, float(info.get("total_memory_gb", 0.0)))
    return max_mem


def analyze_bottlenecks(payload: Dict[str, Any]) -> Dict[str, Any]:
    bottlenecks: List[Dict[str, Any]] = []
    upgrades: List[str] = []

    sysv = _system_values(payload)
    best_tps = _best_inference_tps(payload)
    max_gpu_mem = _max_gpu_mem(payload)

    devices = payload.get("environment", {}).get("devices", {})
    has_gpu = any(v.get("type") == "cuda" for v in devices.values())

    if not has_gpu:
        bottlenecks.append(
            {
                "area": "compute",
                "severity": "high",
                "issue": "No CUDA GPU detected; heavy inference/training is CPU-limited.",
            }
        )
        upgrades.append("Add a CUDA-capable GPU with >=16 GB VRAM for stronger inference/training performance.")

    if max_gpu_mem > 0 and max_gpu_mem < 12:
        bottlenecks.append(
            {
                "area": "vram",
                "severity": "medium",
                "issue": f"Max GPU VRAM is {max_gpu_mem:.2f} GB; larger models and higher batch sizes will be constrained.",
            }
        )
        upgrades.append("Upgrade to a GPU with more VRAM (24 GB+ preferred for larger model work).")

    if sysv["ram_gbps"] < 20:
        bottlenecks.append(
            {
                "area": "memory",
                "severity": "medium",
                "issue": f"RAM bandwidth appears low ({sysv['ram_gbps']:.2f} GB/s).",
            }
        )
        upgrades.append("Use faster memory (higher MT/s, dual/quad channel) to improve data movement performance.")

    if sysv["disk_read_mbps"] < 500 or sysv["disk_write_mbps"] < 400:
        bottlenecks.append(
            {
                "area": "storage",
                "severity": "medium",
                "issue": (
                    f"Disk throughput may bottleneck data/model IO (read {sysv['disk_read_mbps']:.1f} MB/s, "
                    f"write {sysv['disk_write_mbps']:.1f} MB/s)."
                ),
            }
        )
        upgrades.append("Use an NVMe SSD for dataset/model cache and artifacts.")

    if best_tps < 100:
        bottlenecks.append(
            {
                "area": "inference",
                "severity": "low",
                "issue": f"Observed best total throughput is modest ({best_tps:.2f} tok/s on current workload).",
            }
        )

    temp = payload.get("temperature", {})
    cpu_max = temp.get("cpu_max_temp_c")
    gpu_max = temp.get("gpu_max_temp_c", {})

    thermal_alerts: List[str] = []
    if isinstance(cpu_max, (int, float)) and cpu_max >= 90:
        thermal_alerts.append(f"CPU peak temperature is high ({cpu_max:.1f} C).")
        upgrades.append("Improve CPU cooling/airflow or reduce sustained power limits.")

    for dev, v in gpu_max.items():
        if isinstance(v, (int, float)) and v >= 85:
            thermal_alerts.append(f"{dev} peak temperature is high ({v:.1f} C).")
            upgrades.append("Improve GPU cooling/airflow and check fan curve or power target.")

    throttling_events: List[Dict[str, Any]] = []
    for row in payload.get("benchmarks", {}).get("stress", {}).get("results", []):
        first_ips = row.get("first_window_iters_per_sec")
        last_ips = row.get("last_window_iters_per_sec")
        if isinstance(first_ips, (int, float)) and isinstance(last_ips, (int, float)) and first_ips > 0:
            drop = (first_ips - last_ips) / first_ips
            if drop >= 0.2:
                throttling_events.append(
                    {
                        "device": row.get("device"),
                        "dtype": row.get("dtype"),
                        "first_window_iters_per_sec": round(float(first_ips), 4),
                        "last_window_iters_per_sec": round(float(last_ips), 4),
                        "drop_ratio": round(float(drop), 4),
                    }
                )

    throttling_suspected = len(throttling_events) > 0
    if throttling_suspected:
        bottlenecks.append(
            {
                "area": "thermal/power",
                "severity": "high",
                "issue": "Sustained stress throughput dropped significantly, suggesting throttling.",
            }
        )
        upgrades.append("Check cooling, power limits, and clocks to reduce sustained-performance throttling.")

    # Deduplicate while preserving order.
    dedup_upgrades = list(dict.fromkeys(upgrades))

    return {
        "bottlenecks": bottlenecks,
        "upgrade_suggestions": dedup_upgrades,
        "thermal_alerts": thermal_alerts,
        "throttling": {
            "suspected": throttling_suspected,
            "events": throttling_events,
        },
    }
