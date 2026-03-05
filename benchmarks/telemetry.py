from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class TemperatureMonitor:
    def __init__(self, poll_interval_sec: float = 1.0) -> None:
        self.poll_interval_sec = poll_interval_sec
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[Dict[str, Any]] = []
        self._gpu_ready = False
        self._psutil = None
        self._pynvml = None

        try:
            import psutil  # type: ignore

            self._psutil = psutil
        except Exception:
            self._psutil = None

        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._gpu_ready = True
        except Exception:
            self._pynvml = None
            self._gpu_ready = False

    def close(self) -> None:
        if self._gpu_ready and self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

    def _read_cpu_temp_c(self) -> Optional[float]:
        if self._psutil is None:
            return None
        try:
            sensors = self._psutil.sensors_temperatures()
        except Exception:
            return None
        values: List[float] = []
        for entries in sensors.values():
            for entry in entries:
                current = getattr(entry, "current", None)
                if isinstance(current, (int, float)):
                    values.append(float(current))
        if not values:
            return None
        return max(values)

    def _read_gpu_temps_c(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not self._gpu_ready or self._pynvml is None:
            return out
        try:
            count = self._pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = self._pynvml.nvmlDeviceGetHandleByIndex(i)
                temp = self._pynvml.nvmlDeviceGetTemperature(
                    handle, self._pynvml.NVML_TEMPERATURE_GPU
                )
                out[f"cuda:{i}"] = float(temp)
        except Exception:
            return {}
        return out

    def sample_once(self) -> None:
        self.samples.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "cpu_temp_c": self._read_cpu_temp_c(),
                "gpu_temps_c": self._read_gpu_temps_c(),
            }
        )

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.poll_interval_sec)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="temp-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self) -> Dict[str, Any]:
        cpu_values = [s["cpu_temp_c"] for s in self.samples if isinstance(s.get("cpu_temp_c"), (int, float))]
        gpu_max: Dict[str, float] = {}
        for sample in self.samples:
            for dev, temp in sample.get("gpu_temps_c", {}).items():
                if dev not in gpu_max or temp > gpu_max[dev]:
                    gpu_max[dev] = temp
        return {
            "enabled": True,
            "sample_count": len(self.samples),
            "poll_interval_sec": self.poll_interval_sec,
            "sources": {
                "psutil": self._psutil is not None,
                "pynvml": self._gpu_ready,
            },
            "cpu_max_temp_c": max(cpu_values) if cpu_values else None,
            "gpu_max_temp_c": gpu_max,
            "samples": self.samples,
        }
