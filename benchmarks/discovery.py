from __future__ import annotations

import platform
from typing import Dict, List

import torch


TENSOR_DTYPE_CANDIDATES = [
    "fp64",
    "fp32",
    "fp16",
    "bf16",
    "int8",
    "int16",
    "int32",
]

MODEL_DTYPE_CANDIDATES = ["fp32", "fp16", "bf16", "int8", "int4", "int16"]

DTYPE_TO_TORCH = {
    "fp64": torch.float64,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
}


def resolve_devices(raw_devices: List[str]) -> List[str]:
    if raw_devices == ["auto"]:
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("mps")
        return devices
    return raw_devices


def _dtype_supported_on_device(device: str, dtype_key: str) -> bool:
    try:
        dtype = DTYPE_TO_TORCH[dtype_key]
        if device.startswith("cuda") and not torch.cuda.is_available():
            return False
        if device == "mps":
            if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                return False
        if dtype_key == "fp16" and device == "cpu":
            return False
        # BF16 tensor ops are commonly unsupported on older CUDA cards and some CPUs.
        a = torch.randn((32, 32), device=device, dtype=dtype)
        b = torch.randn((32, 32), device=device, dtype=dtype)
        _ = a @ b
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        elif device == "mps":
            torch.mps.synchronize()
        return True
    except Exception:
        return False


def detect_environment(devices: List[str]) -> Dict:
    env = {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "cpu": platform.processor() or "unknown",
        "devices": {},
    }

    for device in devices:
        if device == "cpu":
            info = {"type": "cpu"}
        elif device == "mps":
            info = {"type": "mps", "name": "Apple Metal (MPS)"}
        else:
            info = {"type": "cuda"}
        if device.startswith("cuda") and torch.cuda.is_available():
            idx = int(device.split(":")[-1])
            props = torch.cuda.get_device_properties(idx)
            info.update(
                {
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "sm_count": props.multi_processor_count,
                    "major": props.major,
                    "minor": props.minor,
                }
            )
        env["devices"][device] = info
    return env


def detect_dtype_capabilities(devices: List[str]) -> Dict[str, Dict[str, List[str]]]:
    out: Dict[str, Dict[str, List[str]]] = {}
    for device in devices:
        supported = []
        unsupported = []
        for dtype_key in TENSOR_DTYPE_CANDIDATES:
            if _dtype_supported_on_device(device, dtype_key):
                supported.append(dtype_key)
            else:
                unsupported.append(dtype_key)
        out[device] = {
            "supported_tensor_dtypes": supported,
            "unsupported_tensor_dtypes": unsupported,
        }
    return out
