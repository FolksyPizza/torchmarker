from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM


def load_model_with_dtype(
    model_id: str,
    dtype_key: str,
    device: str,
    trust_remote_code: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    meta: Dict[str, Any] = {"dtype": dtype_key, "device": device, "quantized": False}

    if dtype_key == "int4":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"bitsandbytes config unavailable: {exc}")
        qconf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=qconf,
            device_map="auto" if device.startswith("cuda") else None,
            use_safetensors=False,
            trust_remote_code=trust_remote_code,
        )
        meta["quantized"] = True
        return model, meta

    if dtype_key == "int8":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"bitsandbytes config unavailable: {exc}")
        qconf = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=qconf,
            device_map="auto" if device.startswith("cuda") else None,
            use_safetensors=False,
            trust_remote_code=trust_remote_code,
        )
        meta["quantized"] = True
        return model, meta

    torch_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(dtype_key)

    if torch_dtype is None:
        raise RuntimeError(f"Model path unsupported dtype: {dtype_key}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        use_safetensors=False,
        trust_remote_code=trust_remote_code,
    )
    if device != "cpu":
        model.to(device)
    return model, meta
