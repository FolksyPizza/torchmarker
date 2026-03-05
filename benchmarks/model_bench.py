from __future__ import annotations

import gc
import statistics
import time
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer

from .quantization import load_model_with_dtype


def _sync_if_needed(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elif device == "mps":
        torch.mps.synchronize()


def _peak_memory(device: str) -> Dict[str, float]:
    if device.startswith("cuda"):
        allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
        reserved = torch.cuda.max_memory_reserved(device) / (1024**2)
        return {"gpu_max_allocated_mb": round(allocated, 2), "gpu_max_reserved_mb": round(reserved, 2)}
    if device == "mps" and hasattr(torch, "mps"):
        try:
            allocated = torch.mps.current_allocated_memory() / (1024**2)
            driver = torch.mps.driver_allocated_memory() / (1024**2)
            return {"gpu_max_allocated_mb": round(allocated, 2), "gpu_max_reserved_mb": round(driver, 2)}
        except Exception:
            return {"gpu_max_allocated_mb": 0.0, "gpu_max_reserved_mb": 0.0}
    return {"gpu_max_allocated_mb": 0.0, "gpu_max_reserved_mb": 0.0}


def _build_prompts(prompt_len: int, batch_size: int) -> List[str]:
    words = max(8, prompt_len // 2)
    prompt = " ".join(["throughput"] * words)
    return [prompt] * batch_size


def _run_generate(model, enc, max_new_tokens: int, device: str) -> Tuple[float, int]:
    _sync_if_needed(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=getattr(model.config, "eos_token_id", None),
        )
    _sync_if_needed(device)
    elapsed = time.perf_counter() - t0
    generated = int(out.shape[-1] - enc["input_ids"].shape[-1])
    return elapsed, max(generated, 0)


def benchmark_model_suite(
    model_id: str,
    device: str,
    modes: List[str],
    dtypes: List[str],
    prompt_lengths: List[int],
    batch_sizes: List[int],
    gen_lengths: List[int],
    warmup_runs: int,
    num_runs: int,
) -> Dict:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    results = []
    skips = []
    supported = {}

    for dtype_key in dtypes:
        for mode in modes:
            try:
                model, meta = load_model_with_dtype(model_id, dtype_key, device)
                if model.get_input_embeddings().num_embeddings < len(tokenizer):
                    model.resize_token_embeddings(len(tokenizer))
                if mode == "compile":
                    try:
                        model = torch.compile(model)
                    except Exception as exc:
                        raise RuntimeError(f"compile_error: {exc}")
                model.eval()
                supported[f"{dtype_key}:{mode}:{device}"] = True
            except Exception as exc:
                supported[f"{dtype_key}:{mode}:{device}"] = False
                skips.append(
                    {
                        "suite": "model_inference",
                        "model": model_id,
                        "device": device,
                        "dtype": dtype_key,
                        "mode": mode,
                        "reason": str(exc),
                    }
                )
                continue
            max_positions = int(
                getattr(model.config, "max_position_embeddings", 0) or getattr(model.config, "n_positions", 0) or 0
            )

            for plen in prompt_lengths:
                for batch in batch_sizes:
                    max_prompt_len = plen
                    if max_positions > 0:
                        max_prompt_len = max(1, min(plen, max_positions - 1))
                    prompts = _build_prompts(plen, batch)
                    enc = tokenizer(
                        prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_prompt_len,
                    )
                    if device != "cpu":
                        enc = {k: v.to(device) for k, v in enc.items()}

                    for gen_len in gen_lengths:
                        input_len = int(enc["input_ids"].shape[-1])
                        safe_gen_len = gen_len
                        if max_positions > 0:
                            safe_gen_len = max_positions - input_len
                        if safe_gen_len <= 0:
                            skips.append(
                                {
                                    "suite": "model_inference",
                                    "model": model_id,
                                    "device": device,
                                    "dtype": dtype_key,
                                    "mode": mode,
                                    "reason": (
                                        f"skipped: context limit exceeded (input_len={input_len}, "
                                        f"max_positions={max_positions})"
                                    ),
                                }
                            )
                            continue
                        safe_gen_len = min(gen_len, safe_gen_len)
                        latencies = []
                        gen_tokens_total = 0

                        if device.startswith("cuda"):
                            torch.cuda.reset_peak_memory_stats(device)

                        run_error = None
                        try:
                            for _ in range(warmup_runs):
                                _run_generate(model, enc, safe_gen_len, device)

                            for _ in range(num_runs):
                                elapsed, gen_tok = _run_generate(model, enc, safe_gen_len, device)
                                latencies.append(elapsed)
                                gen_tokens_total += gen_tok * batch
                        except Exception as exc:
                            run_error = exc
                        if run_error is not None:
                            skips.append(
                                {
                                    "suite": "model_inference",
                                    "model": model_id,
                                    "device": device,
                                    "dtype": dtype_key,
                                    "mode": mode,
                                    "reason": f"generate_failed: {run_error}",
                                    "prompt_len": plen,
                                    "batch": batch,
                                    "gen_len": gen_len,
                                }
                            )
                            continue

                        total_time = sum(latencies)
                        input_tokens = int(enc["input_ids"].numel()) * num_runs
                        prefill_tps = input_tokens / total_time if total_time > 0 else 0.0
                        decode_tps = gen_tokens_total / total_time if total_time > 0 else 0.0
                        total_tps = (input_tokens + gen_tokens_total) / total_time if total_time > 0 else 0.0

                        row = {
                            "suite": "model_inference",
                            "model": model_id,
                            "device": device,
                            "dtype": dtype_key,
                            "mode": mode,
                            "prompt_len": plen,
                            "batch": batch,
                            "gen_len": safe_gen_len,
                            "num_runs": num_runs,
                            "prefill_tokens_per_sec": round(prefill_tps, 2),
                            "decode_tokens_per_sec": round(decode_tps, 2),
                            "total_tokens_per_sec": round(total_tps, 2),
                            "p50_latency_ms": round(statistics.median(latencies) * 1000, 4),
                            "p95_latency_ms": round(statistics.quantiles(latencies, n=20)[-1] * 1000, 4)
                            if len(latencies) >= 2
                            else round(latencies[0] * 1000, 4),
                        }
                        row.update(_peak_memory(device))
                        row.update({"quantized": meta.get("quantized", False)})
                        results.append(row)

            del model
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    return {
        "suite": "model_inference",
        "model": model_id,
        "device": device,
        "results": results,
        "skips": skips,
        "supported": supported,
    }
