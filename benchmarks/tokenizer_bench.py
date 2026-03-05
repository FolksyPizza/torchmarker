from __future__ import annotations

import statistics
import time
from typing import Dict, List

from transformers import AutoTokenizer


def _make_text(target_tokens: int) -> str:
    # Approximate 1 token ~= 4 chars for English-ish text.
    words = max(8, target_tokens // 2)
    return " ".join(["benchmark"] * words)


def benchmark_tokenizer_cpu(
    model_id: str,
    prompt_lengths: List[int],
    batch_sizes: List[int],
    warmup_runs: int,
    num_runs: int,
) -> Dict:
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
    results = []

    for plen in prompt_lengths:
        sample = _make_text(plen)
        for batch in batch_sizes:
            texts = [sample] * batch
            latencies = []

            for _ in range(warmup_runs):
                tok(texts, padding=True, truncation=True, return_tensors="pt")

            total_tokens = 0
            for _ in range(num_runs):
                t0 = time.perf_counter()
                enc = tok(texts, padding=True, truncation=True, return_tensors="pt")
                t1 = time.perf_counter()
                lat = t1 - t0
                latencies.append(lat)
                total_tokens += int(enc["input_ids"].numel())

            total_time = sum(latencies)
            results.append(
                {
                    "model": model_id,
                    "prompt_len": plen,
                    "batch": batch,
                    "num_runs": num_runs,
                    "p50_latency_ms": round(statistics.median(latencies) * 1000, 4),
                    "p95_latency_ms": round(statistics.quantiles(latencies, n=20)[-1] * 1000, 4)
                    if len(latencies) >= 2
                    else round(latencies[0] * 1000, 4),
                    "tokens_per_sec": round(total_tokens / total_time, 2) if total_time > 0 else 0.0,
                    "docs_per_sec": round((batch * num_runs) / total_time, 2) if total_time > 0 else 0.0,
                }
            )

    return {"suite": "tokenizer_cpu", "results": results}
