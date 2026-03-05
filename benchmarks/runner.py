from __future__ import annotations

import argparse
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .config import BenchmarkConfig, MATRIX_PRESETS, parse_csv_arg, parse_csv_int_arg
from .logging_utils import setup_logger, write_supported_types_log
from .discovery import MODEL_DTYPE_CANDIDATES, detect_dtype_capabilities, detect_environment, resolve_devices


def _git_rev() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_dtypes(user_dtypes: List[str]) -> List[str]:
    if user_dtypes == ["auto"]:
        return MODEL_DTYPE_CANDIDATES
    return user_dtypes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyTorch CPU/GPU benchmark suite")
    parser.add_argument("--models", default="sshleifer/tiny-gpt2")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--modes", default="eager,compile")
    parser.add_argument("--dtypes", default="auto")
    parser.add_argument("--matrix", choices=sorted(MATRIX_PRESETS.keys()), default="balanced")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt-lengths", default="")
    parser.add_argument("--batch-sizes", default="")
    parser.add_argument("--gen-lengths", default="")
    parser.add_argument("--enable-int4", action="store_true", default=True)
    parser.add_argument("--disable-int4", action="store_true", default=False)
    parser.add_argument("--enable-kernel-fallback", action="store_true", default=True)
    parser.add_argument("--disable-kernel-fallback", action="store_true", default=False)
    parser.add_argument("--enable-stress-tests", action="store_true", default=True)
    parser.add_argument("--disable-stress-tests", action="store_true", default=False)
    parser.add_argument("--stress-duration-sec", type=int, default=30)
    parser.add_argument("--ram-bench-size-mb", type=int, default=512)
    parser.add_argument("--disk-bench-size-mb", type=int, default=512)
    parser.add_argument("--enable-ai-diagnosis", action="store_true", default=True)
    parser.add_argument("--disable-ai-diagnosis", action="store_true", default=False)
    parser.add_argument("--ai-model", default="qwen3.5:2b")
    parser.add_argument("--ai-timeout-sec", type=int, default=90)
    parser.add_argument("--seed", type=int, default=1337)
    return parser


def _config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    cfg = BenchmarkConfig(
        models=parse_csv_arg(args.models),
        devices=parse_csv_arg(args.devices),
        modes=parse_csv_arg(args.modes),
        dtypes=parse_csv_arg(args.dtypes),
        matrix=args.matrix,
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs,
        max_new_tokens=args.max_new_tokens,
        prompt_lengths=parse_csv_int_arg(args.prompt_lengths) if args.prompt_lengths else [],
        batch_sizes=parse_csv_int_arg(args.batch_sizes) if args.batch_sizes else [],
        gen_lengths=parse_csv_int_arg(args.gen_lengths) if args.gen_lengths else [],
        enable_int4=not args.disable_int4,
        enable_kernel_fallback=not args.disable_kernel_fallback,
        enable_stress_tests=not args.disable_stress_tests,
        stress_duration_sec=args.stress_duration_sec,
        ram_bench_size_mb=args.ram_bench_size_mb,
        disk_bench_size_mb=args.disk_bench_size_mb,
        enable_ai_diagnosis=not args.disable_ai_diagnosis,
        ai_model=args.ai_model,
        ai_timeout_sec=args.ai_timeout_sec,
        seed=args.seed,
    )
    if args.output_dir:
        cfg.output_dir = Path(args.output_dir)
    return cfg


def run(cfg: BenchmarkConfig) -> Dict[str, Any]:
    logger = setup_logger()
    _seed_everything(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from .ai_advisor import generate_ai_diagnosis
        from .diagnostics import analyze_bottlenecks
        from .kernel_bench import benchmark_kernel_suite
        from .model_bench import benchmark_model_suite
        from .reporting import write_csv, write_html, write_json, write_markdown
        from .scoring import build_suitability_scores
        from .stress_bench import benchmark_stress_suite
        from .system_bench import benchmark_system_suite
        from .telemetry import TemperatureMonitor
        from .tokenizer_bench import benchmark_tokenizer_cpu
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing dependency: {exc}. Install requirements with "
            "'.venv/bin/pip install -r benchmarks/requirements.txt'."
        ) from exc

    devices = resolve_devices(cfg.devices)
    env = detect_environment(devices)
    dtype_caps = detect_dtype_capabilities(devices)

    selected_dtypes = _resolve_dtypes(cfg.dtypes)
    if not cfg.enable_int4 and "int4" in selected_dtypes:
        selected_dtypes = [d for d in selected_dtypes if d != "int4"]

    payload: Dict[str, Any] = {
        "metadata": {
            "run_id": cfg.output_dir.name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_rev": _git_rev(),
            "command": "benchmark-runner",
        },
        "environment": env,
        "capabilities": {
            "devices": dtype_caps,
            "models": {},
        },
        "benchmarks": {
            "tokenizer": {"suite": "tokenizer_cpu", "results": []},
            "model_inference": [],
            "kernel_microbench": {"suite": "kernel_microbench", "results": [], "skips": []},
            "stress": {"suite": "stress", "results": [], "skips": []},
            "system": {"suite": "system", "results": [], "skips": []},
        },
        "skips": [],
    }

    prompt_lengths = cfg.resolved_prompt_lengths()
    batch_sizes = cfg.resolved_batch_sizes()
    gen_lengths = cfg.resolved_gen_lengths()
    temp_monitor = TemperatureMonitor(poll_interval_sec=1.0)
    temp_monitor.start()

    try:
        for model_id in cfg.models:
            logger.info("Tokenizer benchmark for %s", model_id)
            try:
                tok = benchmark_tokenizer_cpu(
                    model_id=model_id,
                    prompt_lengths=prompt_lengths,
                    batch_sizes=batch_sizes,
                    warmup_runs=cfg.warmup_runs,
                    num_runs=cfg.num_runs,
                )
                payload["benchmarks"]["tokenizer"]["results"].extend(tok["results"])
            except Exception as exc:
                payload["skips"].append(
                    {
                        "suite": "tokenizer_cpu",
                        "model": model_id,
                        "reason": str(exc),
                    }
                )

            for device in devices:
                logger.info("Model benchmark for model=%s device=%s", model_id, device)
                out = benchmark_model_suite(
                    model_id=model_id,
                    device=device,
                    modes=cfg.modes,
                    dtypes=selected_dtypes,
                    prompt_lengths=prompt_lengths,
                    batch_sizes=batch_sizes,
                    gen_lengths=gen_lengths,
                    warmup_runs=cfg.warmup_runs,
                    num_runs=cfg.num_runs,
                )
                payload["benchmarks"]["model_inference"].append(out)
                payload["skips"].extend(out.get("skips", []))
                payload["capabilities"]["models"].setdefault(model_id, {}).update(out.get("supported", {}))

        if cfg.enable_kernel_fallback:
            logger.info("Kernel fallback benchmark")
            kern = benchmark_kernel_suite(
                devices=devices,
                dtypes=selected_dtypes,
                warmup_runs=cfg.warmup_runs,
                num_runs=cfg.num_runs,
            )
            payload["benchmarks"]["kernel_microbench"] = kern
            payload["skips"].extend(kern.get("skips", []))

        if cfg.enable_stress_tests:
            logger.info("Stress benchmark")
            stress = benchmark_stress_suite(
                devices=devices,
                dtypes=selected_dtypes,
                duration_sec=cfg.stress_duration_sec,
            )
            payload["benchmarks"]["stress"] = stress
            payload["skips"].extend(stress.get("skips", []))

        logger.info("System benchmark (RAM/Disk)")
        system = benchmark_system_suite(
            output_dir=cfg.output_dir,
            ram_size_mb=cfg.ram_bench_size_mb,
            disk_size_mb=cfg.disk_bench_size_mb,
        )
        payload["benchmarks"]["system"] = system
        payload["skips"].extend(system.get("skips", []))
    finally:
        temp_monitor.stop()
        payload["temperature"] = temp_monitor.summary()
        temp_monitor.close()

    payload["suitability"] = build_suitability_scores(payload)
    payload["diagnostics"] = analyze_bottlenecks(payload)
    if cfg.enable_ai_diagnosis:
        payload["ai_diagnosis"] = generate_ai_diagnosis(
            payload=payload,
            diagnostics=payload["diagnostics"],
            model=cfg.ai_model,
            timeout_sec=cfg.ai_timeout_sec,
        )
    else:
        payload["ai_diagnosis"] = {
            "enabled": False,
            "model": cfg.ai_model,
            "status": "disabled",
            "reason": None,
            "output": None,
        }

    write_supported_types_log(
        cfg.output_dir / "supported_types.log",
        payload["capabilities"],
    )

    results_json_path = cfg.output_dir / "results.json"
    results_csv_path = cfg.output_dir / "results.csv"
    summary_md_path = cfg.output_dir / "summary.md"
    report_html_path = cfg.output_dir / "report.html"

    write_json(results_json_path, payload)
    write_csv(results_csv_path, payload)
    write_markdown(summary_md_path, payload)
    write_html(
        report_html_path,
        payload,
        template_dir=Path(__file__).parent / "templates",
    )

    (cfg.output_dir / "environment.json").write_text(
        json.dumps(env, indent=2, sort_keys=True), encoding="utf-8"
    )

    report_link = report_html_path.resolve().as_uri()
    logger.info("Report HTML: %s", report_html_path.resolve())
    logger.info("Open report link: %s", report_link)
    print(f"Report link: {report_link}")

    return payload


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    cfg = _config_from_args(args)
    try:
        run(cfg)
    except RuntimeError as exc:
        print(f"error: {exc}")
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
