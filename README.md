# PyTorch CPU/GPU Benchmark Suite

This project benchmarks LLM inference and tokenization across CPU and CUDA devices with dtype coverage and report generation.
It also auto-detects Apple Metal (`mps`) when available, and reports host architecture (ARM/x86-64).

## What it benchmarks

- CPU tokenization throughput/latency
- Model inference throughput and latency (prefill/decode/total)
- DType coverage for `fp32`, `fp16`, `bf16`, `int8`, `int4` (bitsandbytes), and fallback kernel checks for `int16` and other tensor dtypes
- Kernel microbenchmarks (`matmul`) for guaranteed dtype throughput coverage
- Maximum stress tests under sustained load
- RAM speed and disk speed benchmarks
- Temperature monitoring and temperature report in HTML output
- Final suitability scoring for inference, training, and Torch playground/development
- Bottleneck diagnostics, upgrade suggestions, and throttling/high-temp alerts
- Optional AI-accelerated diagnosis via local Ollama model (default `qwen3.5:2b`)
- Support matrix and skip reasons for unsupported combinations

## Install

```bash
.venv/bin/pip install -r benchmarks/requirements.txt
```

## Run

Python entrypoint:

```bash
.venv/bin/python main.py --matrix quick --models sshleifer/tiny-gpt2
```

Shell entrypoint:

```bash
./run_benchmarks.sh --matrix quick --models sshleifer/tiny-gpt2
```

One-command full flow (install deps, run, open HTML report):

```bash
./run_benchmarks.sh --all --quick --models sshleifer/tiny-gpt2
```

## Outputs

Each run creates `artifacts/<timestamp>/` with:

- `results.json`
- `results.csv`
- `summary.md`
- `report.html`
- `supported_types.log`
- `environment.json`

Open report:

```bash
xdg-open artifacts/<timestamp>/report.html
```

## Key options

- `--devices auto|cpu,cuda:0`
- `--dtypes auto|fp32,fp16,bf16,int8,int4,int16`
- `--modes eager,compile`
- `--matrix quick|balanced|exhaustive`
- `--num-runs 10 --warmup-runs 3`
- `--disable-int4`
- `--disable-kernel-fallback`
- `--disable-stress-tests --stress-duration-sec 30`
- `--ram-bench-size-mb 512 --disk-bench-size-mb 512`
- `--disable-ai-diagnosis --ai-model qwen3.5:2b --ai-timeout-sec 90`
- `--all` / `--install` / `--run` / `--open-report`
