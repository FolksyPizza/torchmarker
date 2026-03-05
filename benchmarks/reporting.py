from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from jinja2 import Environment, FileSystemLoader, select_autoescape


def _flatten_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for row in payload.get("benchmarks", {}).get("tokenizer", {}).get("results", []):
        rows.append(row)
    for row in payload.get("benchmarks", {}).get("model_inference", []):
        rows.extend(row.get("results", []))
    for row in payload.get("benchmarks", {}).get("kernel_microbench", {}).get("results", []):
        rows.append(row)
    for row in payload.get("benchmarks", {}).get("stress", {}).get("results", []):
        rows.append(row)
    for row in payload.get("benchmarks", {}).get("system", {}).get("results", []):
        rows.append(row)
    return rows


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, payload: Dict[str, Any]) -> None:
    rows = _flatten_rows(payload)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(path: Path, payload: Dict[str, Any]) -> None:
    token_rows = payload.get("benchmarks", {}).get("tokenizer", {}).get("results", [])
    model_runs = payload.get("benchmarks", {}).get("model_inference", [])
    kernel_rows = payload.get("benchmarks", {}).get("kernel_microbench", {}).get("results", [])
    stress_rows = payload.get("benchmarks", {}).get("stress", {}).get("results", [])
    system_rows = payload.get("benchmarks", {}).get("system", {}).get("results", [])
    skips = payload.get("skips", [])
    temp = payload.get("temperature", {})
    suitability = payload.get("suitability", {})
    diagnostics = payload.get("diagnostics", {})
    ai_diag = payload.get("ai_diagnosis", {})

    lines = ["# Benchmark Summary", ""]
    lines.append("## Environment")
    lines.append(f"- Torch: {payload.get('environment', {}).get('torch_version')}")
    lines.append(f"- CUDA available: {payload.get('environment', {}).get('cuda_available')}")
    lines.append(f"- Devices: {', '.join(payload.get('environment', {}).get('devices', {}).keys())}")
    lines.append("")

    lines.append("## Counts")
    lines.append(f"- Tokenizer rows: {len(token_rows)}")
    lines.append(f"- Model rows: {sum(len(x.get('results', [])) for x in model_runs)}")
    lines.append(f"- Kernel rows: {len(kernel_rows)}")
    lines.append(f"- Stress rows: {len(stress_rows)}")
    lines.append(f"- System rows: {len(system_rows)}")
    lines.append(f"- Skips: {len(skips)}")
    lines.append("")
    lines.append("## Temperature")
    lines.append(f"- Samples: {temp.get('sample_count', 0)}")
    lines.append(f"- CPU max temp (C): {temp.get('cpu_max_temp_c')}")
    lines.append(f"- GPU max temp (C): {temp.get('gpu_max_temp_c')}")
    lines.append("")
    lines.append("## Suitability Scores")
    lines.append(f"- Inference: {suitability.get('inference', {}).get('score')} ({suitability.get('inference', {}).get('tier')})")
    lines.append(f"- Training: {suitability.get('training', {}).get('score')} ({suitability.get('training', {}).get('tier')})")
    lines.append(
        f"- Torch Playground/Dev: {suitability.get('torch_playground_dev', {}).get('score')} "
        f"({suitability.get('torch_playground_dev', {}).get('tier')})"
    )
    lines.append("")
    lines.append("## Bottlenecks")
    for item in diagnostics.get("bottlenecks", [])[:20]:
        lines.append(f"- [{item.get('severity')}] {item.get('area')}: {item.get('issue')}")
    if not diagnostics.get("bottlenecks"):
        lines.append("- No major bottlenecks detected.")
    lines.append("")
    lines.append("## Upgrade Suggestions")
    for s in diagnostics.get("upgrade_suggestions", [])[:20]:
        lines.append(f"- {s}")
    if not diagnostics.get("upgrade_suggestions"):
        lines.append("- None")
    lines.append("")
    lines.append("## AI Diagnosis")
    lines.append(f"- Status: {ai_diag.get('status')}")
    lines.append(f"- Model: {ai_diag.get('model')}")
    if ai_diag.get("reason"):
        lines.append(f"- Reason: {ai_diag.get('reason')}")
    if ai_diag.get("output"):
        lines.append("- Output:")
        lines.append(ai_diag.get("output"))
    lines.append("")

    lines.append("## Skip Reasons")
    if skips:
        for item in skips[:50]:
            lines.append(f"- {item}")
    else:
        lines.append("- None")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_html(path: Path, payload: Dict[str, Any], template_dir: Path) -> None:
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.j2")

    html = template.render(
        metadata=payload.get("metadata", {}),
        environment=payload.get("environment", {}),
        capabilities=payload.get("capabilities", {}),
        tokenizer=payload.get("benchmarks", {}).get("tokenizer", {}),
        model_runs=payload.get("benchmarks", {}).get("model_inference", []),
        kernel=payload.get("benchmarks", {}).get("kernel_microbench", {}),
        stress=payload.get("benchmarks", {}).get("stress", {}),
        system=payload.get("benchmarks", {}).get("system", {}),
        temperature=payload.get("temperature", {}),
        suitability=payload.get("suitability", {}),
        diagnostics=payload.get("diagnostics", {}),
        ai_diagnosis=payload.get("ai_diagnosis", {}),
        skips=payload.get("skips", []),
    )
    path.write_text(html, encoding="utf-8")
