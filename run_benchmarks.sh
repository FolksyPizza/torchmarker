#!/usr/bin/env bash
set -euo pipefail

if [[ -x ".venv/bin/python" ]]; then
  DEFAULT_PY=".venv/bin/python"
else
  DEFAULT_PY="python3"
fi

PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PY}"

DO_INSTALL=0
DO_RUN=0
DO_OPEN=0
MATRIX=""
MODELS=""
OUTPUT_DIR=""
EXTRA_ARGS=()

usage() {
  cat <<'USAGE'
Usage: ./run_benchmarks.sh [flags] [-- <extra benchmark args>]

Orchestration flags:
  --all               Install deps, run benchmark, and open report
  --install           Install benchmark dependencies only
  --run               Run benchmark only
  --open-report       Open latest report.html under artifacts/
  --quick             Shortcut for --matrix quick
  --matrix <name>     quick|balanced|exhaustive
  --models <ids>      Comma-separated model IDs
  --output-dir <dir>  Override benchmark output directory
  -h, --help          Show this help

Any remaining args are forwarded to main.py.
USAGE
}

if [[ $# -eq 0 ]]; then
  DO_INSTALL=1
  DO_RUN=1
  DO_OPEN=1
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      DO_INSTALL=1
      DO_RUN=1
      DO_OPEN=1
      shift
      ;;
    --install)
      DO_INSTALL=1
      shift
      ;;
    --run)
      DO_RUN=1
      shift
      ;;
    --open-report)
      DO_OPEN=1
      shift
      ;;
    --quick)
      MATRIX="quick"
      shift
      ;;
    --matrix)
      MATRIX="$2"
      shift 2
      ;;
    --models)
      MODELS="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$DO_INSTALL" -eq 0 && "$DO_RUN" -eq 0 && "$DO_OPEN" -eq 0 ]]; then
  DO_INSTALL=1
  DO_RUN=1
  DO_OPEN=1
fi

if [[ "$DO_INSTALL" -eq 1 ]]; then
  "$PYTHON_BIN" -m pip install -r benchmarks/requirements.txt
fi

if [[ "$DO_RUN" -eq 1 ]]; then
  RUN_ARGS=()
  if [[ -n "$MATRIX" ]]; then
    RUN_ARGS+=(--matrix "$MATRIX")
  fi
  if [[ -n "$MODELS" ]]; then
    RUN_ARGS+=(--models "$MODELS")
  fi
  if [[ -n "$OUTPUT_DIR" ]]; then
    RUN_ARGS+=(--output-dir "$OUTPUT_DIR")
  fi
  RUN_ARGS+=("${EXTRA_ARGS[@]}")
  "$PYTHON_BIN" main.py "${RUN_ARGS[@]}"
fi

if [[ "$DO_OPEN" -eq 1 ]]; then
  REPORT_PATH=""
  if [[ -n "$OUTPUT_DIR" ]]; then
    REPORT_PATH="$OUTPUT_DIR/report.html"
  else
    LATEST_DIR="$(ls -1dt artifacts/* 2>/dev/null | head -n1 || true)"
    if [[ -n "$LATEST_DIR" ]]; then
      REPORT_PATH="$LATEST_DIR/report.html"
    fi
  fi
  if [[ -n "$REPORT_PATH" && -f "$REPORT_PATH" ]]; then
    if command -v xdg-open >/dev/null 2>&1; then
      xdg-open "$REPORT_PATH" >/dev/null 2>&1 || true
      echo "Opened report: $REPORT_PATH"
    else
      echo "Report ready: $REPORT_PATH"
    fi
  else
    echo "No report found to open."
  fi
fi
