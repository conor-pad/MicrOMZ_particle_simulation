#!/bin/bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
CODE_DIR="./"

# Change this default when you want a different job without passing args.
DEFAULT_JOB="run_suite_parallel"

# You can pass either a Python script filename or its basename:
#   bash run_py.slurm spectrum_eke_monthly
#   bash run_py.slurm spectrum_eke_monthly.py
REQUESTED_JOB="${1:-$DEFAULT_JOB}"

shopt -s nullglob
PYTHON_SCRIPTS=("$CODE_DIR"/*.py)
shopt -u nullglob

if [[ ${#PYTHON_SCRIPTS[@]} -eq 0 ]]; then
    echo "No Python scripts found in $CODE_DIR" >&2
    exit 1
fi

JOB_SCRIPT=""
for script_path in "${PYTHON_SCRIPTS[@]}"; do
    script_name="$(basename "$script_path")"
    script_base="${script_name%.py}"
    if [[ "$REQUESTED_JOB" == "$script_name" || "$REQUESTED_JOB" == "$script_base" ]]; then
        JOB_SCRIPT="$script_name"
        break
    fi
done

if [[ -z "$JOB_SCRIPT" ]]; then
    echo "Unknown job: $REQUESTED_JOB" >&2
    echo "Available jobs:" >&2
    for script_path in "${PYTHON_SCRIPTS[@]}"; do
        echo "  $(basename "$script_path")" >&2
    done
    echo "You can pass either the filename or basename without .py" >&2
    exit 1
fi

SCRIPT_PATH="$CODE_DIR/$JOB_SCRIPT"
if [[ ! -f "$SCRIPT_PATH" ]]; then
    echo "Python script not found: $SCRIPT_PATH" >&2
    exit 1
fi

JOB_NAME="${JOB_SCRIPT%.py}"
LOGFILE="$CODE_DIR/outlog_${JOB_NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "Launching Python job: $JOB_SCRIPT"
echo "Log file: $LOGFILE"

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 nohup "$PYTHON_BIN" "$SCRIPT_PATH" > "$LOGFILE" 2>&1 &

echo "Started PID: $!"
