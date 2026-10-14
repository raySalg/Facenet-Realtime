#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

if [ -f ".venv/bin/python" ]; then
    PYTHON_EXEC=".venv/bin/python"
else
    PYTHON_EXEC="python3"
fi

$PYTHON_EXEC benchmark_tempo_real.py "$@"
