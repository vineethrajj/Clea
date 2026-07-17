#!/usr/bin/env bash
# Setup (first run) + start the Clea web UI on Linux/macOS.
set -e
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    ./.venv/bin/pip install --upgrade pip >/dev/null
    ./.venv/bin/pip install -r requirements.txt
    ./.venv/bin/python -m clea doctor
fi
exec ./.venv/bin/python -m clea serve --port 8000
