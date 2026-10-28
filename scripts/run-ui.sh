#!/usr/bin/env bash
#
# Launch the lethe web UI on this project's own .venv interpreter.
#
# `abliterate ui` runs on whichever Python is first on PATH. A bare
# `pip install -e .` also installs an `abliterate` launcher into a global
# interpreter, and that older copy wins -- so the UI silently serves with a
# transformers that may not know the model you are loading. Naming the
# interpreter here means the answer is never in doubt.
#
# Usage:
#   scripts/run-ui.sh
#   scripts/run-ui.sh --port 8000 --no-browser
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -x ".venv/bin/python" ]; then
    echo "No .venv yet - creating it first (one time, downloads torch)..."
    ./scripts/setup-env.sh
fi

exec .venv/bin/python -m abliterate.cli ui "$@"
