#!/usr/bin/env bash
#
# Create an isolated .venv for lethe and install the project into it.
#
# lethe is a library plus a CLI, and both import torch and transformers. Install
# it with a bare `pip install -e .` and those land in whichever interpreter
# happens to be first on PATH -- normally a global one shared with everything
# else on the machine. Any other tool that then pins a different torch or
# transformers silently breaks lethe, and lethe silently breaks it.
#
# This script gives lethe its own environment, so its dependencies cannot be
# shadowed by a global install and cannot leak into one.
#
# Usage:
#   scripts/setup-env.sh              # CUDA torch (cu126 channel)
#   scripts/setup-env.sh --cpu        # CPU-only torch, ~200 MB instead of ~2.6 GB
#   PYTHON=/usr/bin/python3.11 scripts/setup-env.sh
#   TORCH_INDEX=https://download.pytorch.org/whl/cu124 scripts/setup-env.sh
#
set -euo pipefail

# transformers spec: newest release that ships the current architectures
# (Qwen3.5 and friends need >= 5.2) and still supports Python 3.10; 5.21+
# requires Python >= 3.11.
TRANSFORMERS="${TRANSFORMERS:->=5.20,<5.21}"
EXTRAS="${EXTRAS:-dev}"
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
PYTHON="${PYTHON:-python3}"
CPU=0

for arg in "$@"; do
    case "$arg" in
        --cpu)  CPU=1 ;;
        -h|--help)
            sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
VENV_PY="$VENV/bin/python"

command -v "$PYTHON" >/dev/null 2>&1 || {
    echo "Interpreter '$PYTHON' not found. Set PYTHON=/path/to/python3." >&2
    exit 1
}

VERSION="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
# lethe's floor; transformers 5.x needs torch >= 2.6 anyway.
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' || {
    echo "lethe requires Python >= 3.10; '$PYTHON' is $VERSION." >&2
    exit 1
}

echo "==> interpreter : $("$PYTHON" -c 'import sys; print(sys.executable)')  ($VERSION)"

if [ ! -x "$VENV_PY" ]; then
    echo "==> creating    : $VENV"
    "$PYTHON" -m venv "$VENV"
else
    echo "==> reusing     : $VENV"
fi

echo "==> upgrading   : pip / setuptools / wheel"
"$VENV_PY" -m pip install --quiet --upgrade pip setuptools wheel

if [ "$CPU" -eq 1 ]; then
    echo "==> installing  : torch (CPU-only)"
    "$VENV_PY" -m pip install "torch==$TORCH_VERSION"
else
    echo "==> installing  : torch $TORCH_VERSION from $TORCH_INDEX"
    # torch comes from its own index; everything else stays on PyPI, hence two
    # installs rather than one with a global --index-url.
    "$VENV_PY" -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX" || {
        echo "torch install failed. If that CUDA channel does not match your driver," >&2
        echo "retry with --cpu, or pass one that does (cu121/cu124/cu126/cu128)." >&2
        exit 1
    }
fi

echo "==> installing  : lethe[$EXTRAS] + transformers$TRANSFORMERS (editable)"
"$VENV_PY" -m pip install -e "$ROOT[$EXTRAS]" "transformers$TRANSFORMERS"

echo
echo "==> done. Environment is isolated from your global interpreter:"
"$VENV_PY" - <<'PY'
import sys, torch, transformers
print('    python       ', sys.version.split()[0])
print('    torch        ', torch.__version__, '(cuda available:', torch.cuda.is_available(), ')')
print('    transformers ', transformers.__version__)
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as C
print('    qwen3_5 known:', 'qwen3_5' in C)
PY

echo
echo "Activate it:"
echo "    source .venv/bin/activate"
echo "Or call it directly without activating:"
echo "    .venv/bin/python -m abliterate.cli --help"
echo "    .venv/bin/abliterate --help"
