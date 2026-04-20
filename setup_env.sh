#!/usr/bin/env bash
# Sets up a Python virtual environment with all dependencies for InferLens / vidur.
# Usage: bash setup_env.sh [--dev]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO_ROOT/.venv"
DEV=false

for arg in "$@"; do
  [[ "$arg" == "--dev" ]] && DEV=true
done

# ── 1. Check Python 3.10+ ────────────────────────────────────────────────────
find_python() {
  for cmd in python3.10 python3.11 python3.12 python3 python; do
    if command -v "$cmd" &>/dev/null; then
      version=$("$cmd" -c 'import sys; print(sys.version_info[:2])')
      if "$cmd" -c 'import sys; assert sys.version_info >= (3,10)' 2>/dev/null; then
        echo "$cmd"; return 0
      fi
    fi
  done
  return 1
}

echo "==> Checking Python version..."
if ! PYTHON=$(find_python); then
  echo ""
  echo "ERROR: Python 3.10 or higher is required but was not found."
  echo "Install it from https://www.python.org/downloads/ or via your package manager:"
  echo "  Ubuntu/Debian : sudo apt install python3.10"
  echo "  macOS (brew)  : brew install python@3.10"
  exit 1
fi
echo "    Found: $PYTHON ($($PYTHON --version))"

# ── 2. Create virtual environment ────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
  echo "==> Creating virtual environment at .venv ..."
  "$PYTHON" -m venv "$VENV"
else
  echo "==> Virtual environment already exists at .venv (reusing)"
fi

VENV_PYTHON="$VENV/bin/python"
VENV_PIP="$VENV/bin/pip"

# ── 3. Upgrade pip / build tools ─────────────────────────────────────────────
echo "==> Upgrading pip, setuptools, wheel..."
"$VENV_PYTHON" -m pip install --quiet --upgrade pip setuptools wheel

# ── 4. Core runtime dependencies ─────────────────────────────────────────────
echo "==> Installing core dependencies (requirements.txt)..."
"$VENV_PIP" install --quiet -r "$REPO_ROOT/requirements.txt"

# ── 5. Config-optimizer extras (streamlit, ray, etc.) ────────────────────────
echo "==> Installing config-optimizer extras (streamlit, ray, randomname, paretoset, snakeviz)..."
"$VENV_PIP" install --quiet \
  streamlit \
  "ray[default]" \
  randomname \
  paretoset \
  snakeviz \
  pyyaml

# ── 6. Dev tools (linting/formatting) ─────────────────────────────────────────
if [[ "$DEV" == true ]]; then
  echo "==> Installing dev dependencies (requirements-dev.txt)..."
  "$VENV_PIP" install --quiet -r "$REPO_ROOT/requirements-dev.txt"
fi

# ── 7. Install the package in editable mode ───────────────────────────────────
echo "==> Installing vidur package in editable mode..."
"$VENV_PIP" install --quiet -e "$REPO_ROOT"

# ── 8. W&B default: disabled ─────────────────────────────────────────────────
ENV_FILE="$REPO_ROOT/.env"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "==> Writing .env with WANDB_MODE=disabled ..."
  echo 'WANDB_MODE=disabled' > "$ENV_FILE"
else
  if ! grep -q "WANDB_MODE" "$ENV_FILE"; then
    echo 'WANDB_MODE=disabled' >> "$ENV_FILE"
  fi
fi

# ── 9. Smoke test ─────────────────────────────────────────────────────────────
echo "==> Running smoke test..."
"$VENV_PYTHON" -c "
from vidur.config import SimulationConfig
from vidur.execution_time_predictor import RandomForrestExecutionTimePredictor
print('    Imports OK')
" 2>&1 | sed 's/^/    /'

# ── 10. Print quickstart ──────────────────────────────────────────────────────
cat <<'EOF'

══════════════════════════════════════════════════════════════════
  Environment ready!  Activate with:

      source .venv/bin/activate

  Quick run (Llama-2-7B on A100, synthetic workload, 64 requests):

      python -m vidur.main \
        --replica_config_device a100 \
        --replica_config_model_name meta-llama/Llama-2-7b-hf \
        --cluster_config_num_replicas 1 \
        --replica_config_tensor_parallel_size 1 \
        --replica_config_num_pipeline_stages 1 \
        --request_generator_config_type synthetic \
        --synthetic_request_generator_config_num_requests 64 \
        --length_generator_config_type uniform \
        --interval_generator_config_type poisson \
        --poisson_request_interval_generator_config_qps 2.0 \
        --replica_scheduler_config_type sarathi \
        --sarathi_scheduler_config_batch_size_cap 32 \
        --sarathi_scheduler_config_chunk_size 128

  Output lands in:  simulator_output/<timestamp>/

  All CLI options:  python -m vidur.main --help

  W&B is disabled by default (.env). To enable:
      export WANDB_MODE=online && wandb login

══════════════════════════════════════════════════════════════════
EOF
