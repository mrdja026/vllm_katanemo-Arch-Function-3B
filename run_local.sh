#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
AUTO_INSTALL_CUDA_TOOLKIT=${AUTO_INSTALL_CUDA_TOOLKIT:-false}

if ! command -v nvcc >/dev/null 2>&1; then
  if [ "$AUTO_INSTALL_CUDA_TOOLKIT" = "true" ]; then
    echo "nvcc not found. Installing CUDA toolkit via apt..."
    sudo apt-get update
    sudo apt-get install -y nvidia-cuda-toolkit
  else
    echo "nvcc not found. Set AUTO_INSTALL_CUDA_TOOLKIT=true to install via apt."
  fi
fi

if [ -d "/usr/local/cuda/bin" ] && ! echo "$PATH" | grep -q "/usr/local/cuda/bin"; then
  export PATH="/usr/local/cuda/bin:$PATH"
fi

if [ ! -x "$VENV_DIR/bin/python" ] || [ ! -f "$VENV_DIR/bin/activate" ]; then
  rm -rf "$VENV_DIR"
  if ! python3 -m venv "$VENV_DIR"; then
    echo "Failed to create venv. Install python3-venv and retry."
    echo "Example: sudo apt-get update && sudo apt-get install -y python3-venv"
    exit 1
  fi
fi

VENV_PY="$VENV_DIR/bin/python"
VENV_STREAMLIT="$VENV_DIR/bin/streamlit"

if [ ! -x "$VENV_PY" ]; then
  echo "Python not found in venv. Remove .venv and retry."
  exit 1
fi

"$VENV_PY" -m pip install -U pip
"$VENV_PY" -m pip install -r "$PROJECT_DIR/requirements-app.txt"

export SERVICE_HOST="0.0.0.0"
export SERVICE_PORT="8100"
export VLLM_PORT="8000"
export VLLM_API_URL="http://localhost:${VLLM_PORT}/v1/completions"
export VLLM_MODEL_ID="katanemo/Arch-Function-3B"

"$VENV_PY" "$PROJECT_DIR/quant_arch_function_3b.py" &
SERVICE_PID=$!

cleanup() {
  if kill -0 "$SERVICE_PID" 2>/dev/null; then
    kill "$SERVICE_PID"
  fi
}
trap cleanup EXIT

echo "" | "$VENV_STREAMLIT" run "$PROJECT_DIR/streamlit_app.py" --server.address 0.0.0.0 --server.port 8501
