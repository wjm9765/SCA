#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
MODEL_DIR="${SCRIPT_DIR}/.hf_models"
MODEL_TAG="Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_INSTALL_DIR="${MODEL_DIR}/${MODEL_TAG//\//_}"
LOCAL_VENV_DIR="$HOME/uv_venv"

if [ "$(uname)" != "Darwin" ] && [ "$(uname)" != "Linux" ]; then
    echo "Unsupported OS: $(uname). This script supports only Linux and macOS."
    exit 1
fi

if [ "$(uname)" == "Linux" ]; then
  if command -v apt $> /dev/null; then
    if ! command -v sudo &> /dev/null; then
        echo "Installing sudo..."
        apt update && apt install -y sudo
        if $? -ne 0; then
            echo "Failed to install sudo. Please install it manually and re-run the script."
            exit 1
        fi
    fi
    sudo apt update && sudo apt install -y gh btop nvtop screen git
  fi
fi

if [ "${GITHUB_TOKEN}" != "" ]; then
  echo "Authenticating gh with provided GITHUB_TOKEN..."
  echo "${GITHUB_TOKEN}" | gh auth login --with-token
  gh auth setup-git
  GH_USER=$(gh api -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" /user | jq -r .login)
  GH_EMAIL=$(gh api -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" /user/emails | jq -r ".[0].email")
  git config --global user.name "${GH_USER}"
  git config --global user.email "${GH_EMAIL}"
  echo "Git configured with user: ${GH_USER}, email: ${GH_EMAIL}"
fi

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

UV_EXTRA="--extra cpu"
DETECTED_CUDA=""

if command -v nvcc &> /dev/null; then
    DETECTED_CUDA=$(nvcc --version | sed -n 's/^.*release \([0-9]\+\.[0-9]\+\).*$/\1/p')
    echo "Detected NVCC: $DETECTED_CUDA"

elif command -v nvidia-smi &> /dev/null; then
    DETECTED_CUDA=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "Detected NVIDIA Driver (Max CUDA): $DETECTED_CUDA"
fi

if [[ -n "$DETECTED_CUDA" ]]; then
    CUDA_CLEAN=$(echo "$DETECTED_CUDA" | tr -d '.')

    if [[ "$CUDA_CLEAN" -ge 130 ]]; then
        UV_EXTRA="--extra cu130"
    elif [[ "$CUDA_CLEAN" -ge 128 ]]; then
        UV_EXTRA="--extra cu128"
    elif [[ "$CUDA_CLEAN" -ge 126 ]]; then
        UV_EXTRA="--extra cu126"
    else
        echo "Warning: CUDA version $DETECTED_CUDA is too old (<12.0). Defaulting to CPU."
    fi
else
    echo "No NVIDIA GPU detected. Defaulting to CPU."
fi

if [ ! -d "$LOCAL_VENV_DIR" ]; then
    echo "Creating local venv storage at $LOCAL_VENV_DIR"
    mkdir -p "$LOCAL_VENV_DIR"
else
    echo "Local venv storage already exists at $LOCAL_VENV_DIR"
fi

rm -rf "${SCRIPT_DIR}/.venv"
echo "Symlinking venv to local storage"
ln -s "$LOCAL_VENV_DIR" "${SCRIPT_DIR}/.venv"

echo "Running: uv sync $UV_EXTRA"
cd "$SCRIPT_DIR" || exit 1
uv sync $UV_EXTRA --frozen --dev

if [ ! -d "${MODEL_INSTALL_DIR}" ]; then
  echo "Downloading model to ${MODEL_INSTALL_DIR} ..."
  mkdir -p "${MODEL_INSTALL_DIR}"
  uv tool run --with hf_transfer hf download \
    --cache-dir "${MODEL_DIR}" \
    --repo-type model \
    --max-workers 16 \
    "${MODEL_TAG}"
else
  echo "Model directory ${MODEL_INSTALL_DIR} already exists. Skipping download."
fi
