#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# QWEN-VALERA Setup Script — for JetPack 6.x (after Jetson upgrade)
#
# Run AFTER upgrading your Jetson AGX Orin to JetPack 6.x.
# This script:
#   1. Installs system dependencies
#   2. Installs NVIDIA Jetson PyTorch 2.5.0 (Python 3.10)
#   3. Installs transformers from GitHub (with Qwen3-Omni support)
#   4. Installs all project dependencies
#   5. Downloads the AWQ 4-bit model
# ═══════════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PASSWORD="1055"
ENV_NAME="qwen-valera"

echo "============================================"
echo " QWEN-VALERA Setup (JetPack 6.x)"
echo "============================================"
echo ""

# ── Verify JetPack version ─────────────────────────────────────────────────────

echo "[CHECK] Verifying JetPack version..."
JP_VER=$(dpkg -l 2>/dev/null | grep -oP 'nvidia-l4t-core\s+\S+\s+\K\S+' || echo "unknown")
echo "  L4T core version: $JP_VER"
if [[ "$JP_VER" == 36.* ]]; then
    echo "  ✓ JetPack 6.x detected"
elif [[ "$JP_VER" == 35.* ]]; then
    echo "  ⚠️  JetPack 5.x detected! This script requires JetPack 6.x."
    echo "  Please upgrade your Jetson first (see JETPACK_UPGRADE.md)"
    exit 1
fi
echo ""

# ── System dependencies ──────────────────────────────────────────────────────

echo "[1/5] Installing system dependencies..."
echo "$PASSWORD" | sudo -S apt-get update -qq 2>/dev/null
echo "$PASSWORD" | sudo -S apt-get install -y -qq \
    ffmpeg \
    portaudio19-dev \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    wget \
    antiword \
    catdoc \
    2>&1 | tail -2
echo "  ✓ System packages installed (incl. antiword/catdoc for .doc files)"
echo ""

# ── Conda environment ────────────────────────────────────────────────────────

echo "[2/5] Setting up conda environment..."

# Check if conda is available
if command -v conda &> /dev/null; then
    CONDA_CMD="conda"
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    CONDA_CMD="$HOME/miniconda3/bin/conda"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
    CONDA_CMD="$HOME/anaconda3/bin/conda"
else
    echo "  ⚠️  Conda not found. Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    CONDA_CMD="$HOME/miniconda3/bin/conda"
    echo "  ✓ Miniconda installed"
fi

# Accept ToS
"$CONDA_CMD" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
"$CONDA_CMD" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# Remove existing environment if present
if "$CONDA_CMD" env list 2>/dev/null | grep -q "$ENV_NAME"; then
    echo "  Removing old $ENV_NAME environment..."
    "$CONDA_CMD" env remove -n "$ENV_NAME" -y -q 2>/dev/null || true
fi

# Create environment (Python 3.10 — compatible with JetPack 6 + modern transformers)
echo "  Creating $ENV_NAME environment (Python 3.10)..."
"$CONDA_CMD" create -n "$ENV_NAME" python=3.10 -y -q 2>&1 | tail -1
echo "  ✓ Conda environment created"
echo ""

CONDA_PIP="$("$CONDA_CMD" run -n "$ENV_NAME" which pip)"

# ── Python dependencies ──────────────────────────────────────────────────────

echo "[3/5] Installing Python dependencies..."

# Install NVIDIA Jetson PyTorch 2.5.0 for JetPack 6 (Python 3.10)
echo "  Installing NVIDIA Jetson PyTorch 2.5.0..."
$CONDA_PIP install "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl" 2>&1 | tail -2

# Install torchvision (matching version for JetPack 6)
echo "  Installing torchvision..."
$CONDA_PIP install torchvision==0.20.1 2>&1 | tail -2 || true

# Install transformers from GitHub (has Qwen3OmniMoe support)
echo "  Installing transformers (from GitHub)..."
$CONDA_PIP install git+https://github.com/huggingface/transformers.git 2>&1 | tail -2

# Install remaining requirements
echo "  Installing remaining packages..."
$CONDA_PIP install -q \
    accelerate \
    qwen-omni-utils \
    soundfile \
    sounddevice \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    aiohttp \
    sqlalchemy \
    duckduckgo-search \
    pydantic \
    pydantic-settings \
    python-dotenv \
    loguru \
    huggingface_hub \
    requests \
    numpy \
    python-docx \
    pypdf \
    sentence-transformers \
    2>&1 | tail -2

echo "  ✓ Python packages installed"
echo ""

# ── Verify installation ──────────────────────────────────────────────────────

echo "[4/5] Verifying installation..."
"$CONDA_CMD" run -n "$ENV_NAME" python -c "
import torch
import transformers
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
print(f'  Transformers: {transformers.__version__}')

# Verify Qwen3OmniMoe is importable
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
print('  ✓ Qwen3OmniMoe classes available')
" 2>&1

echo "  ✓ Installation verified"
echo ""

# ── Model download ───────────────────────────────────────────────────────────

echo "[5/5] Downloading model..."
echo "  Model: ${VALERA_MODEL_NAME_OR_PATH:-cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-8bit}"
echo "  Size: ~42 GB (AWQ 8-bit, high quality)"
echo "  To use the lighter ~27 GB model, set VALERA_MODEL_NAME_OR_PATH in .env"
echo ""
read -p "  Download now? (y/n, recommended: y) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    "$CONDA_CMD" run -n "$ENV_NAME" python download_model.py
    echo "  ✓ Model downloaded"
else
    echo "  ⏭️  Skipped. Run 'python download_model.py' later."
fi

# ── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo "============================================"
echo " Setup Complete!"
echo "============================================"
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To start the server:"
echo "  python main.py"
echo ""
echo "To run the voice client:"
echo "  python client.py --mode voice"
echo ""
echo "API docs: http://localhost:8765/docs"
echo ""
