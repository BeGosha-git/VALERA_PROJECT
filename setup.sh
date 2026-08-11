#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# QWEN-VALERA Setup Script
# Creates conda environment, installs system + Python dependencies, downloads model
# ═══════════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PASSWORD="1055"
ENV_NAME="qwen-valera"
USE_MIRROR=false  # Set to true for faster downloads in some regions

echo "============================================"
echo " QWEN-VALERA Voice Assistant Setup"
echo "============================================"
echo ""

# ── System dependencies ──────────────────────────────────────────────────────

echo "[1/5] Installing system dependencies..."
echo "$PASSWORD" | sudo -S apt-get update -qq
echo "$PASSWORD" | sudo -S apt-get install -y -qq \
    ffmpeg \
    portaudio19-dev \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    2>&1 | tail -1
echo "  ✓ System packages installed"

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
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    CONDA_CMD="$HOME/miniconda3/bin/conda"
    "$CONDA_CMD" init bash
    echo "  ✓ Miniconda installed"
fi

# Remove existing environment if present
if "$CONDA_CMD" env list | grep -q "$ENV_NAME"; then
    echo "  Removing existing $ENV_NAME environment..."
    "$CONDA_CMD" env remove -n "$ENV_NAME" -y -q
fi

# Create environment
echo "  Creating $ENV_NAME environment (Python 3.10)..."
"$CONDA_CMD" create -n "$ENV_NAME" python=3.10 -y -q 2>&1 | tail -1
echo "  ✓ Conda environment created"

# Get conda's pip
CONDA_PIP="$("$CONDA_CMD" run -n "$ENV_NAME" which pip)"

# ── Python dependencies ──────────────────────────────────────────────────────

echo "[3/5] Installing Python dependencies..."

# Install PyTorch with CUDA first
echo "  Installing PyTorch with CUDA 12.4..."
"$CONDA_CMD" run -n "$ENV_NAME" pip install -q \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124 \
    2>&1 | tail -2

# Install flash-attention (pre-built wheel for CUDA 12.4)
echo "  Installing flash-attention..."
"$CONDA_CMD" run -n "$ENV_NAME" pip install -q \
    flash-attn==2.7.4.post1 --no-build-isolation \
    2>&1 | tail -2

# Install transformers from GitHub (required for Qwen3-Omni support)
echo "  Installing transformers (from GitHub)..."
"$CONDA_CMD" run -n "$ENV_NAME" pip install -q \
    git+https://github.com/huggingface/transformers.git \
    2>&1 | tail -2

# Install remaining requirements
echo "  Installing remaining packages..."
"$CONDA_CMD" run -n "$ENV_NAME" pip install -q \
    accelerate \
    qwen-omni-utils \
    soundfile \
    sounddevice \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    aiohttp \
    sqlalchemy \
    chromadb \
    duckduckgo-search \
    pydantic \
    pydantic-settings \
    python-dotenv \
    loguru \
    huggingface_hub \
    requests \
    2>&1 | tail -2

echo "  ✓ Python packages installed"

# ── Verify installation ──────────────────────────────────────────────────────

echo "[4/5] Verifying installation..."

"$CONDA_CMD" run -n "$ENV_NAME" python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.0f} GB')
"

echo "  ✓ Installation verified"

# ── Model download ───────────────────────────────────────────────────────────

echo "[5/5] Downloading model..."
echo "  Model: cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit"
echo "  Size: ~10 GB (AWQ 4-bit quantized)"
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
