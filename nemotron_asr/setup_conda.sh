#!/usr/bin/env bash
# ============================================================
# Setup conda environment for Nemotron 3.5 ASR Streaming
# ============================================================
set -euo pipefail

ENV_NAME="nemotron_asr_env"
PYTHON_VER="3.11"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${PROJECT_DIR}/../nemotron-project"
# ВАЖНО: ~/.condarc задаёт относительный envs_dirs (miniconda3/envs/),
# поэтому используем полный путь явно, чтобы окружение создалось
# рядом с mic_to_text_env / nemotron_env / tts_env.
ENV_PATH="/home/georgiy/Desktop/VALERA_PROJECT/miniconda3/envs/${ENV_NAME}"

echo "=============================================="
echo " Setting up Nemotron ASR Streaming Environment"
echo "=============================================="
echo " Environment : ${ENV_NAME}"
echo " Python      : ${PYTHON_VER}"
echo " Project dir : ${PROJECT_DIR}"
echo " Model dir   : ${MODEL_DIR}"
echo "=============================================="

# 1. Create conda environment
echo ""
echo "[1/4] Creating conda environment '${ENV_NAME}'..."
if [ -d "${ENV_PATH}" ]; then
    echo "  → Environment '${ENV_NAME}' already exists, skipping creation."
else
    conda create -y -p "${ENV_PATH}" python=${PYTHON_VER}
    echo "  ✓ Environment created."
fi

# 2. Install PyTorch with CUDA
echo ""
echo "[2/4] Installing PyTorch with CUDA support..."
# ВАЖНО: RTX 50xx (Blackwell, sm_120) требует PyTorch cu128+!
# cu124 НЕ поддерживает эти GPU. НЕ меняйте на cu124.
conda run -p "${ENV_PATH}" pip install --upgrade pip
conda run -p "${ENV_PATH}" pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "  ✓ PyTorch installed."

# 3. Install project dependencies
echo ""
echo "[3/4] Installing project dependencies..."
conda run -p "${ENV_PATH}" pip install -r "${PROJECT_DIR}/requirements.txt"

echo "  ✓ Dependencies installed."

# 4. Verify installation
echo ""
echo "[4/4] Verifying installation..."
conda run -p "${ENV_PATH}" python -c "
import torch; print(f'PyTorch {torch.__version__}');
print(f'CUDA available: {torch.cuda.is_available()}');
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}');
import transformers; print(f'Transformers {transformers.__version__}');
import sounddevice; print(f'sounddevice OK');
import librosa; print(f'librosa OK');
print('All imports successful! ✅')
"

echo ""
echo "=============================================="
echo " Setup complete!"
echo ""
echo " Activate environment:"
echo "   conda activate ${ENV_PATH}"
echo ""
echo " Run streaming ASR:"
echo "   cd ${PROJECT_DIR}"
echo "   python nemotron_streaming_stt.py"
echo ""
echo " Or use the launcher:"
echo "   bash run.sh"
echo "=============================================="
