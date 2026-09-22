#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# QWEN-VALERA Setup Script — for JetPack 6.x (Jetson AGX Orin)
#
# Запускать ПОСЛЕ обновления Jetson до JetPack 6.x и ПОСЛЕ setup_external_storage.sh.
#
# Все версии зафиксированы в файлах, чтобы установка на другом устройстве
# повторилась один в один:
#   apt-deps.txt              — системные пакеты
#   requirements-jetson6.txt  — точные версии всех Python-пакетов (lock-файл):
#                               torch из репозитория NVIDIA (Jetson, CUDA 12),
#                               transformers 4.57.6 (5.x требует torch>=2.6),
#                               compressed-tensors 0.15.0, nvidia-cusparselt-cu12,
#                               numpy 1.x
#   patch_torch_jetson.py     — патчи совместимости Jetson-сборки torch
#   build_torchvision.sh      — сборка torchvision из исходников
#
# Шаги:
#   1. Ставит системные пакеты (apt-deps.txt)
#   2. Создаёт conda-окружение qwen-valera (Python 3.10) на внешнем диске
#   3. Ставит Python-пакеты из requirements-jetson6.txt
#   4. Применяет patch_torch_jetson.py и проверяет CUDA + transformers
#   5. Скачивает модель AWQ 4-bit
# ═══════════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PASSWORD="1055"
ENV_NAME="qwen-valera"

# ── Внешнее хранилище (диск D) ────────────────────────────────────────────────
# Внутренний eMMC всего 57 GB, поэтому окружение, кэши и модель живут на
# внешнем диске (ext4-образ, смонтированный в /mnt/valera).
VALERA_STORAGE="${VALERA_STORAGE:-/mnt/valera}"

if [ ! -d "$VALERA_STORAGE/conda-envs" ]; then
    echo ""
    echo "❌ Внешнее хранилище не подключено: $VALERA_STORAGE"
    echo ""
    echo "   Сначала выполните один раз:"
    echo "       sudo bash setup_external_storage.sh"
    echo ""
    echo "   Затем повторите установку."
    echo ""
    exit 1
fi

export CONDA_ENVS_PATH="$VALERA_STORAGE/conda-envs"
export CONDA_PKGS_DIRS="$VALERA_STORAGE/conda-pkgs"
export PIP_CACHE_DIR="$VALERA_STORAGE/pip-cache"
export HF_HOME="$VALERA_STORAGE/hf-cache"
export TMPDIR="$VALERA_STORAGE/tmp"
mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$HF_HOME" "$TMPDIR"

# NVIDIA Jetson PyTorch 2.5.0 (CUDA 12) для JetPack 6.x / Python 3.10 прописан
# в requirements-jetson6.txt строкой `torch @ https://developer.download.nvidia.com/...`

echo "============================================"
echo " QWEN-VALERA Setup (JetPack 6.x)"
echo "============================================"
echo ""
echo "  Хранилище: $VALERA_STORAGE"
df -h "$VALERA_STORAGE" | tail -1 | awk '{print "  Свободно:  "$4" из "$2}'
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
APT_DEPS="$SCRIPT_DIR/apt-deps.txt"
if [ ! -f "$APT_DEPS" ]; then
    echo "❌ Не найден $APT_DEPS"
    exit 1
fi
APT_PKGS="$(grep -vE '^\s*#|^\s*$' "$APT_DEPS" | tr '\n' ' ')"
echo "$PASSWORD" | sudo -S apt-get update -qq 2>/dev/null
# shellcheck disable=SC2086
echo "$PASSWORD" | sudo -S apt-get install -y -qq $APT_PKGS 2>&1 | tail -2
echo "  ✓ System packages installed (список: apt-deps.txt)"
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
# ⚠️ НЕ ставить torchvision из PyPI: он требует ровно torch==2.5.1 и затирает
#    Jetson-сборку CPU-версией, после чего CUDA пропадает. Проект torchvision
#    не использует, поэтому он здесь не устанавливается.
echo "  Installing NVIDIA Jetson PyTorch 2.5.0 (CUDA 12)..."

# Все версии зафиксированы в requirements-jetson6.txt (снято с рабочего окружения).
# Там же torch из репозитория NVIDIA (PyPI-сборка для aarch64 — БЕЗ CUDA) и
# transformers на конкретном коммите GitHub, а также nvidia-cusparselt-cu12.
REQ_FILE="$SCRIPT_DIR/requirements-jetson6.txt"
if [ ! -f "$REQ_FILE" ]; then
    echo "❌ Не найден $REQ_FILE"
    exit 1
fi

echo "  Installing all Python packages from requirements-jetson6.txt (torch ~807 MB)..."
$CONDA_PIP install -r "$REQ_FILE" 2>&1 | tail -3

# cuSPARSELt: Jetson-сборка torch слинкована с libcusparseLt.so.0, которой нет
# ни в системе, ни в репозиториях Jetson. Пакет nvidia-cusparselt-cu12 приходит
# из requirements, но загрузчик ищет .so рядом с libtorch_cuda.so
# (RUNPATH = $ORIGIN), а не в site-packages/nvidia/... → делаем симлинки.
CONDA_PREFIX_PATH="$("$CONDA_CMD" run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)')"
TORCH_LIB="$CONDA_PREFIX_PATH/lib/python3.10/site-packages/torch/lib"
if [ -d "$TORCH_LIB" ]; then
    for f in "$CONDA_PREFIX_PATH"/lib/python*/site-packages/nvidia/cusparselt/lib/libcusparseLt*.so*; do
        [ -e "$f" ] || continue
        ln -sf "$f" "$TORCH_LIB/$(basename "$f")"
    done
    echo "  ✓ cuSPARSELt слинкована в torch/lib"
fi

# Патчи совместимости Jetson-сборки torch — БЕЗ НИХ transformers не увидит PyTorch.
echo "  Applying Jetson compatibility patches (patch_torch_jetson.py)..."
"$CONDA_CMD" run -n "$ENV_NAME" python "$SCRIPT_DIR/patch_torch_jetson.py" 2>&1 | tail -12

# Ускорение загрузки AWQ-модели: у Qwen3-Omni ~11 500 записей в
# quantization_config.ignore, и compressed-tensors перебирал их для каждого
# модуля модели — загрузка «зависала» на часы. Без этого патча модель не грузится.
echo "  ✓ Python packages installed"
echo ""

# ── torchvision из исходников ────────────────────────────────────────────────
# Qwen3-Omni — omni-модель: её процессор (Qwen3OmniMoeProcessor) поднимает
# Qwen2VLImageProcessor и Qwen2VLVideoProcessor, а они требуют torchvision.
# Готовых колёс под Jetson нет, а сборка с PyPI несовместима по ABI с
# Jetson-сборкой torch → собираем из исходников (15-40 минут, один раз).
echo "[3.5/5] torchvision из исходников (нужен процессору Qwen3-Omni)..."
bash "$SCRIPT_DIR/build_torchvision.sh"
echo ""

# ── Verify installation ──────────────────────────────────────────────────────

echo "[4/5] Verifying installation..."
"$CONDA_CMD" run -n "$ENV_NAME" python -c "
import sys
import numpy as np
import torch
import transformers
from transformers.utils import is_torch_available

print(f'  numpy          : {np.__version__}')
print(f'  torch          : {torch.__version__}')
print(f'  CUDA build     : {torch.version.cuda}')
print(f'  CUDA available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    print(f'  GPU            : {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]})')
print(f'  transformers   : {transformers.__version__}')
print(f'  видит PyTorch  : {is_torch_available()}')

# Классы модели
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
print('  ✓ классы Qwen3OmniMoe доступны')

# Модули проекта (проверяет chromadb, pydub, fastapi и т.д.)
from api.routes import router
print('  ✓ модули проекта импортируются')

errors = []
if not torch.cuda.is_available():
    errors.append('CUDA недоступна — похоже, поставилась CPU-сборка torch с PyPI')
if np.__version__.split('.')[0] != '1':
    errors.append(f'numpy {np.__version__}: нужна версия 1.x (Jetson-torch собран под numpy 1.x)')
if not is_torch_available():
    errors.append('transformers не видит PyTorch — не применён patch_torch_jetson.py')
if errors:
    print('')
    for e in errors:
        print(f'  ❌ {e}')
    print('')
    print('  Запустите:  python patch_torch_jetson.py')
    print('  Подробности в README, раздел «Частые проблемы».')
    sys.exit(1)
" 2>&1

echo "  ✓ Installation verified (CUDA + transformers OK)"
echo ""

# ── Model download ───────────────────────────────────────────────────────────

echo "[5/5] Downloading model..."
# Модель задаётся в .env (VALERA_MODEL_NAME_OR_PATH). По умолчанию — модель без
# квантования, которая запускается «из коробки» и влезает в 61 GB общей памяти.
VALERA_MODEL="${VALERA_MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-Omni-7B}"
export VALERA_MODEL_NAME_OR_PATH="$VALERA_MODEL"
echo "  Model: $VALERA_MODEL"
echo "  Сохраняется на внешний диск:  $VALERA_STORAGE/models"
echo "  Свободно на диске D:          $(df -h "$VALERA_STORAGE" | tail -1 | awk '{print $4}')"
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
