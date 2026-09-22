#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Сборка torchvision из исходников под Jetson-сборку PyTorch.
#
# ── Зачем это нужно ───────────────────────────────────────────────────────────
# Готовых torchvision-колёс для Jetson нет (в репозитории NVIDIA лежит только
# torch). Сборка torchvision с PyPI несовместима по ABI с Jetson-сборкой torch:
#
#     ImportError: _C.so: undefined symbol: _ZNK5torch8autograd4Node4nameEv
#
# поэтому torchvision приходится собирать из исходников против того самого
# torch, который стоит в окружении.
#
# ── Кому это нужно ────────────────────────────────────────────────────────────
# Qwen3-Omni — omni-модель: её процессор (Qwen3OmniMoeProcessor) поднимает
# Qwen2VLImageProcessor и Qwen2VLVideoProcessor, а те требуют torchvision.
# Без него старт сервера падает:
#
#     Failed to load model: Qwen2VLVideoProcessor requires the Torchvision
#     library but it was not found in your environment.
#
# Для голосового режима (audio + text) функциональность torchvision не
# используется — нужен только сам импорт.
#
# ── Запуск ────────────────────────────────────────────────────────────────────
#     bash build_torchvision.sh
#
# Переменные:
#     VALERA_TORCHVISION_VERSION=0.20.0   # версия (0.20.0 ↔ torch 2.5.0)
#     VALERA_BUILD_DIR=/mnt/valera/src    # где собирать
#     VALERA_JOBS=8                       # параллельных задач компиляции
#
# ⏱ Занимает примерно 15–40 минут на Jetson AGX Orin. Запускайте один раз.
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${VALERA_ENV_NAME:-qwen-valera}"
TV_VERSION="${VALERA_TORCHVISION_VERSION:-0.20.0}"
BUILD_DIR="${VALERA_BUILD_DIR:-/mnt/valera/src}"
JOBS="${VALERA_JOBS:-8}"

# ── Внешнее хранилище (если подключено) ──────────────────────────────────────
if [ -f /etc/profile.d/valera-storage.sh ]; then
    # shellcheck disable=SC1091
    . /etc/profile.d/valera-storage.sh
fi

# ── Ищем окружение ───────────────────────────────────────────────────────────
if command -v conda &> /dev/null; then
    CONDA_CMD="conda"
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    CONDA_CMD="$HOME/miniconda3/bin/conda"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
    CONDA_CMD="$HOME/anaconda3/bin/conda"
else
    echo "❌ conda не найден"
    exit 1
fi

CONDA_PREFIX_PATH="$("$CONDA_CMD" run -n "$ENV_NAME" python -c 'import sys; print(sys.prefix)' 2>/dev/null || true)"
if [ -z "$CONDA_PREFIX_PATH" ]; then
    echo "❌ Окружение '$ENV_NAME' не найдено. Сначала запустите setup_jetson6.sh"
    exit 1
fi

PIP="$CONDA_PREFIX_PATH/bin/pip"
PYTHON="$CONDA_PREFIX_PATH/bin/python"

echo "============================================"
echo " Сборка torchvision $TV_VERSION из исходников"
echo "============================================"
echo "  Окружение:  $CONDA_PREFIX_PATH"
echo "  Каталог:    $BUILD_DIR"
echo "  Параллельно: $JOBS задач"
echo "  ⏱  15-40 минут"
echo ""

# ── Проверки ─────────────────────────────────────────────────────────────────
TORCH_INFO="$("$PYTHON" -c "
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
" 2>&1 || true)"
echo "  torch: $TORCH_INFO"

if ! echo "$TORCH_INFO" | grep -q "True"; then
    echo "❌ CUDA в torch недоступна — сначала исправьте PyTorch"
    echo "   (см. README, раздел «Частые проблемы»)"
    exit 1
fi

if "$PYTHON" -c "import torchvision" 2>/dev/null; then
    echo "✓ torchvision уже установлен: $("$PYTHON" -c 'import torchvision; print(torchvision.__version__)')"
    echo "  Пересобрать: удалите его (pip uninstall -y torchvision) и запустите скрипт снова."
    exit 0
fi

if ! command -v git &> /dev/null; then
    echo "❌ git не найден (нужен пакет git, он есть в apt-deps.txt)"
    exit 1
fi

# ── Инструменты сборки (без sudo, всё есть на PyPI) ──────────────────────────
echo ""
echo "[1/4] Инструменты сборки..."
# setuptools < 81 обязателен: в новых версиях удалён pkg_resources,
# а setup.py torchvision его использует.
"$PIP" install -q "setuptools<81" wheel ninja cmake
echo "  ✓ setuptools / wheel / ninja / cmake"

# ── Исходники ────────────────────────────────────────────────────────────────
echo ""
echo "[2/4] Скачиваю исходники torchvision v$TV_VERSION..."
mkdir -p "$BUILD_DIR"
rm -rf "$BUILD_DIR/vision"
git clone --branch "v$TV_VERSION" --depth 1 https://github.com/pytorch/vision.git "$BUILD_DIR/vision"
echo "  ✓ $BUILD_DIR/vision"

# ── Сборка ───────────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Сборка (это долго, ~15-40 минут)..."
cd "$BUILD_DIR/vision"
MAX_JOBS="$JOBS" BUILD_VERSION="$TV_VERSION" \
    "$PIP" install --no-deps --no-build-isolation --no-cache-dir .
echo "  ✓ собрано и установлено"

# ── Проверка ─────────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Проверка..."
"$PYTHON" -c "
import torch, torchvision
print(f'  torch         : {torch.__version__}')
print(f'  cuda          : {torch.cuda.is_available()}')
print(f'  torchvision   : {torchvision.__version__}')
from torchvision.transforms import functional
print('  ✓ torchvision работает')
if not torch.cuda.is_available():
    raise SystemExit('  ❌ CUDA пропала — проверьте, не подменился ли torch')
"
echo ""
echo "============================================"
echo " Готово. Теперь должен стартовать сервер:"
echo "     python main.py"
echo "============================================"
