#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Пересобирает requirements-jetson6.txt из РЕАЛЬНО установленного окружения.
#
# Запускать после того, как вы поставили/обновили пакеты и всё проверили:
#     bash freeze_env.sh
#
# Шапка-комментарий файла сохраняется, список пакетов заменяется на свежий
# `pip freeze` (с сортировкой).
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_FILE="$SCRIPT_DIR/requirements-jetson6.txt"
ENV_NAME="${VALERA_ENV_NAME:-qwen-valera}"

# ── Находим conda ────────────────────────────────────────────────────────────
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

# ── Переменные окружения внешнего диска (если подключён) ─────────────────────
if [ -f /etc/profile.d/valera-storage.sh ]; then
    # shellcheck disable=SC1091
    . /etc/profile.d/valera-storage.sh
fi

if ! "$CONDA_CMD" env list 2>/dev/null | grep -q "$ENV_NAME"; then
    echo "❌ Окружение '$ENV_NAME' не найдено"
    exit 1
fi

echo "⏳ Снимаю список пакетов из окружения '$ENV_NAME'..."
TMP_FILE="$(mktemp)"
trap 'rm -f "$TMP_FILE"' EXIT

# 1. Сохраняем шапку файла: все строки-комментарии и пустые до первого пакета
if [ -f "$REQ_FILE" ]; then
    awk '/^[[:space:]]*#/ || /^[[:space:]]*$/ { print; next } { exit }' "$REQ_FILE" > "$TMP_FILE"
else
    echo "# QWEN-VALERA — lock-файл окружения (создан freeze_env.sh)" > "$TMP_FILE"
fi

# 2. Свежий список пакетов.
#    torchvision ИСКЛЮЧАЕМ: его нельзя ставить из PyPI (колесо собрано против
#    другой сборки torch), он собирается из исходников через build_torchvision.sh.
"$CONDA_CMD" run -n "$ENV_NAME" pip freeze 2>/dev/null \
    | grep -v '^torchvision' | sort >> "$TMP_FILE"

cat >> "$TMP_FILE" <<'TV_NOTE'

# ── torchvision ───────────────────────────────────────────────────────────────
# Здесь его нет НАМЕРЕННО. Колесо с PyPI несовместимо с Jetson-сборкой torch
# (undefined symbol: _ZNK5torch8autograd4Node4nameEv), а с зависимостями оно
# ещё и затирает Jetson-сборку torch (CUDA пропадёт).
# Собирается из исходников отдельно:  bash build_torchvision.sh
TV_NOTE

# 3. Предупреждения о типичных граблях
if grep -qi '^torch==' "$TMP_FILE"; then
    echo "⚠️  Внимание: torch взят с PyPI (строка 'torch==...'), а не из репозитория NVIDIA!"
    echo "    Для Jetson нужна строка вида:"
    echo "      torch @ https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
fi
if ! grep -q '^numpy==1\.' "$TMP_FILE"; then
    echo "⚠️  Внимание: numpy не 1.x — Jetson-сборка torch с ним не работает."
fi

mv "$TMP_FILE" "$REQ_FILE"
trap - EXIT
echo "✓ Обновлено: $REQ_FILE ($(grep -cvE '^[[:space:]]*(#|$)' "$REQ_FILE") пакетов)"
