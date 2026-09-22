#!/bin/bash
# setup_conda.sh – создание Conda-окружения для russian_text_to_speech

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TTS_DIR="$SCRIPT_DIR/russian_text_to_speech"
ENV_FILE="$TTS_DIR/environment.yml"
CONDA_ENV_NAME="tts_env"

echo "=== Настройка Conda-окружения для TTS ==="

# Проверяем, установлен ли conda
if ! command -v conda &> /dev/null; then
    echo "ОШИБКА: conda не найден. Установите Miniconda или Anaconda."
    exit 1
fi

# Инициализируем conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Проверяем, существует ли окружение
if conda env list | grep -q "^$CONDA_ENV_NAME "; then
    echo "Окружение '$CONDA_ENV_NAME' уже существует."
    read -p "Пересоздать? [y/N]: " answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        echo "Удаляю существующее окружение..."
        conda env remove -n "$CONDA_ENV_NAME" -y
    else
        echo "Пропускаю создание окружения."
        exit 0
    fi
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "ОШИБКА: Файл $ENV_FILE не найден!"
    exit 1
fi

echo "Создаю окружение '$CONDA_ENV_NAME' из $ENV_FILE..."
conda env create -f "$ENV_FILE"

echo ""
echo "=== Готово! ==="
echo "Активируйте окружение командой:"
echo "  conda activate $CONDA_ENV_NAME"
