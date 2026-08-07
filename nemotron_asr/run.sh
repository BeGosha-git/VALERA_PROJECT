#!/usr/bin/env bash
# ============================================================
# Nemotron 3.5 ASR Streaming — Launcher
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Полный путь к окружению (в ~/.condarc envs_dirs относительный)
ENV_NAME="nemotron_asr_env"
ENV_PATH="/home/georgiy/Desktop/VALERA_PROJECT/miniconda3/envs/${ENV_NAME}"

# Проверяем, существует ли окружение
if [ ! -d "${ENV_PATH}" ]; then
    echo "❌ Окружение '${ENV_NAME}' не найдено."
    echo "   Сначала запустите:  bash setup_conda.sh"
    exit 1
fi

# Аргументы по умолчанию
LANGUAGE="${1:-ru-RU}"
LATENCY="${2:-320}"

echo "=============================================="
echo " Nemotron 3.5 ASR Streaming"
echo " Язык:     ${LANGUAGE}"
echo " Задержка: ${LATENCY} мс"
echo "=============================================="

cd "${SCRIPT_DIR}"

conda run -p "${ENV_PATH}" python nemotron_streaming_stt.py \
    --language "${LANGUAGE}" \
    --latency "${LATENCY}" \
    "${@:3}"
