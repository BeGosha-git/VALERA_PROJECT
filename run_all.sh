#!/bin/bash
# run_all.sh – объединённый запуск WebRAgent (Docker) + Ollama + TTS

set -e

# Пути к проектам (относительно папки скрипта)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBRAGENT_DIR="$SCRIPT_DIR/WebRAgent"
TTS_DIR="$SCRIPT_DIR/russian_text_to_speech"
MIC_DIR="$SCRIPT_DIR/mic_to_text"
CONDA_ENV_NAME="tts_env"
MIC_ENV_NAME="mic_to_text_env"

# Что запускать: "talk" (TALK.py) или "voice" (voice_assistant.py)
# Можно передать аргументом: ./run_all.sh voice [--no-confirm]
#   voice         — голосовой ассистент с подтверждением
#   voice-fast    — голосовой ассистент БЕЗ подтверждения (сразу ответ)
#   talk          — TALK.py (по умолчанию)
RUN_MODE="${1:-talk}"
EXTRA_ARGS=""

# Функция очистки при завершении
cleanup() {
    echo ""
    echo "=== Остановка всех сервисов ==="
    if [ -f "$WEBRAGENT_DIR/docker-compose.yml" ]; then
        docker compose -f "$WEBRAGENT_DIR/docker-compose.yml" down 2>/dev/null || true
    fi
    # Останавливаем глобальный ollama serve (если запущен этим скриптом)
    if [ -n "$OLLAMA_PID" ] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "Остановка ollama serve (PID $OLLAMA_PID)..."
        kill "$OLLAMA_PID" 2>/dev/null || true
    fi
    echo "Готово."
    exit 0
}

# Перехватываем Ctrl+C
trap cleanup SIGINT SIGTERM

echo "============================================"
echo "  Запуск единого стека: WebRAgent + Ollama + TTS"
echo "============================================"

# 1. Инициализация conda
echo ""
echo "[1/5] Инициализация conda..."
source "$(conda info --base)/etc/profile.d/conda.sh"

# 2. Создание Conda-окружений (если отсутствуют)
echo ""
echo "[2/5] Проверка Conda-окружений..."

create_env_if_missing() {
    local env_name="$1"
    local env_file="$2"
    if ! conda env list | grep -q "^$env_name "; then
        echo "  Окружение '$env_name' не найдено. Создаём из $env_file..."
        conda env create -f "$env_file"
    else
        echo "  Окружение '$env_name' уже существует."
    fi
}

create_env_if_missing "$CONDA_ENV_NAME" "$TTS_DIR/environment.yml"

if [ "$RUN_MODE" = "voice" ]; then
    create_env_if_missing "$MIC_ENV_NAME" "$MIC_DIR/environment.yml"
fi

# 3. Запуск Ollama глобально (вне conda-окружения)
echo ""
echo "[3/5] Запуск ollama serve (глобально)..."
if pgrep -f "ollama serve" > /dev/null 2>&1; then
    echo "Ollama уже запущен."
    OLLAMA_PID=""
else
    ollama serve &
    OLLAMA_PID=$!
    echo "ollama serve запущен (PID $OLLAMA_PID)."
fi

# 4. Активация окружения + Docker Compose

echo "[4/5] Запуск WebRAgent через Docker Compose..."
cd "$WEBRAGENT_DIR"
docker compose up -d
cd "$SCRIPT_DIR"

# Проверяем, что Ollama отвечает
echo "Проверка Ollama (http://localhost:11434)..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama работает."
else
    echo "ПРЕДУПРЕЖДЕНИЕ: Ollama не отвечает. Возможно, ещё запускается."
fi

# Даём сервисам время подняться
echo "Ожидание запуска сервисов (5 секунд)..."
sleep 5

if [ "$RUN_MODE" = "voice" ]; then
    # --- Режим голосового ассистента (с подтверждением) ---
    echo ""
    echo "[4/5] Активация окружения '$MIC_ENV_NAME' (голосовой ассистент)..."
    conda activate "$MIC_ENV_NAME"

    echo ""
    echo "[5/5] Запуск voice_assistant.py..."
    cd "$SCRIPT_DIR"
    python voice_assistant.py
elif [ "$RUN_MODE" = "voice-fast" ]; then
    # --- Режим голосового ассистента (без подтверждения) ---
    echo ""
    echo "[4/5] Активация окружения '$MIC_ENV_NAME' (голосовой ассистент, fast)..."
    conda activate "$MIC_ENV_NAME"

    echo ""
    echo "[5/5] Запуск voice_assistant.py --no-confirm..."
    cd "$SCRIPT_DIR"
    python voice_assistant.py --no-confirm
else
    # --- Режим TALK.py (по умолчанию) ---
    echo ""
    echo "[4/5] Активация окружения '$CONDA_ENV_NAME'..."
    conda activate "$CONDA_ENV_NAME"

    echo ""
    echo "[5/5] Запуск TALK.py..."
    cd "$TTS_DIR"
    python TALK.py
fi

# TALK.py отработал — завершаем
cleanup