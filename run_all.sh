#!/bin/bash
# run_all.sh – объединённый запуск WebRAgent (Docker) + Ollama + TTS (TALK.py)

set -e

# Пути к проектам (относительно папки скрипта)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBRAGENT_DIR="$SCRIPT_DIR/WebRAgent"
TTS_DIR="$SCRIPT_DIR/russian_text_to_speech"
CONDA_ENV_NAME="tts_env"

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

# 2. Создание Conda-окружения (если отсутствует)
if ! conda env list | grep -q "^$CONDA_ENV_NAME "; then
    echo "[2/5] Окружение '$CONDA_ENV_NAME' не найдено. Создаём из environment.yml..."
    conda env create -f "$TTS_DIR/environment.yml"
else
    echo "[2/5] Окружение '$CONDA_ENV_NAME' уже существует."
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
echo "Ожидание запуска сервисов (15 секунд)..."
sleep 5


echo ""
echo "[4/5] Активация окружения '$CONDA_ENV_NAME'..."
conda activate "$CONDA_ENV_NAME"

# 5. Запуск TALK.py

echo ""
echo "[5/5] Запуск TALK.py..."
cd "$TTS_DIR"
python TALK.py

# TALK.py отработал — завершаем
cleanup