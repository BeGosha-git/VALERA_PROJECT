#!/bin/bash
# run_all.sh – объединённый запуск WebRAgent (Docker) + Ollama + TTS
# ============================================================================
#
#   ./run_all.sh [talk|voice|voice-fast] [--stt MODEL] [--keep-ollama] [--preload] [--barge-in]
#
# Режимы:
#   talk          — TALK.py (по умолчанию)
#   voice         — voice_assistant.py (с подтверждением вопроса)
#   voice-fast    — voice_assistant.py (БЕЗ подтверждения, сразу ответ)
#
# Опции (передаются после режима):
#   --stt kairos    — лёгкая модель STT (по умолчанию)
#   --stt nemotron  — NVIDIA Nemotron 3.5 ASR (точнее, но тяжелее)
#   --stt auto      — выбирать модель по длине фразы (короткие → kairos, длинные → nemotron)
#   --keep-ollama   — НЕ перезапускать Ollama (использовать уже запущенную)
#   --new-chat      — начать новый разговор (сбросить контекст)
#   --no-clean      — НЕ чистить текст вопроса (сохранять имена/мат в вопросе)
#   --preload       — прогреть STT-модель при старте (первая фраза быстрее)
#   --barge-in      — прерывать озвучку ответа, если пользователь начал говорить
#
# Важно про Ollama:
#   По умолчанию скрипт ОСТАНАВЛИВАЕТ все процессы `ollama serve`, запущенные
#   вне проекта (чужие), и запускает свою копию. Это гарантирует, что
#   используется нужная модель и настройки проекта.
#   Если вы сами запустили Ollama и хотите её сохранить — добавьте --keep-ollama.
# ============================================================================

set -e

# Пути к проектам (относительно папки скрипта)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBRAGENT_DIR="$SCRIPT_DIR/WebRAgent"
TTS_DIR="$SCRIPT_DIR/russian_text_to_speech"
MIC_DIR="$SCRIPT_DIR/mic_to_text"
CONDA_ENV_NAME="tts_env"
MIC_ENV_NAME="mic_to_text_env"
MODELFILE_PATH="$SCRIPT_DIR/Modelfile"
OLLAMA_MODEL_NAME="valera_live:latest"

# Режим запуска и доп. аргументы
RUN_MODE="${1:-talk}"
EXTRA_ARGS="${*:2}"

# Разбираем флаг --keep-ollama (не перезапускать чужую Ollama)
KEEP_OLLAMA=0
if [[ "$EXTRA_ARGS" == *"--keep-ollama"* ]]; then
    KEEP_OLLAMA=1
    EXTRA_ARGS="${EXTRA_ARGS//--keep-ollama/}"
fi

# PID своей Ollama (запущенной этим скриптом)
OLLAMA_PID=""

# ---------------------------------------------------------------------------
# Управление Ollama
# ---------------------------------------------------------------------------

# Останавливает ВСЕ процессы ollama (serve и др.).
stop_all_ollama() {
    echo ""
    echo "[Ollama] Останавливаю все процессы ollama..."

    # Собираем PID'ы текущего дерева скрипта (себя + потомков),
    # чтобы НЕ убивать собственный процесс и его детей (conda python и т.п.)
    local own_pids
    own_pids=$(pgrep -f "run_all.sh" || true)

    local pids
    pids=$(pgrep -f "ollama" || true)
    if [ -z "$pids" ]; then
        echo "  → Процессов ollama не найдено."
        return
    fi
    for pid in $pids; do
        if echo "$own_pids" | grep -qw "$pid"; then
            continue
        fi
        echo "  → Убиваю PID $pid (ollama)"
        kill "$pid" 2>/dev/null || true
    done
    sleep 2
    # Принудительно добиваем оставшиеся
    pids=$(pgrep -f "ollama" || true)
    if [ -n "$pids" ]; then
        for pid in $pids; do
            if echo "$own_pids" | grep -qw "$pid"; then
                continue
            fi
            echo "  → Принудительно завершаю PID $pid"
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 1
    fi
    echo "  ✓ Все процессы ollama остановлены."
}

# Проверяет, отвечает ли ollama на порту 11434
ollama_is_up() {
    curl -s http://localhost:11434/api/tags > /dev/null 2>&1
}

# Останавливает СВОЮ Ollama (при очистке)
stop_own_ollama() {
    if [ -n "$OLLAMA_PID" ] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "Остановка своей ollama serve (PID $OLLAMA_PID)..."
        kill "$OLLAMA_PID" 2>/dev/null || true
    fi
}

# Создаёт модель проекта (valera_live), если её нет
ensure_model() {
    echo ""
    if ollama list 2>/dev/null | grep -q "$OLLAMA_MODEL_NAME"; then
        echo "[Ollama] Модель $OLLAMA_MODEL_NAME уже установлена."
    else
        echo "[Ollama] Модель $OLLAMA_MODEL_NAME не найдена. Создаю из $MODELFILE_PATH..."
        if [ -f "$MODELFILE_PATH" ]; then
            ollama create "$OLLAMA_MODEL_NAME" -f "$MODELFILE_PATH"
            echo "  ✓ Модель $OLLAMA_MODEL_NAME создана"
        else
            echo "  ⚠️ Modelfile не найден ($MODELFILE_PATH). Модель не создана."
        fi
    fi
}

# ---------------------------------------------------------------------------
# Функция очистки при завершении
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "=== Остановка всех сервисов ==="
    if [ -f "$WEBRAGENT_DIR/docker-compose.yml" ]; then
        docker compose -f "$WEBRAGENT_DIR/docker-compose.yml" down 2>/dev/null || true
    fi
    # Останавливаем СВОЮ ollama (запущенную этим скриптом), чужие — не трогаем
    stop_own_ollama
    echo "Готово."
    exit 0
}

# Перехватываем Ctrl+C
trap cleanup SIGINT SIGTERM

echo "============================================"
echo "  Запуск единого стека: WebRAgent + Ollama + TTS"
echo "  Режим: $RUN_MODE"
echo "  Аргументы: $EXTRA_ARGS"
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

if [ "$RUN_MODE" = "voice" ] || [ "$RUN_MODE" = "voice-fast" ]; then
    create_env_if_missing "$MIC_ENV_NAME" "$MIC_DIR/environment.yml"
fi

# 3. Запуск Ollama (перезапуск, если требуется)
echo ""
echo "[3/5] Настройка Ollama..."

if [ "$KEEP_OLLAMA" = "1" ]; then
    echo "  --keep-ollama: используем уже запущенную Ollama."
    if ollama_is_up; then
        echo "  ✓ Ollama уже отвечает на :11434"
    else
        echo "  ⚠️ Ollama не отвечает, но --keep-ollama указан. Запускаю свою..."
        ollama serve &
        OLLAMA_PID=$!
        echo "  ollama serve запущен (PID $OLLAMA_PID)"
    fi
else
    echo "  Останавливаю чужие/старые процессы ollama для перезапуска проекта..."
    stop_all_ollama

    echo "  Запускаю ollama serve для проекта..."
    ollama serve &
    OLLAMA_PID=$!
    echo "  ollama serve запущен (PID $OLLAMA_PID)"
fi

# Ждём, пока ollama поднимется (до ~30 сек)
echo "  Ожидание запуска Ollama..."
for i in $(seq 1 30); do
    if ollama_is_up; then
        echo "  ✓ Ollama работает."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "  ⚠️ Ollama так и не отвечает на :11434."
    fi
    sleep 1
done

# Гарантируем наличие модели проекта
ensure_model

# 4. Запуск WebRAgent через Docker Compose
echo ""
echo "[4/5] Запуск WebRAgent через Docker Compose..."
cd "$WEBRAGENT_DIR"
docker compose up -d
cd "$SCRIPT_DIR"

# Даём сервисам время подняться
echo "Ожидание запуска сервисов (5 секунд)..."
sleep 5

# 5. Запуск приложения
if [ "$RUN_MODE" = "voice" ]; then
    # --- Режим голосового ассистента (с подтверждением) ---
    echo ""
    echo "[5/5] Активация окружения '$MIC_ENV_NAME' (голосовой ассистент)..."
    conda activate "$MIC_ENV_NAME"

    echo ""
    echo "[5/5] Запуск voice_assistant.py ${EXTRA_ARGS}..."
    cd "$SCRIPT_DIR"
    python voice_assistant.py $EXTRA_ARGS
elif [ "$RUN_MODE" = "voice-fast" ]; then
    # --- Режим голосового ассистента (без подтверждения) ---
    echo ""
    echo "[5/5] Активация окружения '$MIC_ENV_NAME' (голосовой ассистент, fast)..."
    conda activate "$MIC_ENV_NAME"

    echo ""
    echo "[5/5] Запуск voice_assistant.py --no-confirm ${EXTRA_ARGS}..."
    cd "$SCRIPT_DIR"
    python voice_assistant.py --no-confirm $EXTRA_ARGS
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

# Приложение отработало — завершаем
cleanup