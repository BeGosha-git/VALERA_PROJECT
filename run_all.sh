#!/bin/bash
# run_all.sh — единый запуск стека на Jetson (без Docker и без Ollama).
#
# Поднимает два процесса:
#   1. Наш FastAPI-сервер (python main.py) — держит Qwen2.5-Omni-7B в памяти GPU
#      и отдаёт чистую генерацию на /api/v1/llm/generate.
#   2. WebRAgent (Flask) — веб-интерфейс RAG, который по HTTP обращается к (1).
#
# Использование:
#   bash run_all.sh              # запустить всё (без клиента)
#   bash run_all.sh --client     # + голосовой клиент (слушай → отвечай голосом)
#   bash run_all.sh --no-web     # только модель (API)
#   bash run_all.sh --no-model   # только WebRAgent (API уже запущен отдельно)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBRAGENT_DIR="$SCRIPT_DIR/WebRAgent"

# --- Окружение ---------------------------------------------------------------
# Используем интерпретатор из внешнего хранилища (диск D).
PY="/mnt/valera/conda-envs/qwen-valera/bin/python"
if [ ! -x "$PY" ]; then
    echo "⚠️  Не найден python окружения qwen-valera в /mnt/valera."
    echo "   Смонтируйте хранилище:  sudo valera-storage.sh up"
    echo "   или запустите установку:  bash setup_jetson6.sh"
    exit 1
fi

API_BASE="${VALERA_API_BASE:-http://localhost:8765/api/v1}"
export VALERA_API_BASE="$API_BASE"
export VALERA_TIMEOUT="${VALERA_TIMEOUT:-600}"
export VALERA_MODEL_NAME_OR_PATH="${VALERA_MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-Omni-7B}"

# Qdrant без Docker: локальное хранилище на диске
export QDRANT_LOCAL_PATH="${QDRANT_LOCAL_PATH:-$WEBRAGENT_DIR/data/qdrant_storage}"

export FLASK_SECRET_KEY="${FLASK_SECRET_KEY:-valera-local-dev-key}"
export ADMIN_USERNAME="${ADMIN_USERNAME:-admin}"
export ADMIN_PASSWORD="${ADMIN_PASSWORD:-admin}"
export MONGODB_URI="${MONGODB_URI:-mongodb://localhost:27017}"
export MONGODB_DB="${MONGODB_DB:-ragapp}"

RUN_MODEL=1
RUN_WEB=1
RUN_CLIENT=0
for arg in "$@"; do
    case "$arg" in
        --no-web)   RUN_WEB=0 ;;
        --no-model) RUN_MODEL=0 ;;
        --client)   RUN_CLIENT=1 ;;
        -h|--help)  sed -n '2,12p' "$0"; exit 0 ;;
    esac
done

MODEL_PID=""

cleanup() {
    echo ""
    if [ -n "$MODEL_PID" ] && kill -0 "$MODEL_PID" 2>/dev/null; then
        echo "Останавливаю модель (PID $MODEL_PID)..."
        kill "$MODEL_PID" 2>/dev/null || true
    fi
    echo "Готово."
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "============================================"
echo "  VALERA + WebRAgent (локальная Qwen2.5-Omni)"
echo "============================================"

# --- 1. Модель ---------------------------------------------------------------
model_ready() {
    curl -sf --max-time 3 "$API_BASE/health" 2>/dev/null | grep -q '"model_loaded":true'
}

if [ "$RUN_MODEL" = "1" ]; then
    if model_ready; then
        echo "[1/2] Модель уже запущена на $API_BASE — пропускаю."
    else
        echo "[1/2] Запускаю модель: $PY main.py"
        cd "$SCRIPT_DIR"
        setsid nohup "$PY" main.py > /mnt/valera/tmp/server.log 2>&1 < /dev/null &
        MODEL_PID=$!
        printf "      Ожидание загрузки модели"
        for _ in $(seq 1 90); do
            sleep 5
            if model_ready; then echo " — готова."; break; fi
            printf "."
        done
        if ! model_ready; then
            echo ""
            echo "❌ Модель не поднялась. Смотрите лог: /mnt/valera/tmp/server.log"
            exit 1
        fi
    fi
else
    echo "[1/2] Запуск модели пропущен (--no-model)."
fi

# --- 2. WebRAgent ------------------------------------------------------------
if [ "$RUN_WEB" = "1" ]; then
    echo "[2/3] Запускаю WebRAgent (Flask) на http://127.0.0.1:5000"
    cd "$WEBRAGENT_DIR"
    [ -f .env ] || cp .env.example .env
    if [ "$RUN_CLIENT" = "1" ]; then
        setsid nohup "$PY" run.py > /tmp/webagent.log 2>&1 < /dev/null &
        WEB_PID=$!
        sleep 5
    else
        exec "$PY" run.py
    fi
else
    echo "[2/3] Запуск WebRAgent пропущен (--no-web)."
fi

# --- 3. Голосовой клиент -----------------------------------------------------
if [ "$RUN_CLIENT" = "1" ]; then
    cd "$SCRIPT_DIR"
    echo "[3/3] Запускаю голосовой клиент (Ctrl+C — выход)"
    exec "$PY" client.py --mode voice
fi
