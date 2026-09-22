#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
#  run_all.sh — запуск и остановка всего стека VALERA одной командой
# ══════════════════════════════════════════════════════════════════════════════
#
#   bash run_all.sh              запустить всё (модель + WebRAgent)
#   bash run_all.sh --client     то же + голосовой клиент (слушай → отвечай)
#   bash run_all.sh --down       остановить всё
#   bash run_all.sh --status     показать, что запущено
#   bash run_all.sh --no-web     только модель (API :8765)
#   bash run_all.sh --no-model   только WebRAgent (модель уже запущена)
#   bash run_all.sh --tts model  озвучка встроенным голосом Qwen (по умолч. Silero)
#
#  Ctrl+C останавливает всё, что запустил скрипт.
# ══════════════════════════════════════════════════════════════════════════════

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBRAGENT_DIR="$SCRIPT_DIR/WebRAgent"

# --- Окружение ---------------------------------------------------------------
PY="/mnt/valera/conda-envs/qwen-valera/bin/python"
if [ ! -x "$PY" ]; then
    echo "⚠️  Не найден python окружения qwen-valera в /mnt/valera."
    echo "   Смонтируйте хранилище:  sudo valera-storage.sh up"
    exit 1
fi

API_BASE="${VALERA_API_BASE:-http://localhost:8765/api/v1}"
WEB_HOST="${WEB_HOST:-127.0.0.1}"
WEB_PORT="${WEB_PORT:-5000}"
LOG_DIR="/mnt/valera/tmp"
LOG_MODEL="$LOG_DIR/server.log"
LOG_WEB="$LOG_DIR/webagent.log"
mkdir -p "$LOG_DIR" 2>/dev/null || LOG_MODEL=/tmp/server.log
[ -d "$LOG_DIR" ] || LOG_WEB=/tmp/webagent.log

export VALERA_API_BASE="$API_BASE"
export VALERA_TIMEOUT="${VALERA_TIMEOUT:-600}"
export VALERA_MODEL_NAME_OR_PATH="${VALERA_MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-Omni-7B}"
export QDRANT_LOCAL_PATH="${QDRANT_LOCAL_PATH:-$WEBRAGENT_DIR/data/qdrant_storage}"
export EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}"
export FLASK_SECRET_KEY="${FLASK_SECRET_KEY:-valera-local-dev-key}"
export ADMIN_USERNAME="${ADMIN_USERNAME:-admin}"
export ADMIN_PASSWORD="${ADMIN_PASSWORD:-admin}"
export MONGODB_URI="${MONGODB_URI:-mongodb://localhost:27017}"
export MONGODB_DB="${MONGODB_DB:-ragapp}"

# --- Разбор аргументов -------------------------------------------------------
RUN_MODEL=1
RUN_WEB=1
RUN_CLIENT=0
ACTION="start"
TTS_FLAG=""

for arg in "$@"; do
    case "$arg" in
        --down|--stop|-d) ACTION="down" ;;
        --status|-s)      ACTION="status" ;;
        --no-web)         RUN_WEB=0 ;;
        --no-model)       RUN_MODEL=0 ;;
        --client)         RUN_CLIENT=1 ;;
        --tts)            TTS_FLAG="russian_tts" ;;
        --tts=*)          TTS_FLAG="${arg#--tts=}" ;;
        -h|--help)        sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "Неизвестный аргумент: $arg (см. --help)"; exit 2 ;;
    esac
done

# --- Определение процессов ---------------------------------------------------
model_pids()  { pgrep -f "bin/python main\.py" 2>/dev/null; }
web_pids()    { pgrep -f "create_app\(\)\.run" 2>/dev/null; }
client_pids() { pgrep -f "client\.py --mode" 2>/dev/null; }

port_busy() {  # $1 = порт
    (ss -lptn 2>/dev/null || netstat -lptn 2>/dev/null) | grep -q ":$1 "
}

model_ready() {
    curl -sf --max-time 3 "$API_BASE/health" 2>/dev/null | grep -q '"model_loaded":true'
}

web_ready() {
    curl -s -o /dev/null --max-time 3 "http://$WEB_HOST:$WEB_PORT/auth/login" 2>/dev/null
}

# --- Остановка ---------------------------------------------------------------
kill_group() {  # $1 = список PID, $2 = имя
    local pids="$1" name="$2"
    [ -z "$pids" ] && return 0
    echo "   ⏹  Останавливаю $name (PID: $(echo "$pids" | tr '\n' ' '))"
    kill $pids 2>/dev/null || true
    for _ in $(seq 1 30); do                       # до 15 секунд на «мягко»
        sleep 0.5
        pids=$(echo "$pids" | tr ' ' '\n' | while read -r p; do
            [ -n "$p" ] && kill -0 "$p" 2>/dev/null && echo "$p"
        done | tr '\n' ' ')
        [ -z "$pids" ] && return 0
    done
    echo "      (не ответил — завершаю принудительно)"
    kill -9 $pids 2>/dev/null || true
    sleep 1
}

stop_stack() {
    echo "🛑 Останавливаю стек VALERA…"
    kill_group "$(client_pids)" "голосовой клиент"
    # run.py — старый способ запуска WebRAgent
    kill_group "$(pgrep -f 'WebRAgent/run\.py' 2>/dev/null)" "WebRAgent (run.py)"
    kill_group "$(web_pids)" "WebRAgent"
    kill_group "$(model_pids)" "модель (main.py)"

    # Подчистить порты, если что-то осталось
    for p in $(pgrep -f 'bin/python main\.py' 2>/dev/null); do kill -9 "$p" 2>/dev/null; done

    if port_busy 8765 || port_busy "$WEB_PORT"; then
        echo "   ⚠️  Порты всё ещё заняты:"
        (ss -lptn 2>/dev/null | grep -E ":(8765|$WEB_PORT) ")
        return 1
    fi
    echo "   ✓ Всё остановлено, порты 8765 и $WEB_PORT свободны."
}

show_status() {
    echo "📊 Состояние стека VALERA"
    echo "─────────────────────────────────────────────"
    if model_ready; then
        echo "   модель      : ✅ работает  $(curl -s --max-time 3 "$API_BASE/health")"
    elif [ -n "$(model_pids)" ]; then
        echo "   модель      : ⏳ грузится   (PID $(model_pids | tr '\n' ' '))"
    else
        echo "   модель      : ⛔ остановлена"
    fi

    if web_ready; then
        echo "   WebRAgent   : ✅ http://$WEB_HOST:$WEB_PORT"
    elif [ -n "$(web_pids)" ]; then
        echo "   WebRAgent   : ⏳ поднимается (PID $(web_pids | tr '\n' ' '))"
    else
        echo "   WebRAgent   : ⛔ остановлен"
    fi

    local cpid; cpid="$(client_pids)"
    echo "   клиент      : $( [ -n "$cpid" ] && echo "✅ работает (PID $cpid)" || echo "⛔ не запущен" )"
}

# ─────────────────────────────────────────────────────────────────────────────
if [ "$ACTION" = "down" ]; then
    stop_stack
    exit $?
fi

if [ "$ACTION" = "status" ]; then
    show_status
    exit 0
fi

# ═══ ЗАПУСК ═══════════════════════════════════════════════════════════════════
MODEL_PID=""
WEB_PID=""

cleanup() {
    echo ""
    echo "⏹  Получен сигнал остановки…"
    [ -n "$WEB_PID" ]   && kill "$WEB_PID" 2>/dev/null || true
    [ -n "$MODEL_PID" ] && kill "$MODEL_PID" 2>/dev/null || true
    stop_stack
    echo "👋 Стек VALERA остановлен."
    exit 0
}
trap cleanup INT TERM

echo "════════════════════════════════════════════════════"
echo "  VALERA: Qwen2.5-Omni + Silero TTS + WebRAgent"
echo "════════════════════════════════════════════════════"

# Занятые порты от предыдущего запуска — сначала гасим, потом поднимаем заново
if [ -n "$(model_pids)$(web_pids)" ] || port_busy 8765 || port_busy "$WEB_PORT"; then
    echo "ℹ️  Найдены процессы от прошлого запуска — сначала останавливаю их."
    stop_stack
    echo ""
fi

# --- 1. Модель ---------------------------------------------------------------
if [ "$RUN_MODEL" = "1" ]; then
    if model_ready; then
        echo "[1/3] Модель уже отвечает на $API_BASE — пропускаю."
    else
        echo "[1/3] Запускаю модель: python main.py (лог: $LOG_MODEL)"
        cd "$SCRIPT_DIR"
        setsid nohup "$PY" main.py > "$LOG_MODEL" 2>&1 < /dev/null &
        MODEL_PID=$!
        printf "      Загрузка модели (20.8 ГБ, обычно 30 с – 4 мин) "
        for _ in $(seq 1 120); do
            sleep 5
            if model_ready; then echo "— готова ✅"; break; fi
            if ! kill -0 "$MODEL_PID" 2>/dev/null; then
                echo ""
                echo "❌ Процесс модели завершился. Последние строки лога:"
                tail -15 "$LOG_MODEL"
                exit 1
            fi
            printf "."
        done
        if ! model_ready; then
            echo ""
            echo "❌ Модель не поднялась за 10 минут. Смотрите: $LOG_MODEL"
            exit 1
        fi
    fi
else
    echo "[1/3] Запуск модели пропущен (--no-model)."
fi

# --- 2. WebRAgent ------------------------------------------------------------
if [ "$RUN_WEB" = "1" ]; then
    echo "[2/3] Запускаю WebRAgent на http://$WEB_HOST:$WEB_PORT (лог: $LOG_WEB)"
    cd "$WEBRAGENT_DIR"
    [ -f .env ] || cp .env.example .env
    # debug=False: без авто-перезагрузчика, чтобы управлять процессом аккуратно
    setsid nohup "$PY" -c \
        "from app import create_app; create_app().run(host='$WEB_HOST', port=$WEB_PORT, debug=False)" \
        > "$LOG_WEB" 2>&1 < /dev/null &
    WEB_PID=$!

    printf "      Ожидание веб-интерфейса "
    for _ in $(seq 1 12); do
        sleep 2
        if web_ready; then echo "— готов ✅"; break; fi
        if ! kill -0 "$WEB_PID" 2>/dev/null; then
            echo ""
            echo "❌ WebRAgent завершился. Последние строки лога:"
            tail -15 "$LOG_WEB"
            exit 1
        fi
        printf "."
    done
else
    echo "[2/3] Запуск WebRAgent пропущен (--no-web)."
fi

# --- 3. Клиент ---------------------------------------------------------------
echo ""
echo "────────────────────────────────────────────────────"
echo "  модель   : $API_BASE"
echo "  веб      : http://$WEB_HOST:$WEB_PORT  (admin / admin)"
echo "  стоп     : Ctrl+C  или  bash run_all.sh --down"
echo "────────────────────────────────────────────────────"

cd "$SCRIPT_DIR"
if [ "$RUN_CLIENT" = "1" ]; then
    echo "[3/3] Голосовой клиент (Ctrl+C — выход и остановка всего)"
    CLIENT_ARGS=(--mode voice)
    [ -n "$TTS_FLAG" ] && CLIENT_ARGS+=(--tts "$TTS_FLAG")
    "$PY" client.py "${CLIENT_ARGS[@]}"
    # клиент завершился (Ctrl+C) — гасим стек
    stop_stack
    exit 0
fi

# Без клиента: ждём сигнала, держа стек живым
echo "[3/3] Клиент не запрошен. Стек работает — Ctrl+C для остановки."
while true; do
    sleep 5
    # если модель упала — сообщаем и выходим
    if ! model_ready && [ -z "$(model_pids)" ]; then
        echo "⚠️  Модель остановилась внезапно. Смотрите: $LOG_MODEL"
        break
    fi
done

cleanup
