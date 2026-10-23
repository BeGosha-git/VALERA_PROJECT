#!/bin/bash
# run_all.sh – объединённый запуск WebRAgent (Docker) + Ollama + TTS
# ============================================================================
#
#   ./run_all.sh [talk|voice|voice-fast|web|gui|kb-ingest|reset] [опции]
#
# Режимы:
#   talk          — (по умолчанию) стек без приложения: поднимает WebRAgent,
#                   Ollama и TTS и держит их запущенными до Ctrl+C.
#   voice         — voice_assistant.py (с подтверждением вопроса)
#   voice-fast    — voice_assistant.py (БЕЗ подтверждения, сразу ответ)
#   web           — ⭐ РЕКОМЕНДУЕТСЯ. Веб-пульт на «невзрачном» порту 8137:
#                   тумблеры, визуализация звука, устройства, чаты. Открывается
#                   в браузере (в т.ч. с планшета/телефона), заменяет GUI.
#   gui           — настольный графический пульт (valera_gui.py)
#   kb-ingest     — залить базу знаний МИРЭА из архива в WebRAgent
#   reset         — сбросить состояние: файлы (chat_id/log/настройки) и,
#                   с --chats, все чаты WebRAgent из MongoDB
#
# Опции (передаются после режима):
#   --stt kairos    — модель распознавания речи Kairos (лёгкая, быстрая; по умолчанию)
#   --keep-ollama   — НЕ перезапускать Ollama (использовать уже запущенную)
#   --new-chat      — начать новый разговор (сбросить контекст)
#   --no-clean      — НЕ чистить текст вопроса (сохранять имена/мат в вопросе)
#   --preload       — прогреть STT-модель при старте (первая фраза быстрее)
#   --barge-in      — прерывать озвучку ответа, если пользователь начал говорить
#   --no-auto-vad   — выключить авто-регулировку чувствительности (ручной порог)
#   --web-port N    — (только web) порт веб-пульта (по умолчанию 8137)
#   --web-host H    — (только web) адрес веб-пульта; 0.0.0.0 — доступ из сети
#   --with-full     — (только kb-ingest) залить и полный документ базы знаний
#   --reset-kb      — (только kb-ingest) пересоздать коллекцию «МИРЭА» с нуля
#   --chats         — (только reset) удалить все чаты WebRAgent из MongoDB
#   --dry-run       — (только reset) показать, что будет удалено
#   --yes           — (только reset) без вопроса-подтверждения
#   --proxy URL     — HTTP-прокси для исходящих запросов (SearXNG→поисковики,
#                     приложение, Ollama). Пример: --proxy http://127.0.0.1:12334
#
# Важно про Ollama:
#   По умолчанию скрипт ОСТАНАВЛИВАЕТ все процессы `ollama serve`, запущенные
#   вне проекта (чужие), и запускает свою копию. Это гарантирует, что
#   используется нужная модель и настройки проекта.
#   Если вы сами запустили Ollama и хотите её сохранить — добавьте --keep-ollama.
#
# Жизненный цикл:
#   Скрипт владеет ВСЕМ стеком сразу: веб-пульт/GUI запускается вместе с
#   сервером и гасится вместе с ним (Ctrl+C). Порт веб-пульта освобождается
#   от процессов прошлых запусков, поэтому повторный старт не падает с
#   «Address already in use» и не показывает пустую страницу.
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
#OLLAMA_MODEL_NAME="valera_live:latest"
OLLAMA_MODEL_NAME="true_valera:latest"

# Режим запуска и доп. аргументы
RUN_MODE="${1:-talk}"
EXTRA_ARGS="${*:2}"

# Разбираем флаг --keep-ollama (не перезапускать чужую Ollama)
KEEP_OLLAMA=0
if [[ "$EXTRA_ARGS" == *"--keep-ollama"* ]]; then
    KEEP_OLLAMA=1
    EXTRA_ARGS="${EXTRA_ARGS//--keep-ollama/}"
fi

# Разбираем флаг --proxy URL (прокси для исходящих запросов SearXNG и приложения)
PROXY=""
if [[ "$EXTRA_ARGS" == *"--proxy"* ]]; then
    # Извлекаем значение после --proxy (в формате: --proxy http://host:port)
    PROXY="$(echo "$EXTRA_ARGS" | sed -n 's/.*--proxy[ =]\?\([^ ]*\).*/\1/p')"
    EXTRA_ARGS="$(echo "$EXTRA_ARGS" | sed 's/--proxy[ =]\?[^ ]*//')"
fi

# Разбираем флаги, специфичные для kb-ingest (в voice_assistant они не нужны)
KB_WITH_FULL=0
KB_RESET=0
if [[ "$EXTRA_ARGS" == *"--with-full"* ]]; then
    KB_WITH_FULL=1
    EXTRA_ARGS="${EXTRA_ARGS//--with-full/}"
fi
if [[ "$EXTRA_ARGS" == *"--reset-kb"* ]]; then
    KB_RESET=1
    EXTRA_ARGS="${EXTRA_ARGS//--reset-kb/}"
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

# ---------------------------------------------------------------------------
# Управление приложением (веб-пульт / GUI)
# ---------------------------------------------------------------------------
# Жизненный цикл приложения привязан к стеку: запускается вместе с ним и
# гасится вместе с ним (Ctrl+C, ошибка, любой выход). Иначе сайт оставался
# висеть после остановки сервера и держал порт — при следующем запуске
# «порт занят» и пустая страница.

# PID приложения, запущенного этим скриптом
APP_PID=""
APP_PID_FILE="$SCRIPT_DIR/.valera_app.pid"
WEB_PORT_DEFAULT=8137
WEB_PORT=""   # заполняется в режиме web

# Освобождает порт веб-пульта: убивает зависшие процессы прошлых запусков.
# Без этого повторный запуск падает с «Address already in use».
free_app_port() {
    local port="${1:-$WEB_PORT_DEFAULT}"
    # fuser — самый надёжный способ найти владельца порта
    if command -v fuser >/dev/null 2>&1; then
        fuser -k "${port}/tcp" >/dev/null 2>&1 || true
    fi
    # Подстраховка: процессы проекта с веб-режимом
    pkill -f "valera_web.py" >/dev/null 2>&1 || true
    pkill -f "voice_assistant.py --web" >/dev/null 2>&1 || true
    sleep 1
}

# Останавливает приложение этого скрипта (и подчищает осиротевшие процессы)
stop_app() {
    local pid=""
    # 1) PID, сохранённый в переменной
    if [ -n "$APP_PID" ]; then
        pid="$APP_PID"
    # 2) PID из файла (если скрипт перезапускали)
    elif [ -f "$APP_PID_FILE" ]; then
        pid="$(cat "$APP_PID_FILE" 2>/dev/null || true)"
    fi

    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "Остановка приложения (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        sleep 1
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$APP_PID_FILE"

    # 3) На случай осиротевших процессов, оставшихся от старых запусков
    free_app_port "${WEB_PORT:-$WEB_PORT_DEFAULT}"
}

# Создаёт модель проекта (true_valera), если её нет
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

    # Быстрая версия (3B) — почти мгновенные ответы. Создаём, если есть
    # llama3.2:3b и ещё нет true_valera_fast. Если базовой модели нет —
    # подсказываем команду, но не падаем (без сети сборка невозможна).
    local fast_model="true_valera_fast:latest"
    local fast_modelfile="$SCRIPT_DIR/Modelfile-FAST"
    if ollama list 2>/dev/null | grep -q "$fast_model"; then
        echo "[Ollama] Быстрая модель $fast_model уже установлена."
    elif [ -f "$fast_modelfile" ]; then
        if ollama list 2>/dev/null | grep -q "llama3.2:3b"; then
            echo "[Ollama] Создаю быструю модель $fast_model из $fast_modelfile..."
            ollama create "$fast_model" -f "$fast_modelfile" \
                && echo "  ✓ Быстрая модель $fast_model создана" \
                || echo "  ⚠️ Не удалось создать $fast_model"
        else
            echo "[Ollama] Нет базовой llama3.2:3b. Для быстрых ответов выполните:"
            echo "         ollama pull llama3.2:3b && ./run_all.sh web"
        fi
    fi
}

# ---------------------------------------------------------------------------
# Функция очистки при завершении
# ---------------------------------------------------------------------------
cleanup() {
    echo ""
    echo "=== Остановка всех сервисов ==="
    # Приложение (веб-пульт/GUI) — ПЕРВЫМ, он держит порт и микрофон
    stop_app
    if [ -f "$WEBRAGENT_DIR/docker-compose.yml" ]; then
        docker compose -f "$WEBRAGENT_DIR/docker-compose.yml" down 2>/dev/null || true
    fi
    # Останавливаем СВОЮ ollama (запущенную этим скриптом), чужие — не трогаем
    stop_own_ollama
    echo "Готово."
    exit 0
}

# Перехватываем Ctrl+C и SIGTERM (для корректного гашения всего стека)
trap cleanup SIGINT SIGTERM

echo "============================================"
echo "  Запуск единого стека: WebRAgent + Ollama + TTS"
echo "  Режим: $RUN_MODE"
echo "  Аргументы: $EXTRA_ARGS"
# Модель распознавания речи: Kairos — лёгкая и быстрая
if [[ "$EXTRA_ARGS" == *"--stt"* ]]; then
    echo "  STT: Kairos (лёгкая, быстрая)"
fi
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

if [ "$RUN_MODE" = "voice" ] || [ "$RUN_MODE" = "voice-fast" ] || [ "$RUN_MODE" = "web" ] || [ "$RUN_MODE" = "gui" ] || [ "$RUN_MODE" = "kb-ingest" ] || [ "$RUN_MODE" = "reset" ]; then
    create_env_if_missing "$MIC_ENV_NAME" "$MIC_DIR/environment.yml"
fi

# 3. Запуск Ollama (перезапуск, если требуется)
#    Для режима reset Ollama не нужна — там только файлы и MongoDB.
echo ""
echo "[3/5] Настройка Ollama..."

if [ "$RUN_MODE" = "reset" ]; then
    echo "  reset: Ollama не требуется — пропускаю."
elif [ "$KEEP_OLLAMA" = "1" ]; then
    echo "  --keep-ollama: используем уже запущенную Ollama."
    if ollama_is_up; then
        echo "  ✓ Ollama уже отвечает на :11434"
    else
        echo "  ⚠️ Ollama не отвечает, но --keep-ollama указан. Запускаю свою..."
        OLLAMA_HOST=0.0.0.0:11434 \
        OLLAMA_MAX_LOADED_MODELS=1 \
        OLLAMA_NUM_PARALLEL=1 \
        OLLAMA_KEEP_ALIVE=30m \
            ollama serve &
        OLLAMA_PID=$!
        echo "  ollama serve запущен (PID $OLLAMA_PID)"
    fi
else
    echo "  Останавливаю чужие/старые процессы ollama для перезапуска проекта..."
    stop_all_ollama

    echo "  Запускаю ollama serve для проекта..."
    # OLLAMA_HOST=0.0.0.0:11434 — критично для Docker: по умолчанию Ollama
    # слушает только 127.0.0.1, и контейнер WebRAgent не может достучаться до
    # него по 172.17.0.1 (docker bridge). А через localhost запросы с большим
    # контекстом обрываются VPN — отсюда 500 на /api/generate и «зависший»
    # поиск по базе знаний.
    #
    # OLLAMA_MAX_LOADED_MODELS=1 — ОДНА модель в памяти.
    # Без этого Ollama держала одновременно true_valera (5.3 ГБ) и
    # true_valera_fast (2.3 ГБ), и на каждый запрос «другой» модели
    # перезагружала раннер. В этот момент /tokenize отвечал ошибкой
    # `context canceled`, а клиент висел до таймаута (400–500 в логе, 45–60 с
    # тишины для гостя). С одной моделью перезагрузок нет.
    # OLLAMA_NUM_PARALLEL=1 — один запрос за раз: голосовому роботу больше не
    # нужно, а лишние слоты только делят контекст и память.
    # OLLAMA_KEEP_ALIVE=30m — модель не выгружается между вопросами.
    OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_MAX_LOADED_MODELS=1 \
    OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_KEEP_ALIVE=30m \
        ollama serve &
    OLLAMA_PID=$!
    echo "  ollama serve запущен (PID $OLLAMA_PID, слушает 0.0.0.0:11434)"
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

# Гарантируем наличие модели проекта (для reset модель не нужна)
if [ "$RUN_MODE" != "reset" ]; then
    ensure_model
fi

# 4. Запуск WebRAgent через Docker Compose
#    (нужен и для reset — чистка чатов идёт через MongoDB в контейнере)
echo ""
echo "[4/5] Запуск WebRAgent через Docker Compose..."
cd "$WEBRAGENT_DIR"
if [ -n "$PROXY" ]; then
    echo "  🌐 Прокси для SearXNG (исходящие запросы к поисковикам): $PROXY"
    # Внутри контейнера searxng хост доступен как host.docker.internal,
    # поэтому заменяем 127.0.0.1/localhost на этот адрес для прокси.
    PROXY_URL_FOR_DOCKER="$(echo "$PROXY" | sed -E 's#//(127\.0\.0\.1|localhost):#//host.docker.internal:#')"
    echo "  → Для Docker-контейнера: $PROXY_URL_FOR_DOCKER"
    PROXY_URL="$PROXY_URL_FOR_DOCKER" docker compose up -d
else
    docker compose up -d
fi
cd "$SCRIPT_DIR"

# Даём сервисам время подняться
echo "Ожидание запуска сервисов (5 секунд)..."
sleep 5

# 5. Запуск приложения
# Если задан прокси — экспортируем его в окружение (requests в приложении и
# Ollama подхватят HTTP_PROXY/HTTPS_PROXY автоматически).
if [ -n "$PROXY" ]; then
    export HTTP_PROXY="$PROXY"
    export HTTPS_PROXY="$PROXY"
    export ALL_PROXY="$PROXY"
    export NO_PROXY="localhost,127.0.0.1,::1,searxng,qdrant,mongodb,app"
fi

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
elif [ "$RUN_MODE" = "web" ]; then
    # --- Веб-пульт: сайт вместо консоли и настольного GUI ---
    echo ""
    echo "[5/5] Активация окружения '$MIC_ENV_NAME' (веб-пульт)..."
    conda activate "$MIC_ENV_NAME"

    # Порт/адрес можно переопределить флагами --web-port / --web-host
    WEB_PORT="$WEB_PORT_DEFAULT"
    WEB_HOST="127.0.0.1"
    if [[ "$EXTRA_ARGS" == *"--web-port"* ]]; then
        WEB_PORT="$(echo "$EXTRA_ARGS" | sed -n 's/.*--web-port[ =]\?\([0-9]*\).*/\1/p')"
        EXTRA_ARGS="$(echo "$EXTRA_ARGS" | sed 's/--web-port[ =]\?[0-9]*//')"
    fi
    if [[ "$EXTRA_ARGS" == *"--web-host"* ]]; then
        WEB_HOST="$(echo "$EXTRA_ARGS" | sed -n 's/.*--web-host[ =]\?\([^ ]*\).*/\1/p')"
        EXTRA_ARGS="$(echo "$EXTRA_ARGS" | sed 's/--web-host[ =]\?[^ ]*//')"
    fi

    # Освобождаем порт от процессов прошлых запусков — иначе «Address already in use»
    echo "  Освобождаю порт ${WEB_PORT}..."
    free_app_port "$WEB_PORT"

    echo ""
    echo "[5/5] Запуск веб-пульта Валеры на http://${WEB_HOST}:${WEB_PORT}..."
    echo "  Откройте этот адрес в браузере. Интерфейс заменяет настольный GUI:"
    echo "  тумблеры функций, визуализация звука, устройства, триггеры, чаты."
    echo "  Пульт остановится вместе со стеком (Ctrl+C)."
    cd "$SCRIPT_DIR"

    # Запускаем в ФОНЕ и запоминаем PID: так cleanup() сможет его погасить,
    # а скрипт остаётся «хозяином» стека и ждёт Ctrl+C.
    python voice_assistant.py --web --web-port "$WEB_PORT" --web-host "$WEB_HOST" $EXTRA_ARGS &
    APP_PID=$!
    echo "$APP_PID" > "$APP_PID_FILE"
    echo "  ✓ Веб-пульт запущен (PID $APP_PID)"

    # Ждём, пока приложение живо. Если оно упадёт — сообщаем и завершаем стек.
    wait "$APP_PID" || true
    echo ""
    echo "  ⚠️ Веб-пульт завершился — останавливаю стек."
elif [ "$RUN_MODE" = "gui" ]; then
    # --- Графический пульт: тумблеры, визуализация звука, триггеры ---
    echo ""
    echo "[5/5] Активация окружения '$MIC_ENV_NAME' (GUI)..."
    conda activate "$MIC_ENV_NAME"

    echo ""
    echo "[5/5] Запуск valera_gui.py ${EXTRA_ARGS}..."
    echo "  В окне доступны: тумблеры функций, авто/ручная чувствительность,"
    echo "  визуализация звука (микрофон → шумодав → итог) и триггеры «слово → curl»."
    cd "$SCRIPT_DIR"
    python valera_gui.py $EXTRA_ARGS
elif [ "$RUN_MODE" = "kb-ingest" ]; then
    # --- Загрузка базы знаний МИРЭА (быстрый путь: прямо в контейнере) -----
    # Почему НЕ по HTTP: в этом окружении VPN обрезает крупные HTTP-запросы
    # (>16 КБ), поэтому заливка через веб-интерфейс WebRAgent «виснет» на
    # больших файлах. Direct-путь читает файлы внутри контейнера и пишет в
    # Qdrant напрямую через REST (в обход loopback) — вся база (~170 файлов)
    # заливается за ~12 секунд.
    echo ""
    echo "[5/5] Заливка базы знаний в контейнере (быстрый путь)..."
    cd "$SCRIPT_DIR"
    python kb_ingest.py --help >/dev/null 2>&1 || true

    APP_CID="$(docker compose -f "$WEBRAGENT_DIR/docker-compose.yml" ps -q app 2>/dev/null || true)"
    if [ -z "$APP_CID" ]; then
        echo "  ⚠️ Контейнер app не запущен — некуда заливать. Сначала: ./run_all.sh"
        return 1 2>/dev/null || exit 1
    fi

    echo "  Копирую базу знаний в контейнер..."
    docker exec "$APP_CID" rm -rf /tmp/kb >/dev/null 2>&1 || true
    docker cp "$SCRIPT_DIR/knowledge_base/rtu_mirea_llm_kb" "$APP_CID:/tmp/kb" 2>/dev/null
    docker cp "$SCRIPT_DIR/WebRAgent/ingest_local.py" "$APP_CID:/tmp/ingest_local.py" 2>/dev/null

    RESET_FLAG=""
    [ "$KB_RESET" = "1" ] && RESET_FLAG="--reset"

    echo "  Заливаю (прямой REST, вся база)..."
    docker exec "$APP_CID" python /tmp/ingest_local.py \
        --kb /tmp/kb --collection "МИРЭА" $RESET_FLAG
elif [ "$RUN_MODE" = "reset" ]; then
    # --- Сброс состояния Валеры (файлы + чаты WebRAgent) ---
    echo ""
    echo "[5/5] Активация окружения '$MIC_ENV_NAME' (reset)..."
    conda activate "$MIC_ENV_NAME"

    # По умолчанию чистим и чаты (это самое полезное), но можно переопределить
    if [[ "$EXTRA_ARGS" != *"--dry-run"* ]]; then
        EXTRA_ARGS="$EXTRA_ARGS --chats"
    fi

    echo ""
    echo "[5/5] Запуск valera_reset.py ${EXTRA_ARGS}..."
    cd "$SCRIPT_DIR"
    python valera_reset.py $EXTRA_ARGS
else
    # --- Режим по умолчанию: стек работает, приложение не запускается ---
    # Раньше здесь стартовал TALK.py. Теперь мы просто ДЕРЖИМ стек живым:
    # WebRAgent, Ollama и TTS уже подняты выше, а скрипт остаётся «хозяином»
    # стека и ждёт Ctrl+C, после чего cleanup() корректно всё гасит.
    echo ""
    echo "[5/5] Режим по умолчанию: стек запущен, приложение не стартует."
    echo "  Что доступно:"
    echo "    • Веб-пульт Валера:  ./run_all.sh web"
    echo "    • Голосовой бот:     ./run_all.sh voice"
    echo "    • Настольный GUI:    ./run_all.sh gui"
    echo "  Ctrl+C — остановить весь стек."

    # Бесконечный цикл: пока пользователь не нажмёт Ctrl+C (SIGINT/SIGTERM),
    # процесс живёт и держит сервер + Ollama. sleep запускаем в фоне и ждём
    # через wait — так сигнал прерывает ожидание МГНОВЕННО (в отличие от
    # простого цикла «sleep 1», где выход задерживается до конца тика).
    while true; do
        sleep 3600 &
        wait $! 2>/dev/null || true
    done
fi

# Приложение отработало — завершаем
cleanup