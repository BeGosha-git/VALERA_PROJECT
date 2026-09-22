#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# QWEN-VALERA Quick Launch Script
# Usage: bash run.sh [server|client|both]
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="qwen-valera"
VALERA_STORAGE="${VALERA_STORAGE:-/mnt/valera}"

# ── Внешнее хранилище (диск D) ────────────────────────────────────────────────
# Окружение, кэши и модель лежат на внешнем диске (см. setup_external_storage.sh).
if ! mountpoint -q "$VALERA_STORAGE" 2>/dev/null; then
    echo "⏳ Подключаю внешнее хранилище ($VALERA_STORAGE)..."
    sudo systemctl restart valera-storage.service 2>/dev/null \
        || sudo /usr/local/bin/valera-storage.sh up 2>/dev/null \
        || true
fi

if ! mountpoint -q "$VALERA_STORAGE" 2>/dev/null; then
    echo "❌ Внешний диск (диск D) не подключён."
    echo "   Подключите диск и выполните:  sudo bash setup_external_storage.sh"
    exit 1
fi

# Переменные окружения (PIP_CACHE_DIR, CONDA_ENVS_PATH, HF_HOME, TMPDIR)
if [ -f /etc/profile.d/valera-storage.sh ]; then
    # shellcheck disable=SC1091
    . /etc/profile.d/valera-storage.sh
fi

# Find conda
if command -v conda &> /dev/null; then
    CONDA_CMD="conda"
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    CONDA_CMD="$HOME/miniconda3/bin/conda"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
    CONDA_CMD="$HOME/anaconda3/bin/conda"
else
    echo "❌ Conda not found. Run setup_jetson6.sh first."
    exit 1
fi

# Check environment exists
if ! "$CONDA_CMD" env list 2>/dev/null | grep -q "$ENV_NAME"; then
    echo "❌ Environment '$ENV_NAME' not found. Run setup_jetson6.sh first."
    exit 1
fi

MODE="${1:-both}"

case $MODE in
    server)
        echo "🚀 Starting server on http://localhost:8765"
        "$CONDA_CMD" run -n "$ENV_NAME" python main.py
        ;;
    client)
        echo "🎤 Starting voice client..."
        "$CONDA_CMD" run -n "$ENV_NAME" python client.py --mode voice
        ;;
    both)
        echo "🚀 Starting server in background..."
        "$CONDA_CMD" run -n "$ENV_NAME" python main.py &
        SERVER_PID=$!
        sleep 10  # wait for model to load

        echo "🎤 Starting voice client..."
        "$CONDA_CMD" run -n "$ENV_NAME" python client.py --mode voice

        # Cleanup
        kill $SERVER_PID 2>/dev/null
        ;;
    text)
        echo "💬 Starting text client..."
        "$CONDA_CMD" run -n "$ENV_NAME" python client.py --mode text
        ;;
    download)
        echo "📥 Downloading model..."
        "$CONDA_CMD" run -n "$ENV_NAME" python download_model.py
        ;;
    *)
        echo "Usage: bash run.sh [server|client|both|text|download]"
        exit 1
        ;;
esac
