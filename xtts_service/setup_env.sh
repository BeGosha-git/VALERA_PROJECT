#!/bin/bash
# Создаёт окружение xtts_env БЕЗ повторной загрузки torch (на медленном канале
# это несколько гигабайт). Python 3.10 — как в tts_env, поэтому его torch
# (2.13 + CUDA 13, RTX 50xx работает) подключается к xtts_env через .pth.
# Докачивается только coqui-tts и его лёгкие зависимости.
#
# Минус: пока xtts_env жив, не удаляйте и не обновляйте torch в tts_env.
set -e
ENV_NAME="xtts_env"
source "$(conda info --base)/etc/profile.d/conda.sh"

TTS_SP="$(conda run -n tts_env python -c 'import site; print(site.getsitepackages()[0])')"
[ -d "$TTS_SP/torch" ] || { echo "В tts_env нет torch ($TTS_SP) — сначала создайте tts_env."; exit 1; }

if conda env list | grep -q "^$ENV_NAME "; then
    echo "Пересоздаю $ENV_NAME (python 3.10)..."
    conda remove -n "$ENV_NAME" --all -y >/dev/null
fi
conda create -n "$ENV_NAME" python=3.10 pip -y --offline 2>/dev/null \
    || conda create -n "$ENV_NAME" python=3.10 pip -y

SP="$(conda run -n "$ENV_NAME" python -c 'import site; print(site.getsitepackages()[0])')"
echo "$TTS_SP" > "$SP/_tts_env_torch.pth"
echo "torch берётся из: $TTS_SP"

# --prefer-binary: не собирать пакеты из исходников. transformers<5: coqui-tts не дружит с 5.x.
conda activate "$ENV_NAME"
pip install --prefer-binary "transformers>=4.52,<5" "coqui-tts[codec]>=0.26" \
    fastapi "uvicorn[standard]" pydantic num2words
python -c "import torch, TTS; print('OK: torch', torch.__version__, 'cuda', torch.cuda.is_available())"
