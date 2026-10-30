#!/usr/bin/env python3
"""
XTTS-v2 — основной сервис озвучки Валеры (отдельный процесс на GPU).

Модель грузится один раз и остаётся «прогретой» в видеопамяти (~3 ГБ), поэтому
рядом спокойно живёт LLM из Ollama. Клиент — tts_backends.XttsClient в корне
проекта; если сервис недоступен, голос автоматически переключается на Silero.

API
---
POST /tts      {"text": "...", "speaker": "Damien Black", "sample_rate": 48000, "speed": 1.0}
               → сырой PCM S16_LE, моно (заголовки X-Sample-Rate, X-Synth-Seconds)
GET  /health   → {"status": "ok", "device": "...", "speakers": N}
GET  /speakers → список встроенных голосов

Голос
-----
* По умолчанию — встроенный голос XTTS (XTTS_SPEAKER, по умолчанию «Damien Black»).
* Клонирование: положите чистую запись 6–15 с в ``xtts_service/voices/valera.wav``
  (или укажите XTTS_REF_WAV) — тогда говорит этот голос.

Запуск: python xtts_service/server.py   (порт XTTS_PORT, по умолчанию 8151)

Лицензия модели — Coqui CPML (только некоммерческое использование); владелец
проекта её принял, run_all.sh передаёт сервису COQUI_TOS_AGREED=1.
"""

from __future__ import annotations

import io
import math
import os
import re
import threading
import time
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.getenv("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
# Локальная копия модели (config.json, model.pth, vocab.json, dvae.pth, mel_stats.pth,
# speakers_xtts.pth). coqui-tts 0.28 по имени модели всегда качает её заново с
# Hugging Face (tts-hub/XTTS-v2), поэтому при наличии локальной папки грузим из неё.
MODEL_DIR = os.getenv(
    "XTTS_MODEL_DIR",
    os.path.expanduser("~/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2"),
)
DEFAULT_SPEAKER = os.getenv("XTTS_SPEAKER", "Damien Black")
REF_WAV = os.getenv("XTTS_REF_WAV") or os.path.join(HERE, "voices", "valera.wav")
NATIVE_RATE = 24000  # XTTS-v2 всегда отдаёт 24 кГц
PORT = int(os.getenv("XTTS_PORT", "8151"))
HOST = os.getenv("XTTS_HOST", "127.0.0.1")


def _pick_device() -> str:
    """Карта из XTTS_DEVICE; по умолчанию cuda, если есть, иначе CPU."""
    dev = os.getenv("XTTS_DEVICE")
    if dev:
        return dev
    return "cuda" if torch.cuda.is_available() else "cpu"


def _num2words_ru(match: "re.Match") -> str:
    from num2words import num2words

    try:
        return num2words(match.group().replace(",", "."), lang="ru")
    except Exception:  # noqa: BLE001
        return match.group()


def prepare_text(text: str) -> str:
    """Числа — словами (XTTS читает русские цифры нестабильно), лишнее — убрать."""
    text = re.sub(r"-?[0-9][0-9,._]*", _num2words_ru, text)
    text = re.sub(r"[*_#`~<>|\\]", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


class SynthRequest(BaseModel):
    text: str
    speaker: Optional[str] = None
    sample_rate: int = 48000
    speed: float = 1.0


app = FastAPI(title="Valera XTTS")
_tts = None
_device = "cpu"
_lock = threading.Lock()  # одна модель — один синтез за раз
_use_clone = False


def load_model() -> None:
    global _tts, _device, _use_clone
    from TTS.api import TTS  # coqui-tts

    _device = _pick_device()
    t0 = time.time()
    print(f"[xtts] загружаю {MODEL_NAME} на {_device}…", flush=True)
    if os.path.isfile(os.path.join(MODEL_DIR, "model.pth")):
        print(f"[xtts] локальная модель: {MODEL_DIR}", flush=True)
        _tts = TTS(model_path=MODEL_DIR,
                   config_path=os.path.join(MODEL_DIR, "config.json")).to(_device)
    else:
        _tts = TTS(MODEL_NAME).to(_device)
    _use_clone = os.path.isfile(REF_WAV)
    print(f"[xtts] модель готова за {time.time() - t0:.1f} с; "
          f"голос: {'клон ' + REF_WAV if _use_clone else DEFAULT_SPEAKER}", flush=True)
    # Прогрев: первый синтез строит кэши CUDA — делаем его до первого гостя.
    _synth("Привет, я Валера.", None, 24000, 1.0)
    print("[xtts] прогрев завершён", flush=True)


def _speakers() -> list:
    try:
        return list(_tts.synthesizer.tts_model.speaker_manager.name_to_id.keys())
    except Exception:  # noqa: BLE001
        return []


def _synth(text: str, speaker: Optional[str], rate: int, speed: float) -> np.ndarray:
    kwargs = dict(text=prepare_text(text), language="ru", speed=speed)
    if _use_clone:
        kwargs["speaker_wav"] = REF_WAV
    else:
        names = _speakers()
        spk = speaker if speaker in names else (DEFAULT_SPEAKER if DEFAULT_SPEAKER in names
                                                else (names[0] if names else None))
        kwargs["speaker"] = spk
    with _lock, torch.inference_mode():
        wav = np.asarray(_tts.tts(**kwargs), dtype=np.float32)
    if rate != NATIVE_RATE:
        g = math.gcd(rate, NATIVE_RATE)
        wav = resample_poly(wav, rate // g, NATIVE_RATE // g).astype(np.float32)
    peak = float(np.max(np.abs(wav))) or 1.0
    return (wav * (0.95 * 32767 / peak)).astype(np.int16)


@app.on_event("startup")
def _startup() -> None:
    load_model()


@app.get("/health")
def health():
    return {"status": "ok" if _tts is not None else "loading", "device": _device,
            "speakers": len(_speakers()), "clone": _use_clone}


@app.get("/speakers")
def speakers():
    return {"speakers": _speakers(), "default": DEFAULT_SPEAKER, "clone": _use_clone}


@app.post("/tts")
def tts(req: SynthRequest):
    if _tts is None:
        raise HTTPException(503, "model is loading")
    text = (req.text or "").strip()
    if len(text) < 2:
        raise HTTPException(400, "text too short")
    rate = req.sample_rate if req.sample_rate in (16000, 22050, 24000, 44100, 48000) else 48000
    t0 = time.time()
    pcm = _synth(text[:600], req.speaker, rate, max(0.5, min(1.5, req.speed)))
    return Response(
        content=pcm.tobytes(), media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(rate), "X-Synth-Seconds": f"{time.time() - t0:.2f}"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
