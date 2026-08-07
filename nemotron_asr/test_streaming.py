#!/usr/bin/env python3
"""
Тест Nemotron 3.5 ASR — стриминговое распознавание на файле.
Проверяет весь конвейер: загрузка модели → стриминг → текст.
"""
import sys
import time

import numpy as np
import soundfile as sf
import torch

# Добавляем папку проекта в путь
sys.path.insert(0, "/home/georgiy/Desktop/VALERA_PROJECT/nemotron_asr")
from nemotron_streaming_stt import NemotronStreamingASR  # noqa: E402


def main() -> None:
    audio_path = sys.argv[1] if len(sys.argv) > 1 else "/home/georgiy/Desktop/VALERA_PROJECT/temp.wav"
    language = sys.argv[2] if len(sys.argv) > 2 else "ru-RU"

    print(f"Файл: {audio_path}")
    print(f"Язык: {language}")

    # Загрузка
    asr = NemotronStreamingASR(
        model_path="/home/georgiy/Desktop/VALERA_PROJECT/nemotron-project",
        language=language,
        latency_ms=320,
        device="auto",
    )

    # Чтение и ресемплинг в 16 кГц
    audio, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if sr != asr.sampling_rate:
        # Ресемплинг через librosa
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=asr.sampling_rate)
        sr = asr.sampling_rate
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # моно
    print(f"Аудио: {len(audio)/sr:.1f}s, {sr} Гц\n")

    # Стриминговая транскрипция
    t0 = time.time()
    text = asr.transcribe(audio)
    elapsed = time.time() - t0

    print(f"\n{'='*50}")
    print(f"РЕЗУЛЬТАТ ({elapsed:.1f}с):")
    print(f"  {text}")
    print(f"{'='*50}")

    # Проверяем, что получилось не пусто
    if len(text.strip()) > 0:
        print("\n✅ Тест пройден — распознавание работает!")
    else:
        print("\n⚠️ Пустой результат — проверьте аудио/язык.")


if __name__ == "__main__":
    main()
