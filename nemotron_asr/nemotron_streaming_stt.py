#!/usr/bin/env python3
"""
Nemotron 3.5 ASR — Streaming Microphone → Text
================================================
Потоковое распознавание речи с микрофона через
NVIDIA Nemotron-3.5-ASR-Streaming-0.6B (Transformers).

Принцип работы:
  1. Микрофон → VAD (Silero) выделяет речевые фразы
  2. Каждая фраза → стриминговый RNNT-декодер
  3. Текст выводится пословно по мере генерации

Поддерживаемые языки: ru-RU (по умолчанию), en-US, auto и др.
Задержка (chunk): 80 / 160 / 320 / 560 / 1120 мс
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from threading import Thread
from typing import Generator, Optional

import numpy as np
import torch

from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer

# ---------------------------------------------------------------------------
# Конфигурация по умолчанию
# ---------------------------------------------------------------------------
SAMPLE_RATE: int = 16000
BLOCK_SIZE: int = 512  # семплов на колбэк (32 мс при 16 кГц)

# VAD
VAD_THRESHOLD: float = 0.3  # как в voice_assistant.py (0.3 чувствительнее, чем 0.5)
SILENCE_TIMEOUT_SEC: float = 0.75  # пауза для завершения фразы (как в voice_assistant.py)
MIN_SPEECH_DURATION_SEC: float = 0.5  # минимальная длина фразы
MAX_UTTERANCE_SEC: float = 15.0  # максимальная длина фразы (защита)

# Модель
DEFAULT_MODEL_PATH: str = "../nemotron-project"
DEFAULT_LANGUAGE: str = "ru-RU"

# Соответствие: задержка (мс) → num_lookahead_tokens
# (160 мс НЕ поддерживается моделью!)
LATENCY_TO_TOKENS: dict[int, int] = {
    80: 0,
    320: 3,
    560: 6,
    1120: 13,
}


# ---------------------------------------------------------------------------
# Silero VAD — лёгкая обёртка
# ---------------------------------------------------------------------------
class SileroVAD:
    """Детектор голосовой активности на базе Silero VAD (ONNX)."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, threshold: float = VAD_THRESHOLD) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        # silero-vad загружает ONNX-модель автоматически при первом вызове
        from silero_vad import load_silero_vad
        self._model = load_silero_vad(onnx=True)

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        audio_chunk: float32 [-1..1], любой длины.
        Возвращает True если голос.
        """
        t = torch.from_numpy(audio_chunk).float()
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if t.shape[1] < 512:
            t = torch.nn.functional.pad(t, (0, 512 - t.shape[1]))
        prob = self._model(t, self.sample_rate).item()
        return prob > self.threshold


# ---------------------------------------------------------------------------
# AudioSource — поток аудио с микрофона
# ---------------------------------------------------------------------------
class AudioSource:
    """
    Захват аудио с микрофона через `arecord` (ALSA/PipeWire).

    Использует тот же подход, что и voice_assistant.py:
      arecord -D default -r 16000 -c 1 -f S16_LE -t raw

    Устройство 'default' идёт через PipeWire/Pulse, поэтому корректно
    работает с USB-микрофонами (DJI MIC MINI) и сам делает ресемплинг.
    Можно переопределить через переменную окружения ASR_AUDIO_DEVICE
    (например: plughw:CARD=MINI,DEV=0 или hw:4,0).
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, block_size: int = BLOCK_SIZE) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size
        self._queue: queue.Queue = queue.Queue()
        self._proc: Optional[subprocess.Popen] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False

    def _capture_loop(self) -> None:
        """Читает stdout arecord и кладёт блоки float32 в очередь."""
        chunk_bytes = self.block_size * 2  # 512 сэмплов * 2 байта (S16_LE)
        try:
            while self._running:
                data = self._proc.stdout.read(chunk_bytes)
                if not data:
                    break
                # S16_LE → float32 [-1, 1]
                a = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                self._queue.put(a)
        except Exception as e:
            if self._running:
                print(f"[audio] Ошибка захвата: {e}", file=sys.stderr)

    def start(self) -> None:
        device = os.environ.get("ASR_AUDIO_DEVICE", "default")
        cmd = [
            "arecord", "-D", device,
            "-r", str(self.sample_rate),
            "-c", "1",
            "-f", "S16_LE",
            "-t", "raw",
            "-q",
        ]
        print(f"[audio] Запуск захвата: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

    def stop(self) -> None:
        self._running = False
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=2.0)
            except Exception:
                pass
            self._proc = None

    def get(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """Получить очередной блок аудио (float32, 16 кГц). None = таймаут."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear(self) -> None:
        """Очистить очередь."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break


# ---------------------------------------------------------------------------
# NemotronStreamingASR — стриминговое распознавание
# ---------------------------------------------------------------------------
class NemotronStreamingASR:
    """
    Потоковый ASR на базе Nemotron-3.5 через Transformers.
    
    Для каждой фразы:
      1. Создаётся генератор фич из накопленного аудио
      2. model.generate() + TextIteratorStreamer
      3. Текст печатается по мере появления
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        language: str = DEFAULT_LANGUAGE,
        latency_ms: int = 320,
        device: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.language = language
        self.latency_ms = latency_ms

        num_tokens = LATENCY_TO_TOKENS.get(latency_ms, 3)
        print(f"[ASR] Загрузка модели из: {model_path}")
        print(f"[ASR] Язык: {language}  |  Задержка: {latency_ms} мс  |  lookahead_tokens: {num_tokens}")

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.processor.set_num_lookahead_tokens(num_tokens)
        print(f"[ASR] streaming_latency_ms = {self.processor.streaming_latency_ms:.0f}")

        self.model = AutoModelForRNNT.from_pretrained(
            model_path,
            device_map=device,
            dtype=torch.float16,
            trust_remote_code=True,
        )
        self.model.eval()

        self.sampling_rate: int = self.processor.feature_extractor.sampling_rate
        self.samples_first_chunk: int = self.processor.num_samples_first_audio_chunk
        self.samples_per_chunk: int = self.processor.num_samples_per_audio_chunk
        self.hop_length: int = self.processor.feature_extractor.hop_length
        self.n_fft: int = self.processor.feature_extractor.n_fft

        print(f"[ASR] first_chunk: {self.samples_first_chunk} семплов ({self.samples_first_chunk / self.sampling_rate:.2f} с)")
        print(f"[ASR] per_chunk:   {self.samples_per_chunk} семплов ({self.samples_per_chunk / self.sampling_rate:.3f} с)")
        print("[ASR] Готов к работе!\n")

    def _make_feature_generator(
        self, audio: np.ndarray, first_inputs: dict
    ) -> Generator[torch.Tensor, None, None]:
        """
        Генератор mel-фич для стриминга.
        Разбивает аудио на чанки и выдаёт тензоры (1, T_mel, 128).
        """
        device = self.model.device
        dtype = self.model.dtype

        # Первый чанк (уже посчитан в transcribe)
        yield first_inputs["input_features"][:, : self.processor.num_mel_frames_first_audio_chunk, :]

        # Последующие чанки
        mel_frame_idx = self.processor.num_mel_frames_first_audio_chunk
        start_idx = mel_frame_idx * self.hop_length - self.n_fft // 2

        while True:
            end_idx = start_idx + self.samples_per_chunk
            if end_idx > len(audio):
                break

            inputs = self.processor(
                audio[start_idx:end_idx],
                sampling_rate=self.sampling_rate,
                is_streaming=True,
                is_first_audio_chunk=False,
                language=self.language,
                return_tensors="pt",
            )
            inputs = self._move_to_device(inputs)
            yield inputs["input_features"]

            mel_frame_idx += self.processor.num_mel_frames_per_audio_chunk
            start_idx = mel_frame_idx * self.hop_length - self.n_fft // 2

    def _move_to_device(self, inputs: dict):
        """Переносит тензоры на устройство/тип модели, оставляя скаляры как есть."""
        device = self.model.device
        out = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.dtype in (torch.float32, torch.float16):
                    out[k] = v.to(device, dtype=torch.float16)
                else:
                    out[k] = v.to(device)
            else:
                out[k] = v
        return out

    def transcribe(self, audio: np.ndarray) -> str:
        """
        Стриминговая транскрипция аудио-фрагмента.
        Возвращает полный текст.
        Печатает текст по мере генерации (пословно).
        """
        if len(audio) < self.samples_first_chunk:
            return ""

        # Нормализация громкости
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.9

        first_inputs = self.processor(
            audio[: self.samples_first_chunk],
            sampling_rate=self.sampling_rate,
            is_streaming=True,
            is_first_audio_chunk=True,
            language=self.language,
            return_tensors="pt",
        )
        first_inputs = self._move_to_device(first_inputs)

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_special_tokens=True,
            skip_prompt_tokens=True,
        )

        generate_kwargs = {
            **first_inputs,
            "input_features": self._make_feature_generator(audio, first_inputs),
            "streamer": streamer,
            "max_new_tokens": 512,
        }

        thread = Thread(target=self.model.generate, kwargs=generate_kwargs, daemon=True)
        thread.start()

        full_text_parts: list[str] = []
        print("  🎤 ", end="", flush=True)
        for text_chunk in streamer:
            print(text_chunk, end="", flush=True)
            full_text_parts.append(text_chunk)
        print()

        thread.join(timeout=10)
        return "".join(full_text_parts).strip()


# ---------------------------------------------------------------------------
# Основной цикл: микрофон → VAD → ASR
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nemotron 3.5 ASR — потоковое распознавание с микрофона"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_PATH,
        help="Путь к папке с моделью (локально или HF-имя)",
    )
    parser.add_argument(
        "--language", default=DEFAULT_LANGUAGE,
        help="Язык: ru-RU, en-US, de-DE, auto и др.",
    )
    parser.add_argument(
        "--latency", type=int, default=320,
        choices=[80, 320, 560, 1120],
        help="Задержка чанка в мс (меньше = быстрее, но точность чуть ниже)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Устройство: auto, cuda:0, cpu",
    )
    parser.add_argument(
        "--vad-threshold", type=float, default=VAD_THRESHOLD,
        help="Порог VAD (0..1, выше = строже)",
    )
    parser.add_argument(
        "--silence-timeout", type=float, default=SILENCE_TIMEOUT_SEC,
        help="Тишина перед концом фразы (сек)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Загрузка моделей
    # ------------------------------------------------------------------
    print("=" * 60)
    print("  Nemotron 3.5 ASR — Streaming Microphone → Text")
    print("=" * 60)

    asr = NemotronStreamingASR(
        model_path=args.model,
        language=args.language,
        latency_ms=args.latency,
        device=args.device,
    )

    print("[VAD] Загрузка Silero VAD...")
    vad = SileroVAD(sample_rate=SAMPLE_RATE, threshold=args.vad_threshold)
    print("[VAD] Готов.")

    print("\n" + "=" * 60)
    print("  Говорите в микрофон. Ctrl+C — выход.")
    print(f"  Язык: {args.language}  |  Задержка: {args.latency} мс")
    print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Аудио-источник
    # ------------------------------------------------------------------
    audio_src = AudioSource(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)
    audio_src.start()

    # Состояние VAD
    is_speaking: bool = False
    utterance_blocks: list[np.ndarray] = []
    last_speech_time: float = 0.0
    speech_start_time: float = 0.0
    _last_meter = 0.0  # для индикатора уровня

    try:
        while True:
            block = audio_src.get(timeout=0.1)
            if block is None:
                # Таймаут — проверяем тишину
                if is_speaking and (time.time() - last_speech_time) > args.silence_timeout:
                    # Конец фразы
                    audio = np.concatenate(utterance_blocks)
                    duration = len(audio) / SAMPLE_RATE
                    if duration >= MIN_SPEECH_DURATION_SEC:
                        text = asr.transcribe(audio)
                        if text:
                            print(f"  📝 {text}\n")
                    is_speaking = False
                    utterance_blocks.clear()
                continue

            now = time.time()

            # Живой индикатор уровня (раз в ~200 мс)
            if now - _last_meter > 0.2:
                _last_meter = now
                peak = float(np.abs(block).max())
                bars = int(min(peak * 40, 40))
                print(f"\r🎚️ {'█' * bars}{'░' * (40 - bars)} | {peak:.3f}    ", end="", flush=True)

            speech = vad.is_speech(block)

            if speech:
                if not is_speaking:
                    # Начало новой фразы
                    is_speaking = True
                    utterance_blocks = [block]
                    speech_start_time = now
                    print(f"\n─ Новый фрагмент ({time.strftime('%H:%M:%S')}) ─")
                else:
                    utterance_blocks.append(block)
                    # Защита от сверхдлинных фраз
                    if (now - speech_start_time) > MAX_UTTERANCE_SEC:
                        print("\n  ⚠️ Фраза слишком длинная, обрезаем.")
                        audio = np.concatenate(utterance_blocks)
                        text = asr.transcribe(audio)
                        if text:
                            print(f"  📝 {text}\n")
                        is_speaking = False
                        utterance_blocks.clear()
                last_speech_time = now
            elif is_speaking:
                utterance_blocks.append(block)
                if (now - last_speech_time) > args.silence_timeout:
                    # Конец фразы
                    audio = np.concatenate(utterance_blocks)
                    duration = len(audio) / SAMPLE_RATE
                    if duration >= MIN_SPEECH_DURATION_SEC:
                        text = asr.transcribe(audio)
                        if text:
                            print(f"  📝 {text}\n")
                    is_speaking = False
                    utterance_blocks.clear()

    except KeyboardInterrupt:
        print("\n\n[EXIT] Завершение работы...")
    finally:
        audio_src.stop()
        print("[EXIT] Микрофон остановлен. До свидания!")


if __name__ == "__main__":
    main()
