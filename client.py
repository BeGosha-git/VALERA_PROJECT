"""Terminal-based client for QWEN-VALERA voice assistant.

Captures microphone audio, sends to API, plays back response audio.
Can also work in text-only mode.
"""

import argparse
import base64
import io
import json
import queue
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

API_BASE = "http://localhost:8765/api/v1"


def list_devices():
    """Print available audio devices."""
    print("\n🎤 Audio Devices:")
    print("-" * 60)
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        marker = " ◀ DEFAULT" if i == sd.default.device[0] or i == sd.default.device[1] else ""
        print(f"  [{i}] {dev['name']}")
        print(f"       in={dev['max_input_channels']}ch  out={dev['max_output_channels']}ch  "
              f"sr={int(dev['default_samplerate'])}Hz{marker}")
    print()


def record_audio(duration: float, sample_rate: int = 24000, device: int = None) -> np.ndarray:
    """Record audio from microphone."""
    total_samples = int(duration * sample_rate)
    print(f"🎙️  Recording {duration}s... (press Ctrl+C to stop early)")

    recording = sd.rec(
        total_samples,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    try:
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()
        recording = recording[:sd.get_stream().read_available // 4]

    return recording.flatten()


def record_until_silence(
    sample_rate: int = 24000,
    device: int = None,
    max_duration: float = 15.0,
    start_timeout: float = 25.0,
    silence_seconds: float = 1.2,
    threshold: float = None,
    block: float = 0.1,
    verbose: bool = True,
    min_speech: float = 0.45,
    pre_roll: float = 0.35,
) -> Optional[np.ndarray]:
    """Слушает микрофон и возвращает фразу, как только наступит тишина.

    Отличия от простого «порог + тишина» (иначе были ложные срабатывания на
    шум, и в модель уходило 0.6 с мусора):

    * **старт только по устойчивой речи** — нужно ``min_speech`` секунд подряд
      выше порога, одиночный всплеск шума фразу не запускает;
    * **пре-ролл** — последние ``pre_roll`` секунд до старта добавляются к
      записи, поэтому начало слова не теряется;
    * **минимальная длина** — если речи набралось меньше ``min_speech``,
      возвращается None (фраза переслушивается), а не мусор в модель;
    * хвостовая тишина обрезается.

    Args:
        sample_rate: частота дискретизации.
        device: id входного устройства.
        max_duration: максимум секунд на одну фразу.
        start_timeout: сколько ждать начало речи (сек).
        silence_seconds: сколько тишины считать концом фразы.
        threshold: порог RMS. None → измерить шум комнаты и взять его x3.
        block: размер блока чтения (сек).
        verbose: печатать измеренный шум и порог.
        min_speech: минимум секунд устойчивой речи для старта/приёма.
        pre_roll: сколько секунд до старта речи сохранить.

    Returns:
        numpy-массив с речью или None, если речи не было.
    """
    block_size = int(block * sample_rate)
    silence_blocks = max(1, int(silence_seconds / block))
    max_blocks = int(max_duration / block)
    start_blocks = int(start_timeout / block)
    speech_blocks_needed = max(1, int(min_speech / block))
    pre_roll_blocks = max(0, int(pre_roll / block))

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
        blocksize=block_size,
    ) as stream:
        # Калибровка под конкретный микрофон
        if threshold is None:
            calibration = [rms(stream.read(block_size)[0].flatten()) for _ in range(5)]
            noise = float(np.median(calibration)) if calibration else 0.0
            threshold = max(0.015, noise * 3.0)
            if verbose:
                print(
                    f" (шум {noise:.3f}, порог {threshold:.3f})",
                    end="",
                    flush=True,
                )

        history: list[np.ndarray] = []   # пре-ролл
        chunks: list[np.ndarray] = []
        speech_run = 0
        silent_run = 0
        started = False
        voiced_blocks = 0

        for i in range(start_blocks + max_blocks):
            data, _ = stream.read(block_size)
            mono = data.flatten()
            level = rms(mono)

            if not started:
                history.append(mono)
                if len(history) > pre_roll_blocks:
                    history.pop(0)

                if level >= threshold:
                    speech_run += 1
                    # Старт только после устойчивой речи, а не одиночного щелчка
                    if speech_run >= speech_blocks_needed:
                        started = True
                        chunks = list(history)  # забираем пре-ролл
                        history = []
                        voiced_blocks = speech_run
                        silent_run = 0
                else:
                    speech_run = 0
                if not started and i >= start_blocks:
                    return None  # речи так и не было
                continue

            chunks.append(mono)
            if level >= threshold:
                voiced_blocks += 1
                silent_run = 0
            else:
                silent_run += 1

            if silent_run >= silence_blocks or len(chunks) >= max_blocks:
                break

    if not started:
        return None

    # Слишком мало реальной речи — считаем, что это шум, и слушаем снова
    if voiced_blocks * block < min_speech:
        if verbose:
            print(f" (речи всего {voiced_blocks * block:.2f} с — пропускаю)", end="")
        return None

    audio = np.concatenate(chunks)
    # Обрезаем хвостовую тишину, но оставляем небольшой запас
    tail_cut = int(max(0.0, silence_seconds - 0.4) * sample_rate)
    return audio[:-tail_cut] if tail_cut and len(audio) > tail_cut else audio


_audio_out_warmed = False


def warm_up_output(sample_rate: int = 24000):
    """Прогревает аудиовыход короткой тишиной.

    Первый `sd.play()` открывает поток, и первые десятки миллисекунд звука
    теряются («звук идёт не с первого звука»). Один холостой прогон решает это.
    """
    global _audio_out_warmed
    if _audio_out_warmed:
        return
    try:
        silence = np.zeros(int(0.2 * sample_rate), dtype=np.float32)
        _play_raw(silence, sample_rate)
        _audio_out_warmed = True
    except Exception as e:
        print(f"⚠️  Не удалось прогреть аудиовыход: {e}")


def _play_raw(audio: np.ndarray, sample_rate: int):
    """Запускает воспроизведение с большим буфером (меньше потерь в начале)."""
    try:
        sd.play(audio, samplerate=sample_rate, blocking=True, latency="high")
    except TypeError:
        # старые версии sounddevice не принимают latency
        sd.play(audio, samplerate=sample_rate, blocking=True)
    sd.wait()


class StreamingPlayer:
    """Проигрывает аудио-куски по мере поступления, без пауз между ними.

    Куски приходят от сервера (по ~4 слова). Поток вывода один, поэтому
    фрагменты звучат слитно: пока играет первый, сервер уже синтезирует второй.
    """

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.queue: queue.Queue = queue.Queue()
        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self.played = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            with sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                latency="high",
            ) as stream:
                self._stream = stream
                while True:
                    item = self.queue.get()
                    if item is None:  # конец
                        break
                    stream.write(item)
        except Exception as exc:  # noqa: BLE001
            print(f"⚠️  Ошибка воспроизведения: {exc}")

    def add(self, audio: np.ndarray) -> None:
        """Ставит кусок в очередь (с небольшим запасом тишины для слитности)."""
        if audio is None or not len(audio):
            return
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        max_val = float(np.max(np.abs(audio)))
        if max_val > 1.0:
            audio = audio / max_val * 0.95
        pad = np.zeros(int(0.06 * self.sample_rate), dtype=np.float32)
        self.queue.put(np.concatenate([audio, pad]))
        self.played += 1

    def finish(self) -> None:
        self.queue.put(None)
        if self._thread:
            self._thread.join()
        sd.stop()


def stream_voice(audio: np.ndarray, sample_rate: int, session_id: str = None,
                 tts_backend: str = None, persona: str = None,
                 chunk_words: int = 4) -> dict:
    """Отправляет голос и играет ответ по мере поступления (не ждёт весь ответ)."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)

    data = {"session_id": session_id or "", "chunk_words": str(chunk_words)}
    if tts_backend:
        data["tts_backend"] = tts_backend
    if persona:
        data["persona"] = persona

    player = StreamingPlayer(sample_rate)
    player.start()

    t_start = time.time()
    first_audio = None
    full_text = ""
    session_out = session_id or ""
    chunks = 0

    with requests.post(
        f"{API_BASE}/chat/voice/stream",
        files={"audio": ("recording.wav", buffer, "audio/wav")},
        data=data,
        stream=True,
        timeout=(10, 900),
    ) as resp:
        resp.raise_for_status()
        print("⏳ Слушаю ответ…", end="", flush=True)
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            try:
                evt = json.loads(raw[6:])
            except json.JSONDecodeError:
                continue

            kind = evt.get("type")
            if kind == "text":
                full_text += evt.get("delta", "")
                if first_audio is None:
                    print("\r🤖 Валера: ", end="", flush=True)
                print(evt.get("delta", ""), end="", flush=True)
            elif kind == "audio":
                if first_audio is None:
                    first_audio = time.time() - t_start
                wav_data, sr = sf.read(io.BytesIO(base64.b64decode(evt["wav"])))
                player.add(wav_data)
                chunks += 1
            elif kind == "done":
                full_text = evt.get("text", full_text)
                session_out = session_id or session_out
            elif kind == "error":
                print(f"\n❌ Ошибка сервера: {evt.get('message')}")

    player.finish()
    print()
    if first_audio is not None:
        print(f"   🔊 первый звук через {first_audio:.2f} с "
              f"({chunks} кусков по ~{chunk_words} слова)")
    return {"text": full_text, "session_id": session_out}


def play_audio(audio: np.ndarray, sample_rate: int = 24000):
    """Play audio through speakers."""
    if audio.ndim != 1:
        audio = audio.reshape(-1)
    audio = np.asarray(audio, dtype=np.float32)

    max_val = np.max(np.abs(audio)) if audio.size else 0.0
    if max_val > 1.0:
        audio = audio / max_val * 0.95

    # Пауза спереди и сзади: устройство не «съест» начало речи
    lead_in = np.zeros(int(0.25 * sample_rate), dtype=np.float32)
    lead_out = np.zeros(int(0.15 * sample_rate), dtype=np.float32)
    audio = np.concatenate([lead_in, audio, lead_out])

    warm_up_output(sample_rate)
    print("🔊 Playing response...")
    _play_raw(audio, sample_rate)
    sd.stop()


def send_text(text: str, session_id: str = None, persona: str = None) -> dict:
    """Send text to API."""
    payload = {"text": text, "session_id": session_id, "enable_search": True}
    if persona:
        payload["persona"] = persona
    resp = requests.post(f"{API_BASE}/chat/text", json=payload)
    resp.raise_for_status()
    return resp.json()


def send_audio(audio: np.ndarray, sample_rate: int, session_id: str = None,
               text_hint: str = None,
               tts_backend: str = None,
               persona: str = None) -> dict:
    """Send audio to API and get response."""
    # Save to temporary WAV
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)

    data = {
        "session_id": session_id or "",
        "text_hint": text_hint or "",
    }
    if tts_backend:
        # "russian_tts" (Silero, по умолчанию на сервере) или "model" (Talker Qwen)
        data["tts_backend"] = tts_backend
    if persona:
        # "guide" (по умолчанию) или "mat" — с матом
        data["persona"] = persona

    resp = requests.post(
        f"{API_BASE}/chat/voice/raw",
        files={"audio": ("recording.wav", buffer, "audio/wav")},
        data=data,
        timeout=900,
    )
    resp.raise_for_status()

    # Parse audio response
    audio_data, sr = sf.read(io.BytesIO(resp.content))
    # Заголовок приходит percent-encoded UTF-8 (HTTP допускает только latin-1)
    return {
        "text": unquote(resp.headers.get("X-Response-Text", "")),
        "audio": audio_data,
        "sample_rate": sr,
        "session_id": resp.headers.get("X-Session-Id", ""),
        "audio_path": resp.headers.get("X-Audio-Path", ""),
    }


def text_mode(session_id: str = None, persona: str = None):
    """Interactive text chat mode."""
    print("\n💬 Text Chat Mode (type 'quit' to exit, 'new' for new session)")
    print(f"   Персона: {persona or 'по настройке сервера (.env)'}")
    print("-" * 50)

    while True:
        try:
            text = input("\n👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if not text:
            continue
        if text.lower() == "quit":
            print("👋 Goodbye!")
            break
        if text.lower() == "new":
            session_id = None
            print("🆕 New session started.")
            continue

        result = send_text(text, session_id, persona=persona)
        session_id = result["session_id"]
        print(f"\n🤖 Valera: {result['text']}")
        print(f"   ⏱️  {result['inference_time_ms']:.0f}ms", end="")
        if result.get("search_used"):
            print(" | 🌐 search used", end="")
        print()


def voice_mode(
    sample_rate: int = 24000,
    duration: float = 15.0,
    device: int = None,
    tts_backend: str = None,
    push_to_talk: bool = False,
    silence_seconds: float = 1.0,
    threshold: float = None,
    persona: str = None,
    stream_mode: bool = True,
    chunk_words: int = 4,
):
    """Непрерывный голосовой диалог: слушает → сразу отвечает голосом.

    Ничего нажимать не нужно: клиент ждёт начало речи, останавливает запись
    после секунды тишины, отправляет аудио на сервер и сразу проигрывает
    синтезированный ответ. Выход — Ctrl+C.

    Режим `push_to_talk=True` возвращает старое поведение (Enter — запись).
    """
    print("\n🎧 Голосовой режим: говорите в микрофон, ассистент ответит голосом")
    print(f"   TTS: {tts_backend or 'по настройке сервера (.env)'}")
    print(f"   Персона: {'МАТ' if persona == 'mat' else (persona or 'по настройке сервера')}")
    if stream_mode:
        print(f"   Режим: стриминг — озвучка порциями по {chunk_words} слова")
    else:
        print("   Режим: ждать весь ответ (без стриминга)")
    if push_to_talk:
        print(f"   Режим: нажмите Enter, затем говорите {duration:.0f} с")
    else:
        print(
            f"   Режим: авто (тишина {silence_seconds} с = конец фразы, "
            f"до {duration:.0f} с на фразу)"
        )
    print("   Выход — Ctrl+C")
    print("-" * 60)

    session_id = None

    while True:
        # --- 1. Слушаем ---------------------------------------------------
        try:
            if push_to_talk:
                input("\n⏺️  Enter — начать запись: ")
                audio = record_audio(duration, sample_rate, device)
            else:
                print("\n🎙️  Слушаю…", end="", flush=True)
                audio = record_until_silence(
                    sample_rate=sample_rate,
                    device=device,
                    max_duration=duration,
                    silence_seconds=silence_seconds,
                    threshold=threshold,
                )
                if audio is None:
                    print(" (речи не услышал)")
                    continue
                print(f" услышал {len(audio) / sample_rate:.1f} с")
        except KeyboardInterrupt:
            print("\n👋 Пока!")
            break
        except Exception as e:
            print(f"\n❌ Ошибка записи: {e}")
            continue

        if len(audio) < sample_rate * 0.3:
            print("⚠️  Слишком коротко, пропускаю.")
            continue

        # --- 2. Отправляем и отвечаем -------------------------------------
        try:
            if stream_mode:
                # Стриминг: озвучка идёт порциями, пока модель думает дальше
                stream_voice(
                    audio,
                    sample_rate,
                    session_id,
                    tts_backend=tts_backend,
                    persona=persona,
                    chunk_words=chunk_words,
                )
                continue

            print("⏳ Думаю…")
            result = send_audio(
                audio, sample_rate, session_id,
                text_hint=None, tts_backend=tts_backend, persona=persona,
            )
        except requests.exceptions.ConnectionError:
            print("❌ Нет связи с сервером. Запущен ли python main.py?")
            continue
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            continue

        session_id = result["session_id"]

        if result["text"]:
            print(f"\n🤖 Валера: {result['text']}")

        if result["audio"] is not None and len(result["audio"]) > 0:
            play_audio(result["audio"], result["sample_rate"])
            if result.get("audio_path"):
                print(f"💾 Файл ответа: {result['audio_path']}")
        else:
            print("🔇 (аудио не сгенерировано)")


def main():
    parser = argparse.ArgumentParser(description="QWEN-VALERA Voice Client")
    parser.add_argument("--mode", choices=["voice", "text"], default="voice",
                        help="Interaction mode (default: voice)")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Максимум секунд на одну фразу (voice mode)")
    parser.add_argument("--sample-rate", type=int, default=24000,
                        help="Audio sample rate")
    parser.add_argument("--device", type=int, default=None,
                        help="Input audio device ID")
    parser.add_argument("--list-devices", action="store_true",
                        help="List audio devices and exit")
    parser.add_argument("--tts", choices=["russian_tts", "model"], default=None,
                        help="Чем озвучивать ответ: russian_tts (Silero, по умолчанию) "
                             "или model (встроенный голос Qwen)")
    parser.add_argument("--push-to-talk", action="store_true",
                        help="Старый режим: Enter — запись, вместо авто-определения тишины")
    parser.add_argument("--silence", type=float, default=1.0,
                        help="Сколько секунд тишины считать концом фразы (по умолч. 1.0)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Порог RMS для тишины. По умолчанию измеряется шум комнаты x3")
    parser.add_argument("--no-stream", action="store_true",
                        help="Не стримить: ждать весь ответ (по умолчанию — стриминг)")
    parser.add_argument("--chunk-words", type=int, default=4,
                        help="Сколько слов озвучивать за раз в стриминге (по умолч. 4)")
    parser.add_argument("--mat", action="store_true",
                        help="Режим с матом (персона guide_mat)")
    parser.add_argument("--persona", choices=["guide", "mat", "default"], default=None,
                        help="Персона: guide (по умолч.), mat — с матом, default")
    parser.add_argument("--server", type=str, default="http://localhost:8765",
                        help="API server URL")

    args = parser.parse_args()

    global API_BASE
    API_BASE = f"{args.server}/api/v1"

    if args.list_devices:
        list_devices()
        return

    # Check server
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
        health = r.json()
        print(f"✅ Server: {args.server}")
        model_loaded = health.get("model_loaded")
        print(f"   Model loaded: {model_loaded}")
        if not model_loaded:
            print("   ⏳ Модель ещё загружается (~7 минут после старта сервера).")
            print("      Подождите и запустите клиент снова.")
        if health.get("gpu_memory_used_gb"):
            print(f"   GPU memory: {health['gpu_memory_used_gb']:.1f}/{health['gpu_memory_total_gb']:.1f} GB")
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {args.server}")
        print("   Start the server first: python main.py")
        sys.exit(1)
    except requests.exceptions.HTTPError as exc:
        print(f"❌ Server error: {exc}")
        print("   Проверьте лог сервера: data/valera.log")
        sys.exit(1)

    if args.mode == "text":
        text_mode(persona=args.persona or ("mat" if args.mat else None))
    else:
        list_devices()
        voice_mode(
            args.sample_rate,
            args.duration,
            args.device,
            tts_backend=args.tts,
            push_to_talk=args.push_to_talk,
            silence_seconds=args.silence,
            threshold=args.threshold,
            persona=args.persona or ("mat" if args.mat else None),
            stream_mode=not args.no_stream,
            chunk_words=args.chunk_words,
        )


if __name__ == "__main__":
    main()
