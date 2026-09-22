"""Terminal-based client for QWEN-VALERA voice assistant.

Captures microphone audio, sends to API, plays back response audio.
Can also work in text-only mode.
"""

import argparse
import io
import sys
import tempfile
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
    start_timeout: float = 20.0,
    silence_seconds: float = 1.0,
    threshold: float = None,
    block: float = 0.1,
    verbose: bool = True,
) -> Optional[np.ndarray]:
    """Слушает микрофон и возвращает фразу, как только наступит тишина.

    Работает как в ветке PC: не нужно ничего нажимать — клиент сам ждёт начало
    речи, а затем останавливает запись после ``silence_seconds`` тишины.

    Args:
        sample_rate: частота дискретизации.
        device: id входного устройства.
        max_duration: максимум секунд на одну фразу.
        start_timeout: сколько ждать начало речи (сек).
        silence_seconds: сколько тишины считать концом фразы.
        threshold: порог RMS. None → измерить шум комнаты и взять его x3.
        block: размер блока чтения (сек).
        verbose: печатать измеренный уровень шума и порог.

    Returns:
        numpy-массив с речью или None, если речь так и не началась.
    """
    block_size = int(block * sample_rate)
    silence_blocks = max(1, int(silence_seconds / block))
    max_blocks = int(max_duration / block)
    start_blocks = int(start_timeout / block)

    chunks: list[np.ndarray] = []
    silent_run = 0
    started = False

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
        blocksize=block_size,
    ) as stream:
        # Калибровка под конкретный микрофон: порог не должен зависеть от
        # того, что в комнате шумно или включён автоматический усилитель.
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

        for i in range(start_blocks + max_blocks):
            data, _ = stream.read(block_size)
            mono = data.flatten()
            level = rms(mono)

            if not started:
                if level >= threshold:
                    started = True
                    chunks.append(mono)
                elif i >= start_blocks:
                    return None  # речь не началась — выходим, вызывающий повторит
                continue

            chunks.append(mono)
            silent_run = silent_run + 1 if level < threshold else 0
            if silent_run >= silence_blocks or len(chunks) >= max_blocks:
                break

    if not started:
        return None
    audio = np.concatenate(chunks)
    # Обрезаем хвостовую тишину
    keep = int(max(0.2, len(audio) / sample_rate - silence_seconds * 0.5) * sample_rate)
    return audio[:keep] if keep < len(audio) else audio


def play_audio(audio: np.ndarray, sample_rate: int = 24000):
    """Play audio through speakers."""
    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val * 0.95
    print("🔊 Playing response...")
    sd.play(audio, samplerate=sample_rate)
    sd.wait()


def send_text(text: str, session_id: str = None) -> dict:
    """Send text to API."""
    resp = requests.post(
        f"{API_BASE}/chat/text",
        json={"text": text, "session_id": session_id, "enable_search": True},
    )
    resp.raise_for_status()
    return resp.json()


def send_audio(audio: np.ndarray, sample_rate: int, session_id: str = None,
               text_hint: str = None,
               tts_backend: str = None) -> dict:
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


def text_mode(session_id: str = None):
    """Interactive text chat mode."""
    print("\n💬 Text Chat Mode (type 'quit' to exit, 'new' for new session)")
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

        result = send_text(text, session_id)
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
):
    """Непрерывный голосовой диалог: слушает → сразу отвечает голосом.

    Ничего нажимать не нужно: клиент ждёт начало речи, останавливает запись
    после секунды тишины, отправляет аудио на сервер и сразу проигрывает
    синтезированный ответ. Выход — Ctrl+C.

    Режим `push_to_talk=True` возвращает старое поведение (Enter — запись).
    """
    print("\n🎧 Голосовой режим: говорите в микрофон, ассистент ответит голосом")
    print(f"   TTS: {tts_backend or 'по настройке сервера (.env)'}")
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
        print("⏳ Думаю…")
        try:
            result = send_audio(
                audio, sample_rate, session_id, text_hint=None, tts_backend=tts_backend
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
        text_mode()
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
        )


if __name__ == "__main__":
    main()
