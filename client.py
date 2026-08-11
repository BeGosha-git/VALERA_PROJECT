"""Terminal-based client for QWEN-VALERA voice assistant.

Captures microphone audio, sends to API, plays back response audio.
Can also work in text-only mode.
"""

import argparse
import io
import sys
import tempfile
from pathlib import Path

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
               text_hint: str = None) -> dict:
    """Send audio to API and get response."""
    # Save to temporary WAV
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)

    resp = requests.post(
        f"{API_BASE}/chat/voice/raw",
        files={"audio": ("recording.wav", buffer, "audio/wav")},
        data={
            "session_id": session_id or "",
            "text_hint": text_hint or "",
        },
    )
    resp.raise_for_status()

    # Parse audio response
    audio_data, sr = sf.read(io.BytesIO(resp.content))
    return {
        "text": resp.headers.get("X-Response-Text", ""),
        "audio": audio_data,
        "sample_rate": sr,
        "session_id": resp.headers.get("X-Session-Id", ""),
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


def voice_mode(sample_rate: int = 24000, duration: float = 10.0, device: int = None):
    """Interactive voice chat mode."""
    print("\n🎤 Voice Chat Mode (press Enter to record, type 'text' for text mode, 'quit' to exit)")
    print("-" * 50)

    session_id = None

    while True:
        try:
            cmd = input("\n⏺️  Press Enter to speak (or command): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            break

        if cmd.lower() == "quit":
            print("👋 Goodbye!")
            break
        if cmd.lower() == "text":
            text_mode(session_id)
            continue
        if cmd.lower() == "new":
            session_id = None
            print("🆕 New session started.")
            continue

        # Record
        try:
            audio = record_audio(duration, sample_rate, device)
        except KeyboardInterrupt:
            continue

        if len(audio) < sample_rate * 0.5:
            print("⚠️  Too short, skipping.")
            continue

        # Optional text hint
        hint = input("💬 Text hint (optional, Enter to skip): ").strip() or None

        # Send to API
        print("⏳ Processing...")
        try:
            result = send_audio(audio, sample_rate, session_id, hint)
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to server. Is it running?")
            continue
        except Exception as e:
            print(f"❌ Error: {e}")
            continue

        session_id = result["session_id"]

        # Show text
        if result["text"]:
            print(f"\n🤖 Valera: {result['text']}")

        # Play audio
        if result["audio"] is not None and len(result["audio"]) > 0:
            play_audio(result["audio"], result["sample_rate"])


def main():
    parser = argparse.ArgumentParser(description="QWEN-VALERA Voice Client")
    parser.add_argument("--mode", choices=["voice", "text"], default="voice",
                        help="Interaction mode (default: voice)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Max recording duration in seconds (voice mode)")
    parser.add_argument("--sample-rate", type=int, default=24000,
                        help="Audio sample rate")
    parser.add_argument("--device", type=int, default=None,
                        help="Input audio device ID")
    parser.add_argument("--list-devices", action="store_true",
                        help="List audio devices and exit")
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
        r = requests.get(f"{args.server}/health", timeout=5)
        health = r.json()
        print(f"✅ Server: {args.server}")
        print(f"   Model loaded: {health['model_loaded']}")
        if health.get("gpu_memory_used_gb"):
            print(f"   GPU memory: {health['gpu_memory_used_gb']:.1f}/{health['gpu_memory_total_gb']:.1f} GB")
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {args.server}")
        print("   Start the server first: python main.py")
        sys.exit(1)

    if args.mode == "text":
        text_mode()
    else:
        list_devices()
        voice_mode(args.sample_rate, args.duration, args.device)


if __name__ == "__main__":
    main()
