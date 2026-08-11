"""Test script: verify Qwen3-Omni model loads and generates audio.

Run after setup:
    conda activate qwen-valera
    python test_model.py

This will:
1. Load the model
2. Run a text-only inference
3. Generate a test audio clip
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from core.model import QwenOmniModel

logger.remove()
logger.add(sys.stderr, level="INFO")


def test_text(model_inst: QwenOmniModel):
    """Test text-only inference."""
    print("\n" + "=" * 50)
    print("TEST 1: Text inference")
    print("=" * 50)

    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": (
                    "You are a helpful voice assistant. Answer briefly and "
                    "naturally in Russian."
                )}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Привет! Расскажи коротко о себе."}
            ],
        },
    ]

    t0 = time.time()
    text, audio = model_inst.generate_response(conversation)
    elapsed = time.time() - t0

    print(f"\n💬 Response ({elapsed:.1f}s):")
    print(f"   {text}")
    if audio is not None:
        print(f"   🎵 Audio generated: {len(audio) / 24000:.1f}s")
    else:
        print("   ⚠️  No audio generated (expected for text-only test)")
    return True


def test_audio(model_inst: QwenOmniModel):
    """Test audio generation (TTS)."""
    print("\n" + "=" * 50)
    print("TEST 2: Audio generation (TTS)")
    print("=" * 50)

    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": (
                    "You are a helpful voice assistant. Answer briefly and "
                    "naturally in Russian."
                )}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "Поздравь меня с запуском голосового ассистента. "
                    "Одна короткая фраза."
                )}
            ],
        },
    ]

    t0 = time.time()
    text, audio = model_inst.generate_response(conversation)
    elapsed = time.time() - t0

    print(f"\n💬 Response ({elapsed:.1f}s):")
    print(f"   {text}")

    if audio is not None:
        # Save audio
        import soundfile as sf
        out_path = Path(__file__).parent / "data" / "test_output.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio, 24000)
        print(f"   🎵 Audio saved to: {out_path}")
        print(f"   ⏱️  Duration: {len(audio) / 24000:.1f}s")
        return True
    else:
        print("   ❌ No audio generated!")
        return False


def main():
    print("=" * 50)
    print("QWEN-VALERA Model Test")
    print("=" * 50)

    model_inst = QwenOmniModel()

    print("\nLoading model...")
    try:
        model_inst.load()
    except Exception as e:
        print(f"\n❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    success = True
    success &= test_text(model_inst)
    success &= test_audio(model_inst)

    print("\n" + "=" * 50)
    if success:
        print("✅ ALL TESTS PASSED!")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    print("=" * 50)

    model_inst.unload()


if __name__ == "__main__":
    main()
