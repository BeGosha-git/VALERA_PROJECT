"""Audio input/output utilities.

Handles:
- Microphone capture (sounddevice)
- Audio playback from numpy arrays
- Audio file I/O with proper resampling to model's native 24kHz
"""

import io
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from loguru import logger

from config import settings

# Model's native sample rate
NATIVE_SR = settings.sample_rate  # 24000


def list_audio_devices() -> list[dict]:
    """List available audio input/output devices."""
    devices = []
    for i, dev in enumerate(sd.query_devices()):
        devices.append({
            "id": i,
            "name": dev["name"],
            "max_input_channels": dev["max_input_channels"],
            "max_output_channels": dev["max_output_channels"],
            "default_samplerate": dev["default_samplerate"],
        })
    return devices


def get_default_microphone() -> Optional[int]:
    """Get the default input device ID."""
    try:
        return sd.default.device[0]  # input device
    except (OSError, sd.PortAudioError):
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                return i
    return None


def get_default_speaker() -> Optional[int]:
    """Get the default output device ID."""
    try:
        return sd.default.device[1]  # output device
    except (OSError, sd.PortAudioError):
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev["max_output_channels"] > 0:
                return i
    return None


def capture_microphone(
    duration: float = 5.0,
    sample_rate: int = NATIVE_SR,
    device: Optional[int] = None,
) -> np.ndarray:
    """Capture audio from microphone.

    Args:
        duration: Recording duration in seconds.
        sample_rate: Sample rate (default: model native 24000 Hz).
        device: Input device ID. None = system default.

    Returns:
        Recorded audio as float32 numpy array, shape (samples,).

    Raises:
        RuntimeError: If no input device is available.
    """
    if device is None:
        device = get_default_microphone()

    if device is None:
        raise RuntimeError(
            "No microphone found. Available devices:\n"
            + "\n".join(f"  [{d['id']}] {d['name']}" for d in list_audio_devices())
        )

    total_samples = int(duration * sample_rate)
    logger.info(f"Recording {duration}s from device {device} at {sample_rate}Hz...")

    recording = sd.rec(
        total_samples,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()  # wait until recording is complete

    return recording.flatten()


def capture_until_silence(
    max_duration: float = 30.0,
    silence_threshold: float = 0.02,
    silence_duration: float = 1.5,
    sample_rate: int = NATIVE_SR,
    device: Optional[int] = None,
) -> np.ndarray:
    """Record until silence is detected or max duration reached.

    Args:
        max_duration: Maximum recording duration in seconds.
        silence_threshold: RMS threshold below which audio is considered silence.
        silence_duration: Seconds of continuous silence to stop recording.
        sample_rate: Sample rate.
        device: Input device ID.

    Returns:
        Recorded audio as float32 numpy array.
    """
    if device is None:
        device = get_default_microphone()

    if device is None:
        raise RuntimeError("No microphone found.")

    chunk_samples = int(0.2 * sample_rate)  # 200ms chunks
    silence_samples = int(silence_duration * sample_rate)
    max_samples = int(max_duration * sample_rate)

    logger.info(
        f"Listening... (max {max_duration}s, silence threshold={silence_threshold})"
    )

    audio_chunks: list[np.ndarray] = []
    silent_counter = 0
    total_samples = 0

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device,
    ) as stream:
        while total_samples < max_samples:
            chunk, _ = stream.read(chunk_samples)
            chunk = chunk.flatten()
            audio_chunks.append(chunk)
            total_samples += len(chunk)

            # Check RMS for silence
            rms = np.sqrt(np.mean(chunk**2))
            if rms < silence_threshold:
                silent_counter += len(chunk)
                if silent_counter >= silence_samples:
                    logger.info("Silence detected, stopping.")
                    break
            else:
                silent_counter = 0

    audio = np.concatenate(audio_chunks)
    logger.info(f"Recorded {len(audio) / sample_rate:.1f}s of audio.")
    return audio


def play_audio(
    audio: np.ndarray,
    sample_rate: int = NATIVE_SR,
    device: Optional[int] = None,
) -> None:
    """Play audio through speakers.

    Args:
        audio: Audio waveform as float32 numpy array (N,) or (N, 1).
        sample_rate: Sample rate of the audio.
        device: Output device ID. None = system default.
    """
    if device is None:
        device = get_default_speaker()

    if audio.ndim == 1:
        audio = audio.reshape(-1, 1)

    # Normalize to avoid clipping
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val * 0.95

    sd.play(audio, samplerate=sample_rate, device=device)
    sd.wait()


def save_audio(
    audio: np.ndarray,
    filepath: Path | str,
    sample_rate: int = NATIVE_SR,
) -> Path:
    """Save audio to WAV file.

    Args:
        audio: Audio waveform (samples,) or (samples, channels).
        filepath: Output path.
        sample_rate: Sample rate.

    Returns:
        Path to saved file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(filepath), audio, sample_rate)
    return filepath


def load_audio(filepath: Path | str, target_sr: int = NATIVE_SR) -> np.ndarray:
    """Load audio file and resample to target sample rate if needed.

    Args:
        filepath: Path to audio file.
        target_sr: Target sample rate.

    Returns:
        Audio as float32 numpy array (samples,).
    """
    audio, sr = sf.read(str(filepath))

    # Convert stereo to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        import samplerate as libsamplerate
        ratio = target_sr / sr
        audio = libsamplerate.resample(audio, ratio, "sinc_best")

    return audio.astype(np.float32)


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = NATIVE_SR) -> bytes:
    """Convert numpy audio to WAV bytes (for API responses)."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue()
