import sys
import time
import queue
import threading
import numpy as np
import sounddevice as sd
from kairos_asr import KairosASR
import torch
from silero_vad import load_silero_vad
import soundfile as sf
import os
import tempfile

# ------------------------ Конфигурация ------------------------
SAMPLE_RATE = 16000
SILENCE_TIMEOUT_SEC = 0.75
MIN_SPEECH_DURATION_SEC = 0.5
VAD_THRESHOLD = 0.5

# Шумоподавление
NOISE_REDUCTION = True               # Включить/выключить
NOISE_PROFILE_DURATION = 1.5         # Длительность записи шума (сек)
# ------------------------------------------------------------

class SileroVAD:
    def __init__(self, sample_rate=16000, threshold=0.5):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.model = load_silero_vad(onnx=True)

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        audio_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
        if audio_tensor.shape[1] < 512:
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, 512 - audio_tensor.shape[1]))
        prob = self.model(audio_tensor, self.sample_rate).item()
        return prob > self.threshold


class ContinuousSTT:
    def __init__(self, model: KairosASR, vad: SileroVAD,
                 silence_timeout=0.75, min_speech_duration=0.5,
                 noise_profile=None):
        self.model = model
        self.vad = vad
        self.silence_timeout = silence_timeout
        self.min_speech_duration = min_speech_duration
        self.noise_profile = noise_profile  # массив для шумоподавления

        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.thread = None

        self.current_phrase = []
        self.last_speech_time = 0.0
        self.speech_started = False

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"⚠️ Статус: {status}", file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def _process_audio(self):
        audio_buffer = np.array([], dtype=np.float32)

        while self.is_recording:
            try:
                block = self.audio_queue.get(timeout=0.1)
                if block is None:
                    continue
                audio_buffer = np.concatenate((audio_buffer, block.flatten()))

                while len(audio_buffer) >= 512:
                    chunk = audio_buffer[:512]
                    audio_buffer = audio_buffer[512:]

                    is_speech = self.vad.is_speech(chunk)
                    current_time = time.time()

                    if is_speech:
                        if not self.speech_started:
                            self.speech_started = True
                            self.current_phrase = []
                        self.current_phrase.append(chunk)
                        self.last_speech_time = current_time
                    else:
                        if self.speech_started:
                            if current_time - self.last_speech_time >= self.silence_timeout:
                                self._finalize_phrase()
                                self.speech_started = False
                                self.current_phrase = []
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Ошибка: {e}", file=sys.stderr)

        if self.speech_started:
            self._finalize_phrase()

    def _finalize_phrase(self):
        if not self.current_phrase:
            return
        audio_np = np.concatenate(self.current_phrase)
        duration = len(audio_np) / SAMPLE_RATE
        if duration < self.min_speech_duration:
            print(f"⏩ Пропуск: {duration:.2f} сек")
            self.current_phrase = []
            return

        self.current_phrase = []
        threading.Thread(target=self._transcribe, args=(audio_np,), daemon=True).start()

    def _transcribe(self, audio_np):
        try:
            # Применяем шумоподавление, если включено и есть профиль
            if NOISE_REDUCTION and self.noise_profile is not None:
                import noisereduce as nr
                audio_np = nr.reduce_noise(
                    y=audio_np,
                    sr=SAMPLE_RATE,
                    y_noise=self.noise_profile,
                    prop_decrease=0.8,   # сила подавления (0-1)
                    n_fft=1024,
                )

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            sf.write(tmp_path, audio_np, SAMPLE_RATE, subtype='PCM_16')
            result = self.model.transcribe(wav_file=tmp_path)
            text = result.full_text.strip()
            if text:
                print(f"\n📝 {text}")
            else:
                print("🔇 (пусто)")
            os.unlink(tmp_path)
        except Exception as e:
            print(f"❌ Ошибка распознавания: {e}")

    def start(self):
        if self.is_recording:
            return

        # Запись шумового профиля (если включено)
        noise_profile = None
        if NOISE_REDUCTION:
            print(f"🔇 Запись шума в течение {NOISE_PROFILE_DURATION} сек...")
            noise_audio = sd.rec(
                int(NOISE_PROFILE_DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            noise_profile = noise_audio.flatten()
            print("✅ Профиль шума записан")

        self.is_recording = True
        self.speech_started = False
        self.current_phrase = []
        self.noise_profile = noise_profile

        self.thread = threading.Thread(target=self._process_audio, daemon=True)
        self.thread.start()

        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            callback=self._audio_callback,
            blocksize=int(SAMPLE_RATE * 0.1),
        )
        self.stream.start()
        print("🎙️ Говорите... (пауза 0.75 сек)")
        print("   Нажмите Ctrl+C для остановки.\n")

    def stop(self):
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        print("\n🛑 Остановлено.")


def main():
    print("⏳ Загрузка модели Kairos-ASR...")
    asr = KairosASR(device="auto")
    print("✅ Модель загружена")

    vad = SileroVAD(sample_rate=SAMPLE_RATE, threshold=VAD_THRESHOLD)

    stt = ContinuousSTT(
        model=asr,
        vad=vad,
        silence_timeout=SILENCE_TIMEOUT_SEC,
        min_speech_duration=MIN_SPEECH_DURATION_SEC,
        noise_profile=None  # будет заполнен в start()
    )

    try:
        stt.start()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        stt.stop()
        print("👋 Выход.")


if __name__ == "__main__":
    main()