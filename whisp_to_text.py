import sys
import time
import queue
import threading
import numpy as np
import sounddevice as sd
import whisper
import soundfile as sf
import tempfile
import os
import argparse
import torch
import noisereduce as nr

# ------------------------ КОНФИГУРАЦИЯ ------------------------
SAMPLE_RATE = 16000
SILENCE_TIMEOUT_SEC = 0.75
MIN_SPEECH_DURATION_SEC = 0.5
ENERGY_THRESHOLD = 0.01
NOISE_PROFILE_DURATION = 1.5   # длительность записи шума
NOISE_REDUCTION = True         # включить/выключить шумоподавление
# ------------------------------------------------------------

class ContinuousSTT:
    def __init__(self, model, silence_timeout=0.75, min_speech_duration=0.5):
        self.model = model
        self.silence_timeout = silence_timeout
        self.min_speech_duration = min_speech_duration
        self.noise_profile = None  # будет записан при старте

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

                block_size = int(SAMPLE_RATE * 0.1)
                while len(audio_buffer) >= block_size:
                    chunk = audio_buffer[:block_size]
                    audio_buffer = audio_buffer[block_size:]

                    energy = np.sqrt(np.mean(chunk**2))
                    is_speech = energy > ENERGY_THRESHOLD

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
            # Применяем шумоподавление, если включено и профиль записан
            if NOISE_REDUCTION and self.noise_profile is not None:
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

            fp16 = torch.cuda.is_available()
            result = self.model.transcribe(
                tmp_path,
                language="ru",
                task="transcribe",
                fp16=fp16
            )
            text = result.get("text", "").strip()
            text = ' '.join(text.split())

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
        if NOISE_REDUCTION:
            print(f"🔇 Запись шума в течение {NOISE_PROFILE_DURATION} сек... (молчите!)")
            noise_audio = sd.rec(
                int(NOISE_PROFILE_DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            self.noise_profile = noise_audio.flatten()
            print("✅ Профиль шума записан")

        self.is_recording = True
        self.speech_started = False
        self.current_phrase = []

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
    parser = argparse.ArgumentParser(description="Непрерывное распознавание речи с Whisper + шумоподавление")
    parser.add_argument("--model", type=str, default="medium",
                        choices=["tiny", "base", "small", "medium", "large", "large-v3"],
                        help="Размер модели Whisper (по умолчанию: medium)")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Порог энергии для детектора речи (0.005-0.03)")
    parser.add_argument("--noise-reduction", action="store_true", default=True,
                        help="Включить шумоподавление")
    parser.add_argument("--no-noise-reduction", dest="noise_reduction", action="store_false",
                        help="Отключить шумоподавление")
    args = parser.parse_args()

    global ENERGY_THRESHOLD, NOISE_REDUCTION
    ENERGY_THRESHOLD = args.threshold
    NOISE_REDUCTION = args.noise_reduction

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Используется устройство: {device.upper()}")
    print(f"🔊 Шумоподавление: {'ВКЛ' if NOISE_REDUCTION else 'ВЫКЛ'}")

    print(f"⏳ Загрузка модели Whisper {args.model}...")
    model = whisper.load_model(args.model, device=device)
    print(f"✅ Модель {args.model} загружена на {device.upper()}")

    stt = ContinuousSTT(
        model=model,
        silence_timeout=SILENCE_TIMEOUT_SEC,
        min_speech_duration=MIN_SPEECH_DURATION_SEC
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