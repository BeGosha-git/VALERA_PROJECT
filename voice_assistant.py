"""
voice_assistant.py
Голосовой ассистент: микрофон → распознавание речи → RAG-запрос → синтез речи

Требует:
  - Запущенный WebRAgent (docker compose up -d)
  - Запущенный Ollama (ollama serve)
  - NeuralSpeaker из russian_text_to_speech (model.pt, NeuralSpeaker.py)

Зависимости: mic_to_text/environment.yml
"""
import sys
import os
import time
import queue
import threading
import tempfile
import subprocess
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from silero_vad import load_silero_vad
import requests
from bs4 import BeautifulSoup

# Пути к подпроектам (файл лежит в корне проекта)

from russian_text_to_speech.NeuralSpeaker import NeuralSpeaker
from kairos_asr import KairosASR

# ------------------------ Конфигурация ------------------------
SAMPLE_RATE = 16000
SILENCE_TIMEOUT_SEC = 0.75
MIN_SPEECH_DURATION_SEC = 0.5
VAD_THRESHOLD = 0.3          # порог VAD
NOISE_REDUCTION = True
NOISE_PROFILE_DURATION = 1.5

# Устройство захвата. 'default' идёт через PipeWire/Pulse (умеет ресемплинг).
# Для конкретного микрофона можно указать, например:
#   plughw:CARD=MINI,DEV=0   (DJI MIC MINI)
#   hw:4,0
# Посмотреть доступные:  arecord -L
AUDIO_DEVICE = "default"

# RAG-клиент
RAG_BASE_URL = "http://localhost:5000"
RAG_USERNAME = "admin"
RAG_PASSWORD = "change_me_in_production"

# TTS
TTS_SPEAKER = "eugene"
TTS_SAMPLE_RATE = 48000
# --------------------------------------------------------------


class SileroVAD:
    """Детектор голосовой активности на базе Silero VAD."""

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.model = load_silero_vad(onnx=True)

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        audio_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
        if audio_tensor.shape[1] < 512:
            audio_tensor = torch.nn.functional.pad(
                audio_tensor, (0, 512 - audio_tensor.shape[1])
            )
        prob = self.model(audio_tensor, self.sample_rate).item()
        return prob > self.threshold


# ---------------------------------------------------------------------------
# STT-модели
# ---------------------------------------------------------------------------

# Путь к Nemotron-модели (папка с config.json / model.safetensors)
NEMOTRON_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "nemotron-project"
)
# Язык для Nemotron: ru-RU, en-US, auto и т.д.
NEMOTRON_LANGUAGE = "ru-RU"
# Задержка стриминга (мс): 80 / 320 / 560 / 1120
NEMOTRON_LATENCY_MS = 320


class NemotronSTT:
    """
    Распознавание речи через NVIDIA Nemotron-3.5-ASR-Streaming-0.6B
    (Transformers, локальная папка nemotron-project/).

    API совместим с KairosASR: transcribe(wav_file=..., sample_rate=...).
    Модель грузится один раз и переиспользуется для всех фраз.
    """

    def __init__(
        self,
        model_path: str = NEMOTRON_MODEL_PATH,
        language: str = NEMOTRON_LANGUAGE,
        latency_ms: int = NEMOTRON_LATENCY_MS,
        device: str = "auto",
    ):
        from transformers import AutoModelForRNNT, AutoProcessor

        self.language = language
        # lookahead-токены под нужную задержку
        lookahead_map = {80: 0, 320: 3, 560: 6, 1120: 13}
        num_tokens = lookahead_map.get(latency_ms, 3)

        print(f"[Nemotron-STT] Загрузка модели из: {model_path}")
        print(f"[Nemotron-STT] Язык: {language}  |  Задержка: {latency_ms} мс")

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.processor.set_num_lookahead_tokens(num_tokens)

        self.model = AutoModelForRNNT.from_pretrained(
            model_path,
            device_map=device,
            dtype=torch.float16,
            trust_remote_code=True,
        )
        self.model.eval()

        self.sampling_rate = self.processor.feature_extractor.sampling_rate
        self.samples_first_chunk = self.processor.num_samples_first_audio_chunk
        self.samples_per_chunk = self.processor.num_samples_per_audio_chunk
        self.hop_length = self.processor.feature_extractor.hop_length
        self.n_fft = self.processor.feature_extractor.n_fft
        print(f"[Nemotron-STT] Готов (устройство: {self.model.device}).")

    def _move_to_device(self, inputs: dict):
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

    def _make_feature_generator(self, audio: np.ndarray, first_inputs: dict):
        """Генератор mel-фич для стримингового RNNT."""
        device = self.model.device
        yield first_inputs["input_features"][
            :, : self.processor.num_mel_frames_first_audio_chunk, :
        ]

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
            yield self._move_to_device(inputs)["input_features"]
            mel_frame_idx += self.processor.num_mel_frames_per_audio_chunk
            start_idx = mel_frame_idx * self.hop_length - self.n_fft // 2

    def transcribe(self, wav_file: str, sample_rate: int = None) -> str:
        """
        Транскрибирует WAV-файл и возвращает текст.
        Сигнатура совместима с KairosASR.transcribe(wav_file=...).
        """
        from transformers import TextIteratorStreamer

        audio, sr = sf.read(wav_file, dtype="float32", always_2d=False)
        if sr != self.sampling_rate:
            import librosa

            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=self.sampling_rate
            )
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # Паддим, если короче первого чанка
        if len(audio) < self.samples_first_chunk:
            padded = np.zeros(self.samples_first_chunk, dtype=np.float32)
            padded[: len(audio)] = audio
            audio = padded

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
        thread = threading.Thread(
            target=self.model.generate, kwargs=generate_kwargs, daemon=True
        )
        thread.start()

        parts = []
        for text_chunk in streamer:
            parts.append(text_chunk)
        thread.join(timeout=10)
        return "".join(parts).strip()


class RAGClient:
    """
    Клиент для взаимодействия с WebRAgent RAG API.
    (копия из TALK.py для независимости)
    """

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        login_endpoint: str = "/auth/login",
        use_csrf: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.login_endpoint = login_endpoint
        self.use_csrf = use_csrf
        self.session = requests.Session()
        self.session.headers.update({"X-Requested-With": "XMLHttpRequest"})
        self._logged_in = False

    def login(self) -> bool:
        login_url = f"{self.base_url}{self.login_endpoint}"
        csrf_token = None

        if self.use_csrf:
            try:
                resp = self.session.get(login_url)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, "html.parser")
                    token_input = soup.find("input", {"name": "csrf_token"})
                    if token_input and token_input.get("value"):
                        csrf_token = token_input["value"]
                    else:
                        token_input = soup.find("input", {"name": "_csrf_token"})
                        if token_input and token_input.get("value"):
                            csrf_token = token_input["value"]
            except Exception as e:
                print(f"⚠️ Ошибка CSRF: {e}", file=sys.stderr)

        data = {"username": self.username, "password": self.password}
        if csrf_token:
            data["csrf_token"] = csrf_token

        try:
            response = self.session.post(login_url, data=data, allow_redirects=False)
            if response.status_code in (200, 302):
                self._logged_in = True
                return True
            else:
                print(f"❌ Ошибка входа: статус {response.status_code}", file=sys.stderr)
                return False
        except requests.RequestException as e:
            print(f"❌ Ошибка соединения при входе: {e}", file=sys.stderr)
            return False

    def query(
        self,
        query_text: str,
        collection_id: Optional[str] = None,
        use_agent_search: bool = False,
        use_web_search: bool = False,
        max_results: int = 4,
        conversation_context: Optional[list] = None,
    ) -> dict:
        if not self._logged_in:
            raise RuntimeError("Необходимо сначала выполнить login()")
        if not query_text:
            raise ValueError("query_text не может быть пустым")
        if not use_web_search and not collection_id:
            raise ValueError("collection_id обязателен, если use_web_search=False")

        data = {
            "query": query_text,
            "use_agent_search": "on" if use_agent_search else "",
            "use_web_search": "on" if use_web_search else "",
            "use_deep_search": "",
            "agent_strategy": "direct",
            "max_results": str(max(1, min(max_results, 10))),
        }
        if collection_id:
            data["collection_id"] = collection_id
        if conversation_context:
            import json
            data["conversation_context"] = json.dumps(conversation_context)

        query_url = f"{self.base_url}/query"
        try:
            response = self.session.post(query_url, data=data)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ошибка запроса: статус {response.status_code}, ответ: {response.text}"
                )
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Ошибка соединения при выполнении запроса: {e}")

    def get_collections(self):
        """Получает список коллекций (как в TALK.py)."""
        if not self._logged_in:
            raise RuntimeError("Необходимо сначала выполнить login()")

        url = f"{self.base_url}/collections"
        try:
            response = self.session.get(url)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ошибка получения коллекций: статус {response.status_code}"
                )
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Ошибка соединения при получении коллекций: {e}")

    def close(self):
        self.session.close()


def clean_html(html_text: Optional[str]) -> str:
    """Извлекает чистый текст из HTML-ответа RAG."""
    if not html_text:
        return "Ответ отсутствует"
    soup = BeautifulSoup(html_text, "html.parser")
    return soup.get_text(separator=" ", strip=True)


# Стебли слов, которые надо удалять из текста вопроса (для теста)
# "валер" покрывает: Валера, Валеры, Валера, Валерий, Валерка...
# "робот" покрывает: робот, робота, роботу, роботы, роботе...
FILTER_STEMS = [
    "валер",
]
# Мат (нецензурная лексика) — стебли для удаления
PROFANITY_STEMS = [
    "бля", "сук", "пидор", "ублюд", "ебан", "ебал", "ебат", "ебли",
    "хуй", "хуя", "хуе", "пизд", "наху", "ебучи", "мудак", "мудил",
    "дерьм", "говн", "заебал", "несук", "трах", "срань", "гандон",
    "шлюх", "проститут", "сволоч", "тварь",
]


def clean_question(text: str) -> str:
    """Убирает из текста имена/слова и мат (по стеблям)."""
    import re

    cleaned = text
    for stem in PROFANITY_STEMS + FILTER_STEMS:
        cleaned = re.sub(rf"\b{stem}\w*", "", cleaned, flags=re.IGNORECASE)
    # Схлопываем лишние пробелы
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_yes_answer(answer: str) -> bool:
    """Определяет, ответил ли пользователь 'да' (только разрешённые фразы)."""
    import re

    a = answer.lower().strip()
    # Многословные фразы
    for phrase in ("так точно", "хотел бы"):
        if phrase in a:
            return True
    # Одиночные слова по границам
    return bool(re.search(r"\b(да|конечно|ага|хочу)\b", a))


# Слова, близкие к аббревиатуре МИРЭА, которые надо заменять на "МИРЭА"
# (включая формы склонения "мир": мир, мира, миру, мире, миры, миров, мирами, мирах)
MIREA_ALIASES = [
    "мир", "мира", "миру", "мире", "миры", "миров", "мирами", "мирах",
    "мирэ", "мирэа",
]

# Ключевые слова, при которых поиск идёт по коллекции (а не по интернету).
# Включает МИРЭА-алиасы + дополнительные слова, связанные с РТУ МИРЭА.
COLLECTION_KEYWORDS = MIREA_ALIASES + [
    "мегалаборатория", "мегалаборатории", "мегалабораторий",
    "лаборатория", "лаборатории", "лабораторий",
    "институт", "института", "институте",
    "вуз", "вуза", "вузе", "вузов",
    "испытания", "испытаний", "испытаниях",
    "учитесь", "учёба", "учёбы", "учеба", "учебы",
    "тхт",
    "стромынка", "стромынки",
    "вернадка", "вернадки", "вернадку", "вернадке", "вернадкой",
    "рту",
]


def normalize_mirea(text: str) -> str:
    """Заменяет слова близкие к МИРЭА на 'МИРЭА' (по границам слов)."""
    import re

    pattern = r"\b(" + "|".join(MIREA_ALIASES) + r")\b"
    return re.sub(pattern, "МИРЭА", text, flags=re.IGNORECASE)


def is_mirea_related(text: str) -> bool:
    """Есть ли в тексте упоминание МИРЭА или ключевых слов коллекции."""
    import re

    pattern = r"\b(" + "|".join(COLLECTION_KEYWORDS) + r")\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def arecord_command(extra_args=None):
    """Базовая команда захвата через arecord (16 кГц, моно, S16_LE, raw)."""
    cmd = [
        "arecord", "-D", AUDIO_DEVICE,
        "-r", str(SAMPLE_RATE),
        "-c", "1",
        "-f", "S16_LE",
        "-t", "raw",
        "-q",
    ]
    if extra_args:
        cmd += extra_args
    return cmd


class VoiceAssistant:
    """
    Голосовой ассистент:
    микрофон → STT → RAG → TTS → динамики

    Захват через `arecord` (ALSA/PipeWire) — это надёжнее, чем sounddevice,
    т.к. корректно работает с USB-микрофонами (DJI) через PipeWire.
    VAD выполняется в потоке захвата, тяжёлая работа (STT/RAG/TTS) — в фоне.
    """

    def __init__(
        self,
        rag_client: RAGClient,
        neural_speaker: NeuralSpeaker,
        vad: SileroVAD,
        silence_timeout: float = 0.75,
        min_speech_duration: float = 0.5,
        no_confirm: bool = False,
    ):
        self.rag = rag_client
        self.tts = neural_speaker
        self.vad = vad
        self._vad_model = vad.model  # прямой доступ к ONNX-модели (быстрее)
        self.silence_timeout = silence_timeout
        self.min_speech_duration = min_speech_duration
        self._no_confirm = no_confirm

        self.is_recording = False
        self.noise_profile: Optional[np.ndarray] = None
        self._capture_proc: Optional[subprocess.Popen] = None
        self._vad_buffer = np.array([], dtype=np.float32)
        self._collection_id: Optional[str] = None  # кэш ID коллекции для МИРЭА

        # История последних 10 обменов (user/assistant) для контекста ИИ
        self._conversation_history: list = []

        # Флаг занятости: пока идёт обработка (STT/RAG/TTS),
        # речь с микрофона игнорируется и новые запросы не отправляются
        self._busy = False

        # Состояние VAD (только из потока захвата)
        self._current_phrase: list = []
        self._last_speech_time = 0.0
        self._speech_started = False

        # Очередь готовых фраз (16кГц аудио) для обработки в фоне
        self._phrase_queue: queue.Queue = queue.Queue()

        # Блокировка, чтобы не обрабатывать параллельно несколько фраз
        self._processing_lock = threading.Lock()

    # ---- VAD по приходящим PCM-байтам ----
    def _feed_pcm(self, data: bytes):
        # Пока идёт обработка — игнорируем всю речь с микрофона
        if self._busy:
            self._vad_buffer = np.array([], dtype=np.float32)
            self._current_phrase = []
            self._speech_started = False
            return

        # int16 LE → float32 [-1, 1]
        a = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        self._vad_buffer = np.concatenate((self._vad_buffer, a))

        while len(self._vad_buffer) >= 512:
            chunk = self._vad_buffer[:512]
            self._vad_buffer = self._vad_buffer[512:]

            audio_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
            prob = self._vad_model(audio_tensor, SAMPLE_RATE).item()
            is_speech = prob > VAD_THRESHOLD
            current_time = time.time()

            if is_speech:
                if not self._speech_started:
                    self._speech_started = True
                    self._current_phrase = []
                self._current_phrase.append(chunk)
                self._last_speech_time = current_time
            else:
                if self._speech_started:
                    if current_time - self._last_speech_time >= self.silence_timeout:
                        self._finalize_phrase()
                        self._speech_started = False
                        self._current_phrase = []

    # ---- поток захвата: читает stdout arecord и кормит VAD ----
    def _capture_loop(self):
        try:
            self._capture_proc = subprocess.Popen(
                arecord_command(),
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except Exception as e:
            print(f"❌ Не удалось запустить arecord: {e}", file=sys.stderr)
            self.is_recording = False
            return

        chunk_bytes = 512 * 2  # 512 сэмплов * 2 байта (S16_LE)
        while self.is_recording:
            data = self._capture_proc.stdout.read(chunk_bytes)
            if not data:
                break
            try:
                self._feed_pcm(data)
            except Exception as e:
                print(f"❌ Ошибка VAD: {e}", file=sys.stderr)

        # Добиваем остаток при остановке
        if self._speech_started:
            self._finalize_phrase()
        if self._capture_proc:
            try:
                self._capture_proc.terminate()
            except Exception:
                pass

    def _finalize_phrase(self):
        if not self._current_phrase:
            return
        audio_np = np.concatenate(self._current_phrase)
        duration = len(audio_np) / SAMPLE_RATE
        self._current_phrase = []
        if duration < self.min_speech_duration:
            return
        try:
            self._phrase_queue.put_nowait(audio_np)
        except queue.Full:
            pass  # пропускаем, если очередь забита

    # ---- фоновый воркер: забирает готовые фразы и обрабатывает ----
    def _phrase_worker(self):
        while self.is_recording:
            try:
                audio_np = self._phrase_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._busy:
                continue  # не обрабатываем новые фразы, пока заняты
            self._handle_phrase(audio_np)

    def _handle_phrase(self, audio_np: np.ndarray):
        """Полный пайплайн: Шумоподавление → STT → RAG → TTS"""
        self._busy = True
        try:
            self._handle_phrase_inner(audio_np)
        finally:
            self._busy = False

    def _handle_phrase_inner(self, audio_np: np.ndarray):
        with self._processing_lock:
            # 1. Шумоподавление
            if NOISE_REDUCTION and self.noise_profile is not None and len(self.noise_profile) > 0:
                try:
                    import noisereduce as nr

                    audio_np = nr.reduce_noise(
                        y=audio_np,
                        sr=SAMPLE_RATE,
                        y_noise=self.noise_profile,
                        prop_decrease=0.8,
                        n_fft=1024,
                    )
                except ImportError:
                    pass

            # 2. Сохраняем во временный WAV
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp:
                    tmp_path = tmp.name
                sf.write(tmp_path, audio_np, SAMPLE_RATE, subtype="PCM_16")

                # 3. Распознаём речь (STT)
                from kairos_asr import KairosASR

                asr = KairosASR(device="auto")
                result = asr.transcribe(wav_file=tmp_path)
                text = result.full_text.strip()
            except Exception as e:
                print(f"❌ Ошибка распознавания речи: {e}", file=sys.stderr)
                return
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

            if not text:
                print("🔇 (пусто)")
                return

            # Чистим текст вопроса (убираем имена/мат) для теста
            clean_text = clean_question(text)
            if clean_text != text:
                print(f"🧹 После очистки: {clean_text}")

            if not clean_text:
                print("🚫 Текст пустой после очистки.")
                return

            # Нормализация МИРЭА: близкие слова → "МИРЭА"
            query_text = normalize_mirea(clean_text)
            mirea_related = is_mirea_related(clean_text)
            if query_text != clean_text:
                print(f"🔄 МИРЭА-нормализация: {query_text}")

            # Подтверждение: спрашиваем, хочет ли пользователь узнать об этом
            if not self._no_confirm:
                if not self._confirm(query_text):
                    print("⏭️  Пользователь отказался — не отправляю в ИИ.\n")
                    return

            # 4. Отправляем запрос в RAG (коллекция для МИРЭА, иначе интернет)
            try:
                if mirea_related:
                    collection_id = self._get_collection_id()
                    if collection_id:
                        print(f"📚 Поиск по коллекции (МИРЭА): {collection_id}")
                        rag_result = self.rag.query(
                            query_text=query_text,
                            collection_id=collection_id,
                            use_web_search=False,
                            use_agent_search=False,
                            max_results=3,
                            conversation_context=self._conversation_history,
                        )
                    else:
                        print("⚠️ Нет коллекций — ищу в интернете.")
                        rag_result = self.rag.query(
                            query_text=query_text,
                            use_web_search=True,
                            use_agent_search=False,
                            max_results=3,
                            conversation_context=self._conversation_history,
                        )
                else:
                    rag_result = self.rag.query(
                        query_text=query_text,
                        use_web_search=True,
                        use_agent_search=True,
                        max_results=5,
                        conversation_context=self._conversation_history,
                    )
                answer = clean_html(rag_result.get("response"))
                # Обрезаем, чтобы не было слишком длинно для TTS
                answer = answer[:700]
                print(f"🤖 Ответ RAG: {answer[:200]}...")
            except Exception as e:
                print(f"❌ Ошибка RAG-запроса: {e}", file=sys.stderr)
                self.tts.speak(
                    "Извините, произошла ошибка при обработке запроса.",
                    speaker=TTS_SPEAKER,
                    save_file=False,
                    sample_rate=TTS_SAMPLE_RATE,
                )
                return

            if not answer or answer == "Ответ отсутствует":
                answer = "К сожалению, я не смог найти ответ на ваш вопрос."

            # Сохраняем обмен в историю (держим последние 10)
            self._conversation_history.append(
                {"role": "user", "content": query_text}
            )
            self._conversation_history.append(
                {"role": "assistant", "content": answer}
            )
            self._conversation_history = self._conversation_history[-20:]  # 10 обменов

            # 5. Озвучиваем ответ
            print("🔊 Озвучиваю ответ...")
            try:
                self.tts.speak(
                    answer,
                    speaker=TTS_SPEAKER,
                    save_file=False,
                    sample_rate=TTS_SAMPLE_RATE,
                )
            except Exception as e:
                print(f"❌ Ошибка синтеза речи: {e}", file=sys.stderr)

            print("✅ Готово. Слушаю дальше...\n")

    def _record_noise_profile(self) -> np.ndarray:
        """Записывает шумовой профиль через arecord."""
        proc = subprocess.run(
            arecord_command(["-d", str(int(NOISE_PROFILE_DURATION))]),
            capture_output=True,
        )
        data = proc.stdout
        if not data:
            raise RuntimeError("arecord вернул пустые данные")
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    def _record_answer(self, seconds: float = 3.0) -> str:
        """Записывает ответ пользователя (да/нет) и распознаёт его."""
        from kairos_asr import KairosASR

        proc = subprocess.run(
            arecord_command(["-d", str(int(max(seconds, 1)))]),
            capture_output=True,
        )
        data = proc.stdout
        if not data:
            return ""
        a = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            sf.write(tmp_path, a, SAMPLE_RATE, subtype="PCM_16")
            asr = KairosASR(device="auto")
            result = asr.transcribe(wav_file=tmp_path)
            return result.full_text.strip()
        except Exception as e:
            print(f"❌ Ошибка распознавания ответа: {e}", file=sys.stderr)
            return ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _confirm(self, question: str) -> bool:
        """Спрашивает подтверждение и возвращает True/False (ответ да/нет)."""
        clean_question_for_speech = clean_question(question)
        if not clean_question_for_speech:
            clean_question_for_speech = "этот вопрос"
        self.tts.speak(
            f"Вы хотели бы узнать об {clean_question_for_speech}?",
            speaker=TTS_SPEAKER,
            save_file=False,
            sample_rate=TTS_SAMPLE_RATE,
        )
        print("🗣️  Подтверждение: 'Вы хотели бы узнать об %s?'" % clean_question_for_speech)
        print("    Скажите ДА или НЕТ...")
        answer = self._record_answer(seconds=3.0)
        print(f"    Ответ: {answer!r}")
        return is_yes_answer(answer)

    def _get_collection_id(self) -> Optional[str]:
        """Возвращает ID первой коллекции (кэшируется)."""
        if self._collection_id is None:
            try:
                collections = self.rag.get_collections()
                if collections:
                    self._collection_id = collections[0]["id"]
            except Exception as e:
                print(f"❌ Не удалось получить коллекции: {e}", file=sys.stderr)
        return self._collection_id

    def start(self):
        if self.is_recording:
            return

        # Шумовой профиль ДО запуска основного захвата
        if NOISE_REDUCTION:
            print(f"🔇 Запись шума в течение {NOISE_PROFILE_DURATION} сек...")
            try:
                self.noise_profile = self._record_noise_profile()
                print("✅ Профиль шума записан")
            except Exception as e:
                self.noise_profile = None
                print(f"⚠️ Не удалось записать шум, шумоподавление отключено: {e}")

        self._vad_buffer = np.array([], dtype=np.float32)
        self._current_phrase = []
        self._speech_started = False
        self._phrase_queue = queue.Queue()

        self.is_recording = True

        # Фоновый воркер для обработки готовых фраз
        self._worker_thread = threading.Thread(target=self._phrase_worker, daemon=True)
        self._worker_thread.start()

        # Поток захвата
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        print("🎙️  Говорите... (пауза 0.75 сек для завершения фразы)")
        print("    Нажмите Ctrl+C для остановки.\n")

    def stop(self):
        self.is_recording = False
        if self._capture_proc is not None:
            try:
                self._capture_proc.terminate()
            except Exception:
                pass
        if hasattr(self, '_capture_thread') and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=3.0)
        if hasattr(self, '_worker_thread') and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)
        print("\n🛑 Остановлено.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Голосовой ассистент (Микрофон → RAG → TTS)")
    parser.add_argument(
        "--no-confirm", action="store_true",
        help="Отключить подтверждение — отправлять запрос сразу в ИИ без вопроса пользователю"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  Голосовой ассистент (Микрофон → RAG → TTS)")
    if args.no_confirm:
        print("  ⚡ Режим: без подтверждения (сразу ответ)")
    print("=" * 50)

    # 1. Инициализация RAG-клиента
    print("\n[1/3] Подключение к WebRAgent...")
    rag = RAGClient(
        base_url=RAG_BASE_URL,
        username=RAG_USERNAME,
        password=RAG_PASSWORD,
    )
    if not rag.login():
        print("❌ Не удалось войти в WebRAgent. Проверьте, что сервис запущен.")
        sys.exit(1)
    print("✅ Подключено к WebRAgent")

    # 2. Инициализация NeuralSpeaker (TTS)
    print("\n[2/3] Загрузка модели синтеза речи (NeuralSpeaker)...")
    neural_speaker = NeuralSpeaker()
    print("✅ Модель TTS загружена")

    # 3. Инициализация VAD
    print("\n[3/3] Инициализация VAD...")
    vad = SileroVAD(sample_rate=SAMPLE_RATE, threshold=VAD_THRESHOLD)
    print("✅ VAD готов")

    # 4. Запуск ассистента
    assistant = VoiceAssistant(
        rag_client=rag,
        neural_speaker=neural_speaker,
        vad=vad,
        silence_timeout=SILENCE_TIMEOUT_SEC,
        min_speech_duration=MIN_SPEECH_DURATION_SEC,
        no_confirm=args.no_confirm,
    )

    try:
        assistant.start()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n")
    finally:
        assistant.stop()
        rag.close()
        print("👋 Выход.")


if __name__ == "__main__":
    main()
