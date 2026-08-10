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
# Все настройки читаются из окружения (.env) с разумными значениями по умолчанию.
# Поддерживается python-dotenv: создайте файл .env рядом со скриптом.

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass  # python-dotenv не установлен — работаем с окружением как есть


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


SAMPLE_RATE = 16000
SILENCE_TIMEOUT_SEC = _env_float("SILENCE_TIMEOUT_SEC", 0.75)
MIN_SPEECH_DURATION_SEC = _env_float("MIN_SPEECH_DURATION_SEC", 0.5)
VAD_THRESHOLD = _env_float("VAD_THRESHOLD", 0.3)          # порог VAD
NOISE_REDUCTION = os.getenv("NOISE_REDUCTION", "true").lower() == "true"
NOISE_PROFILE_DURATION = _env_float("NOISE_PROFILE_DURATION", 1.5)

# Устройство захвата. 'default' идёт через PipeWire/Pulse (умеет ресемплинг).
# Для конкретного микрофона можно указать, например:
#   plughw:CARD=MINI,DEV=0   (DJI MIC MINI)
#   hw:4,0
# Посмотреть доступные:  arecord -L
AUDIO_DEVICE = os.getenv("AUDIO_DEVICE", "default")

# RAG-клиент
RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://localhost:5000")
RAG_USERNAME = os.getenv("RAG_USERNAME", "admin")
RAG_PASSWORD = os.getenv("RAG_PASSWORD", "change_me_in_production")

# Файл, где хранится chat_id WebRAgent.
# Благодаря этому контекст разговора сохраняется между перезапусками ассистента.
CHAT_ID_FILE = os.getenv(
    "CHAT_ID_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_id.txt"),
)

# TTS
TTS_SPEAKER = os.getenv("TTS_SPEAKER", "eugene")
TTS_SAMPLE_RATE = _env_int("TTS_SAMPLE_RATE", 48000)

# Nemotron STT
NEMOTRON_MODEL_PATH = os.getenv(
    "NEMOTRON_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "nemotron-project"),
)
NEMOTRON_LANGUAGE = os.getenv("NEMOTRON_LANGUAGE", "ru-RU")
NEMOTRON_LATENCY_MS = _env_int("NEMOTRON_LATENCY_MS", 320)
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
        mel_frame_idx = self.processor.num_mel_frames_first_audio_chunk
        start_idx = mel_frame_idx * self.hop_length - self.n_fft // 2

        # Первый чанк (уже посчитан в transcribe)
        yield first_inputs["input_features"][
            :, : self.processor.num_mel_frames_first_audio_chunk, :
        ]

        while True:
            end_idx = start_idx + self.samples_per_chunk
            if end_idx > len(audio):
                # Последний неполный чанк — паддим нулями до полного размера,
                # чтобы хвост фразы (последние слова) попал в распознавание.
                if start_idx >= len(audio):
                    break
                chunk = np.zeros(self.samples_per_chunk, dtype=np.float32)
                chunk[: len(audio) - start_idx] = audio[start_idx:len(audio)]
            else:
                chunk = audio[start_idx:end_idx]

            inputs = self.processor(
                chunk,
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
            "max_new_tokens": 1024,
        }
        thread = threading.Thread(
            target=self.model.generate, kwargs=generate_kwargs, daemon=True
        )
        thread.start()

        parts = []
        for text_chunk in streamer:
            parts.append(text_chunk)
        # Ждём, пока генерация действительно закончится (не обрезаем хвост)
        thread.join(timeout=60)
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

    # ------------------------------------------------------------------
    # Чат-API WebRAgent (хранит историю на сервере в MongoDB)
    # ------------------------------------------------------------------
    def create_chat(self) -> Optional[str]:
        """
        Создаёт новый чат на сервере WebRAgent.
        Возвращает chat_id или None при ошибке.
        """
        if not self._logged_in:
            raise RuntimeError("Необходимо сначала выполнить login()")

        url = f"{self.base_url}/chat/new"
        try:
            response = self.session.post(url)
            if response.status_code != 200:
                print(
                    f"❌ Ошибка создания чата: статус {response.status_code}, ответ: {response.text}",
                    file=sys.stderr,
                )
                return None
            data = response.json()
            return data.get("chat_id")
        except requests.RequestException as e:
            print(f"❌ Ошибка соединения при создании чата: {e}", file=sys.stderr)
            return None

    def chat_query(
        self,
        chat_id: str,
        query_text: str,
        collection_id: Optional[str] = None,
        use_agent_search: bool = False,
        use_web_search: bool = False,
        max_results: int = 4,
    ) -> dict:
        """
        Отправляет запрос в существующий чат.
        WebRAgent сам хранит историю и передаёт контекст (последние 10 сообщений).
        """
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
            "agent_strategy": "direct",
            "max_results": str(max(1, min(max_results, 10))),
        }
        if collection_id:
            data["collection_id"] = collection_id

        query_url = f"{self.base_url}/chat/{chat_id}/query"
        try:
            response = self.session.post(query_url, data=data)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ошибка запроса: статус {response.status_code}, ответ: {response.text}"
                )
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Ошибка соединения при выполнении запроса: {e}")

    def get_chat_messages(self, chat_id: str, limit: int = 50):
        """Получает сообщения чата с сервера."""
        if not self._logged_in:
            raise RuntimeError("Необходимо сначала выполнить login()")

        url = f"{self.base_url}/chat/{chat_id}/messages"
        try:
            response = self.session.get(url, params={"limit": limit})
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ошибка получения сообщений чата: статус {response.status_code}"
                )
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Ошибка соединения при получении сообщений: {e}")

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


# ---------------------------------------------------------------------------
# Логирование заданных вопросов
# ---------------------------------------------------------------------------
# Файл журнала вопросов (переопределяется тестами)
LOG_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "questions.log"
)


def log_question(
    raw_text: str = "",
    query_text: str = "",
    mirea: bool = False,
    collection_id: Optional[str] = None,
    web_search: bool = False,
    answer: str = "",
) -> None:
    """
    Записывает заданный вопрос (и ответ) в журнал LOG_FILE.
    Полезно для отладки/аудита того, что реально уходит в ИИ.
    """
    import json
    from datetime import datetime

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "raw_text": raw_text,
        "query_text": query_text,
        "mirea_related": mirea,
        "collection_id": collection_id,
        "web_search": web_search,
        "answer": answer,
    }
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ Не удалось записать в лог: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Обработка ответа через ИИ (пересказ для голосового вывода)
# ---------------------------------------------------------------------------
# Адрес Ollama для локального пересказа ответа. Берём из .env, дефолт 11434.
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
REPHRASE_MODEL = os.getenv("REPHRASE_MODEL", "valera_live:latest")


def rephrase_answer_with_ai(
    question: str,
    raw_answer: str,
    max_words: int = 80,
) -> str:
    """
    Пересказывает «сырой» ответ RAG (со ссылками/сниппетами) через локальную
    модель Ollama в короткий разговорный ответ, пригодный для озвучки.

    Убирает URL, форматирование и избыточность — оставляет суть как живой
    ответ ассистента. Если Ollama недоступна — возвращает исходный текст.
    """
    if not raw_answer or len(raw_answer.strip()) < 5:
        return raw_answer

    # Текущая дата/время — чтобы пересказ не «плавал» в датах
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Europe/Moscow"))
    except Exception:
        now = datetime.now()
    now_str = now.strftime("%d.%m.%Y %H:%M")

    prompt = (
        "Ты — робот Валера. Пользователь спросил: "
        f"\"{question}\"\n\n"
        f"Сегодня {now_str} по московскому времени.\n\n"
        "Ниже приведён черновой ответ из поиска (может содержать ссылки и "
        "служебный текст). Перескажи его КОРОТКО, по-русски, разговорно, "
        f"не более {max_words} слов. НЕ упоминай URL и ссылки, не цитируй "
        "источники. Все английские слова (feelsLike, slight, cloudy и т.п.) "
        "переводи на русский. Просто дай полезный ответ человеку.\n\n"
        f"Черновик: {raw_answer}\n\n"
        "Ответ:"
    )
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": REPHRASE_MODEL,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",  # держим модель в памяти между вызовами
                "options": {"num_predict": 200, "temperature": 0.5},
            },
            timeout=60,
        )
        if resp.status_code != 200:
            return raw_answer
        data = resp.json()
        text = data.get("response", "").strip()
        # Падаем обратно на черновик, если модель вернула пусто
        if not text:
            return raw_answer
        return text[:1000]
    except Exception as e:
        print(f"⚠️ Не удалось пересказать ответ через ИИ: {e}", file=sys.stderr)
        return raw_answer


def warmup_ollama(model: str = REPHRASE_MODEL, timeout: int = 120) -> None:
    """
    Прогревает LLM-модель в Ollama коротким вызовом, чтобы она загрузилась
    в VRAM до первого реального запроса (иначе первый ответ ждёт cold start).
    """
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": "Привет",  # короткий прогрев
                "stream": False,
                "keep_alive": "30m",
                "options": {"num_predict": 5},
            },
            timeout=timeout,
        )
        if resp.status_code == 200:
            print(f"🔥 LLM прогрев: {model} готова")
        else:
            print(f"⚠️ LLM прогрев: статус {resp.status_code}", file=sys.stderr)
    except Exception as e:
        print(f"⚠️ LLM прогрев не удался: {e}", file=sys.stderr)


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
        stt: str = "kairos",
        new_chat: bool = False,
        no_clean: bool = False,
        tts_speaker: str = "eugene",
        vad_threshold: float = 0.3,
        preload: bool = False,
        barge_in: bool = False,
    ):
        self.rag = rag_client
        self.tts = neural_speaker
        self.vad = vad
        self._vad_model = vad.model  # прямой доступ к ONNX-модели (быстрее)
        self.silence_timeout = silence_timeout
        self.min_speech_duration = min_speech_duration
        self._no_confirm = no_confirm
        self._stt = stt
        self._new_chat = new_chat
        self._no_clean = no_clean
        self._tts_speaker = tts_speaker
        self._vad_threshold = vad_threshold
        self._preload = preload
        self._barge_in = barge_in

        # Флаг прерывания озвучки (barge-in): при речи во время ответа
        # озвучка будет прервана, чтобы ассистент сразу слушал новый вопрос.
        self._interrupt_tts = False

        # TTS-поток: при barge-in озвучка идёт в фоне, не блокируя обработку.
        self._tts_thread: Optional[threading.Thread] = None

        # STT-модели: грузим лениво и переиспользуем
        self._asr = None          # выбранная модель (kairos или nemotron)
        self._asr_kairos = None   # отдельная Kairos-модель (для --stt auto)
        self._asr_nemotron = None # отдельная Nemotron-модель (для --stt auto)

        self.is_recording = False
        self.noise_profile: Optional[np.ndarray] = None
        self._capture_proc: Optional[subprocess.Popen] = None
        self._vad_buffer = np.array([], dtype=np.float32)
        self._collection_id: Optional[str] = None  # кэш ID коллекции для МИРЭА

        # Чат WebRAgent: контекст вопросов/ответов хранится на сервере (MongoDB).
        # Создаём чат при старте, далее каждый запрос идёт через /chat/<id>/query,
        # а WebRAgent сам передаёт модель последние сообщения как контекст.
        self._chat_id: Optional[str] = None

        # Локальная история (резерв) — держим последние 10 обменов
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
            is_speech = prob > self._vad_threshold
            current_time = time.time()

            if is_speech:
                # Если идёт озвучка ответа и включён barge-in — прерываем её
                if self._barge_in and self._tts_thread and self._tts_thread.is_alive():
                    self._interrupt_tts = True
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

    def _get_asr(self):
        """
        Возвращает STT-модель согласно self._stt.
        Грузится лениво при первом использовании.
        """
        if self._asr is not None:
            return self._asr

        if self._stt in ("nemotron", "auto"):
            print("[STT] Инициализация Nemotron-модели...")
            self._asr_nemotron = NemotronSTT(
                model_path=NEMOTRON_MODEL_PATH,
                language=NEMOTRON_LANGUAGE,
                latency_ms=NEMOTRON_LATENCY_MS,
            )
            if self._stt == "nemotron":
                self._asr = self._asr_nemotron
        if self._stt in ("kairos", "auto"):
            if self._asr_kairos is None:
                print("[STT] Инициализация Kairos-модели...")
                from kairos_asr import KairosASR

                self._asr_kairos = KairosASR(device="auto")
            if self._stt == "kairos":
                self._asr = self._asr_kairos
        return self._asr

    def _warmup_all(self):
        """
        Полный прогрев всех ИИ-компонентов перед началом работы:
        STT (Nemotron/Kairos) + LLM (Ollama). Вызывается всегда при старте,
        чтобы первый вопрос не ждал загрузку моделей в память/VRAM.
        """
        print("🔥 Полный прогрев ИИ-моделей...")
        # 1. Прогрев STT (та, что будет использоваться)
        try:
            self._get_asr()
            print("   ✅ STT готова")
        except Exception as e:
            print(f"   ⚠️ Прогрев STT не удался: {e}", file=sys.stderr)

        # 2. Прогрев LLM (Ollama)
        try:
            warmup_ollama(model=REPHRASE_MODEL)
            print("   ✅ LLM готова")
        except Exception as e:
            print(f"   ⚠️ Прогрев LLM не удался: {e}", file=sys.stderr)

        print("🔥 Прогрев завершён.")

    def _select_stt(self, audio_np: np.ndarray) -> str:
        """
        При --stt auto выбирает модель по длине фразы:
        короткие (<=2с) — Kairos (быстро), длинные — Nemotron (точнее).
        Иначе возвращает выбранную модель как есть.
        """
        if self._stt != "auto":
            return self._stt
        duration = len(audio_np) / SAMPLE_RATE
        return "kairos" if duration <= 2.0 else "nemotron"

    @staticmethod
    def _should_use_agent_search(text: str) -> bool:
        """
        Эвристика сложности вопроса для выбора между быстрым веб-поиском
        и агентным (декомпозиция на подзапросы).

        Простые вопросы (короткие, <=5 слов, без перечислений и сравнений)
        → False (быстрый поиск, 1 вызов LLM).
        Сложные (длинные, с деталями, перечислениями, сравнениями) → True.
        """
        import re

        words = text.split()
        word_count = len(words)

        # Длинные вопросы с деталями — агентный поиск
        if word_count >= 8:
            return True

        # Вопросы со сравнением / перечислением / уточнением — агентный
        complex_markers = [
            "сравни", "отличи", "разниц", "лучше", "хуже", "чем",
            "а также", "и что", "подробн", "детальн", "объясни", "почему",
            "как работает", "что будет", "в чём разница",
            "первое", "второе", "во-первых", "во-вторых", "узнай", "найди"
        ]
        low = text.lower()
        if any(m in low for m in complex_markers):
            return True

        # Всё остальное (короткие и средние вопросы) — быстрый поиск
        return False

    def _handle_phrase(self, audio_np: np.ndarray):
        """Полный пайплайн: Шумоподавление → STT → RAG → TTS"""
        self._busy = True
        try:
            self._handle_phrase_inner(audio_np)
        finally:
            self._busy = False

    def _speak(self, text: str) -> None:
        """
        Озвучивает текст. При --barge-in — в отдельном потоке,
        чтобы можно было прервать озвучку новым вопросом.
        """
        if not text:
            return

        def _do_speak():
            try:
                self.tts.speak(
                    text,
                    speaker=self._tts_speaker,
                    save_file=False,
                    sample_rate=TTS_SAMPLE_RATE,
                )
            except Exception as e:
                print(f"❌ Ошибка синтеза речи: {e}", file=sys.stderr)
            finally:
                self._interrupt_tts = False

        if self._barge_in:
            # Ждём завершения предыдущей озвучки (быстрое прерывание флагом)
            if self._tts_thread and self._tts_thread.is_alive():
                self._interrupt_tts = True
                self._tts_thread.join(timeout=2.0)
            self._interrupt_tts = False
            self._tts_thread = threading.Thread(target=_do_speak, daemon=True)
            self._tts_thread.start()
        else:
            _do_speak()

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

            # 2. Нормализация громкости (RMS) для стабильного распознавания
            peak = np.abs(audio_np).max()
            if peak > 0 and peak < 1.0:
                audio_np = audio_np / peak * 0.9

            # 2. Сохраняем во временный WAV
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp:
                    tmp_path = tmp.name
                sf.write(tmp_path, audio_np, SAMPLE_RATE, subtype="PCM_16")

                # 3. Распознаём речь (STT)
                # Для --stt auto выбираем модель по длине фразы
                chosen = self._select_stt(audio_np)
                if chosen == "nemotron":
                    # Убеждаемся, что Nemotron загружена
                    if self._asr_nemotron is None:
                        print("[STT] Инициализация Nemotron-модели (auto)...")
                        self._asr_nemotron = NemotronSTT(
                            model_path=NEMOTRON_MODEL_PATH,
                            language=NEMOTRON_LANGUAGE,
                            latency_ms=NEMOTRON_LATENCY_MS,
                        )
                    text = self._asr_nemotron.transcribe(wav_file=tmp_path)
                else:
                    # Kairos (по умолчанию или авто для коротких фраз)
                    if self._asr_kairos is None:
                        print("[STT] Инициализация Kairos-модели...")
                        from kairos_asr import KairosASR

                        self._asr_kairos = KairosASR(device="auto")
                    result = self._asr_kairos.transcribe(wav_file=tmp_path)
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

            if self._no_clean:
                # --- Режим БЕЗ очистки слов: используем распознанный текст как есть ---
                clean_text = text
                print(f"🗣️ (без очистки) Текст: {clean_text}")
            else:
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
                        rag_result = self.rag.chat_query(
                            chat_id=self._chat_id,
                            query_text=query_text,
                            collection_id=collection_id,
                            use_web_search=False,
                            use_agent_search=False,
                            max_results=3,
                        )
                    else:
                        print("⚠️ Нет коллекций — ищу в интернете.")
                        rag_result = self.rag.chat_query(
                            chat_id=self._chat_id,
                            query_text=query_text,
                            use_web_search=True,
                            use_agent_search=False,
                            max_results=3,
                        )
                else:
                    # Адаптивно: простые вопросы — быстрый веб-поиск без декомпозиции,
                    # сложные (многословные / многосторонние) — агентный поиск.
                    use_agent = self._should_use_agent_search(query_text)
                    if use_agent:
                        print("🔍 Агентный поиск (сложный вопрос)...")
                    rag_result = self.rag.chat_query(
                        chat_id=self._chat_id,
                        query_text=query_text,
                        use_web_search=True,
                        use_agent_search=use_agent,
                        max_results=5 if use_agent else 5,
                    )
                answer = clean_html(rag_result.get("response"))
                # Пересказываем ответ через ИИ, чтобы озвучивать живой текст,
                # а не сырые ссылки/сниппеты из поиска.
                answer_ai = rephrase_answer_with_ai(query_text, answer)
                if answer_ai and answer_ai.strip():
                    answer = answer_ai
                # Обрезаем, чтобы не было слишком длинно для TTS
                answer = answer[:999]
                print(f"🤖 Ответ RAG: {answer[:200]}...")
            except Exception as e:
                print(f"❌ Ошибка RAG-запроса: {e}", file=sys.stderr)
                self._speak("Извините, произошла ошибка при обработке запроса.")
                return

            if not answer or answer == "Ответ отсутствует":
                answer = "К сожалению, я не смог найти ответ на ваш вопрос."

            # Логируем заданный вопрос и ответ (для отладки/аудита)
            log_question(
                raw_text=text,
                query_text=query_text,
                mirea=mirea_related,
                collection_id=self._collection_id if mirea_related else None,
                web_search=not mirea_related,
                answer=answer,
            )

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
            self._speak(answer)

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
            # Для короткого ответа (да/нет) всегда используем Kairos (быстро)
            if self._asr_kairos is None:
                print("[STT] Инициализация Kairos-модели (ответ)...")
                from kairos_asr import KairosASR

                self._asr_kairos = KairosASR(device="auto")
            result = self._asr_kairos.transcribe(wav_file=tmp_path)
            return result.full_text.strip()
        except Exception as e:
            print(f"❌ Ошибка распознавания ответа: {e}", file=sys.stderr)
            return ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _confirm(self, question: str) -> bool:
        """Спрашивает подтверждение и возвращает True/False (ответ да/нет)."""
        if self._no_clean:
            clean_question_for_speech = question
        else:
            clean_question_for_speech = clean_question(question)
        if not clean_question_for_speech:
            clean_question_for_speech = "этот вопрос"
        self.tts.speak(
            f"Вы хотели бы узнать об {clean_question_for_speech}?",
            speaker=self._tts_speaker,
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

    def _load_chat_id(self) -> Optional[str]:
        """Загружает сохранённый chat_id из файла."""
        try:
            if os.path.exists(CHAT_ID_FILE):
                with open(CHAT_ID_FILE, "r", encoding="utf-8") as f:
                    cid = f.read().strip()
                    return cid if cid else None
        except Exception as e:
            print(f"⚠️ Не удалось прочитать {CHAT_ID_FILE}: {e}", file=sys.stderr)
        return None

    def _save_chat_id(self, chat_id: str) -> None:
        """Сохраняет chat_id в файл для продолжения контекста между запусками."""
        try:
            with open(CHAT_ID_FILE, "w", encoding="utf-8") as f:
                f.write(chat_id)
        except Exception as e:
            print(f"⚠️ Не удалось сохранить chat_id: {e}", file=sys.stderr)

    def _clear_chat_id_file(self) -> None:
        """Удаляет файл chat_id (для принудительного нового чата)."""
        try:
            if os.path.exists(CHAT_ID_FILE):
                os.remove(CHAT_ID_FILE)
        except Exception as e:
            print(f"⚠️ Не удалось удалить {CHAT_ID_FILE}: {e}", file=sys.stderr)

    def start(self):
        if self.is_recording:
            return

        # Используем существующий чат (контекст сохраняется между запусками)
        # или создаём новый, если его нет / запрошен новый.
        if not self._chat_id:
            if self._new_chat:
                self._clear_chat_id_file()
            saved = self._load_chat_id()
            if saved and not self._new_chat:
                self._chat_id = saved
                print(f"💬 Продолжаю существующий чат: {self._chat_id}")
            else:
                print("💬 Создание нового чата в WebRAgent...")
                self._chat_id = self.rag.create_chat()
                if self._chat_id:
                    self._save_chat_id(self._chat_id)
                    print(f"✅ Чат создан: {self._chat_id}")
                else:
                    print("⚠️ Не удалось создать чат — контекст разговора работать не будет.")

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

        # Полный прогрев всех ИИ-моделей (STT + LLM) в фоне — всегда при старте,
        # чтобы первый вопрос не ждал загрузку моделей в память/VRAM.
        preload_thread = threading.Thread(target=self._warmup_all, daemon=True)
        preload_thread.start()

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
    parser.add_argument(
        "--stt", choices=["kairos", "nemotron", "auto"], default="kairos",
        help="Модель распознавания речи: kairos (по умолчанию), nemotron (NVIDIA), auto (выбор по длине фразы)"
    )
    parser.add_argument(
        "--new-chat", action="store_true",
        help="Начать новый чат (забыть предыдущий контекст разговора)"
    )
    parser.add_argument(
        "--no-clean", action="store_true",
        help="НЕ чистить текст вопроса (не удалять имена и мат) перед отправкой в ИИ"
    )
    parser.add_argument(
        "--vad-threshold", type=float, default=VAD_THRESHOLD,
        help="Порог VAD (0..1, ниже = чувствительнее к тишине/шуму)"
    )
    parser.add_argument(
        "--silence-timeout", type=float, default=SILENCE_TIMEOUT_SEC,
        help="Пауза тишины перед завершением фразы (сек)"
    )
    parser.add_argument(
        "--min-speech", type=float, default=MIN_SPEECH_DURATION_SEC,
        help="Минимальная длительность фразы (сек)"
    )
    parser.add_argument(
        "--speaker", default=TTS_SPEAKER,
        help="Голос TTS (eugene, xenia, aidar, baya, kseniya)"
    )
    parser.add_argument(
        "--preload", action="store_true",
        help="Прогреть STT-модель при старте (первая фраза быстрее)"
    )
    parser.add_argument(
        "--barge-in", action="store_true",
        help="Прерывать озвучку ответа, если пользователь начал говорить"
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  Голосовой ассистент (Микрофон → RAG → TTS)")
    print(f"  STT-модель: {args.stt}")
    print(f"  VAD-порог: {args.vad_threshold} | Таймаут: {args.silence_timeout}с | Голос: {args.speaker}")
    if args.no_confirm:
        print("  ⚡ Режим: без подтверждения (сразу ответ)")
    if args.new_chat:
        print("  🆕 Новый чат (контекст сброшен)")
    if args.no_clean:
        print("  🧽 Без очистки слов (имя/мат сохраняются в вопросе)")
    if args.preload:
        print("  🔥 Прогрев модели при старте")
    if args.barge_in:
        print("  ⏭️ Barge-in: озвучку можно прервать речью")
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
    vad = SileroVAD(sample_rate=SAMPLE_RATE, threshold=args.vad_threshold)
    print("✅ VAD готов")

    # 4. Запуск ассистента
    assistant = VoiceAssistant(
        rag_client=rag,
        neural_speaker=neural_speaker,
        vad=vad,
        silence_timeout=args.silence_timeout,
        min_speech_duration=args.min_speech,
        no_confirm=args.no_confirm,
        stt=args.stt,
        new_chat=args.new_chat,
        no_clean=args.no_clean,
        tts_speaker=args.speaker,
        vad_threshold=args.vad_threshold,
        preload=args.preload,
        barge_in=args.barge_in,
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
