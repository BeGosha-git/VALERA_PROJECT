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
import json
import re
import time
import queue
import threading
import tempfile
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# import random  # Используется в отключённой ветке реакции на поддакивания.
import numpy as np
import soundfile as sf
import torch
from silero_vad import load_silero_vad
import requests
from bs4 import BeautifulSoup

# Пути к подпроектам (файл лежит в корне проекта)

from russian_text_to_speech.NeuralSpeaker import NeuralSpeaker
from tts_backends import get_xtts_client
from asr_fix import fix_asr
from valera_memory import (
    CONTINUE_RE, FOLLOWUP_WINDOW_SEC, GREET_RE, THANKS_RE,
    ConversationMemory, FollowUp, is_offer_confirmation, needs_topic, recall_kind, style_hint,
)
from kairos_asr import KairosASR
from phrase_gate import PhraseGate
                                                                                                              
# Рантайм-состояние проекта: общие настройки, буферы для GUI и триггеры
# «слово → сетевая команда» (робот Unitree H1). Один и тот же экземпляр
# используется и CLI-ассистентом, и GUI, поэтому тумблеры в окне применяются
# мгновенно, без перезапуска.
from valera_runtime import SensitivityController, get_runtime

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
# Сколько «запасного» звука с микрофона оставляем вокруг фразы (сек).
# Полсекунды до/после не дают срезать первый и последний слог.
PHRASE_PADDING_SEC = _env_float("PHRASE_PADDING_SEC", 0.3)
# Максимальная длительность одной фразы — защита от «залипания» микрофона
MAX_PHRASE_DURATION_SEC = _env_float("MAX_PHRASE_DURATION_SEC", 20.0)

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
# Пауза тишины ПЕРЕД речью (мс). Колонки/усилитель открываются не мгновенно,
# поэтому первые ~200-300 мс звука «проглатывались». Небольшой буфер тишины
# в начале даёт устройству время открыться — начало фразы не срезается.
TTS_LEADING_SILENCE_MS = _env_int("TTS_LEADING_SILENCE_MS", 250)
# Синтез идёт по предложениям: первое предложение озвучивается, пока
# синтезируется второе — первые слова звучат заметно раньше.
TTS_PARALLEL_SENTENCES = os.getenv("TTS_PARALLEL_SENTENCES", "1") not in ("0", "false", "False")

# Лимит слов в ответе. 40 слов — коротко и по делу, без «воды»: ответ читает
# синтезатор речи, длинные тирады слушать тяжело. Используется и в промпте,
# и в num_predict, и при обрезке (limit_words).
ANSWER_MAX_WORDS = 40
# num_predict — потолок генерации в токенах. Русский у llama3.1 ≈ 2–2,5 токена
# на слово, поэтому 80 токенов (~35 слов) обрывали ответ — берём 110 (~50 слов),
# чтобы модель успела закончить мысль, а до 40 слов ответ уже подрежет
# limit_words по границе предложения (обрыв на полуслове выглядит плохо).
ANSWER_MAX_TOKENS = 110


# ---------------------------------------------------------------------------
# ДЛИНА ОТВЕТА ЗАВИСИТ ОТ ВОПРОСА
# ---------------------------------------------------------------------------
# Простой фактический вопрос — одна фраза, «расскажи/объясни/экскурсия» — развёрнуто.
# Потолок считается один раз на реплику (set_answer_budget) и дальше используется
# всеми обрезками (finalize_answer, пересказ, прямой ответ, озвучка).
ANSWER_WORDS_CHAT = 18      # приветствие, спасибо
ANSWER_WORDS_SHORT = 30     # «где/когда/сколько/кто …»
ANSWER_WORDS_NORMAL = 35    # обычный вопрос
ANSWER_WORDS_LONG = 70      # «расскажи/объясни/как устроено/почему/подробнее»
ANSWER_WORDS_TOUR = 70      # экскурсия
_answer_words_now = ANSWER_MAX_WORDS

_LONG_Q_RE = re.compile(
    r"(?i)\b(?:расскажи\w*|объясни\w*|опиши\w*|поясни\w*|подробн\w*|как\s+(?:работает|работают|устроен\w*|"
    r"это\s+работает|ходит|видит|управля\w*)|почему|зачем|в\s+чём\s+(?:разница|суть|смысл)|чем\s+отлича\w*|"
    r"что\s+такое|для\s+чего|сравни\w*|посоветуй\w*|что\s+(?:ты\s+)?умеешь)\b"
)
_TOUR_Q_RE = re.compile(r"(?i)\b(?:экскурси\w*|проведи\w*|покажи\w*\s+(?:лаборатор\w*|всё|все)|обзор)\b")
_SHORT_Q_RE = re.compile(
    r"(?i)^\s*(?:а\s+)?(?:где|когда|сколько|кто|какой|какая|какие|какое|чей|есть\s+ли|можно\s+ли|во\s+сколько|"
    r"как\s+(?:зовут|называется|добраться|проехать|пройти))\b"
)


def answer_budget(question: str) -> int:
    """Сколько слов ответа уместно на этот вопрос."""
    q = (question or "").strip()
    if not q:
        return ANSWER_MAX_WORDS
    n = len(q.split())
    if _TOUR_Q_RE.search(q):
        return ANSWER_WORDS_TOUR
    if _LONG_Q_RE.search(q) or CONTINUE_RE.search(q):
        return ANSWER_WORDS_LONG
    if (GREET_RE.search(q) or THANKS_RE.search(q)) and n <= 6:
        return ANSWER_WORDS_CHAT
    if _SHORT_Q_RE.match(q) and n <= 6:
        return ANSWER_WORDS_SHORT
    return ANSWER_WORDS_NORMAL


def set_answer_budget(question: str) -> int:
    """Фиксирует потолок длины для текущей реплики (вызывается в начале обработки)."""
    global _answer_words_now
    _answer_words_now = answer_budget(question)
    return _answer_words_now


def current_answer_words() -> int:
    return _answer_words_now


def tokens_for_words(words: int) -> int:
    """Потолок генерации в токенах: русский ≈ 2,3 токена на слово + запас."""
    return int(words * 2.4) + 30
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
# Обработка аудио: АРУ, энергетический гейт, шумодав
# ---------------------------------------------------------------------------
def rms(audio: np.ndarray) -> float:
    """Среднеквадратичный уровень сигнала (0..1)."""
    if audio is None or len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def apply_gain(audio: np.ndarray, gain: float) -> np.ndarray:
    """
    Усиливает сигнал с мягким ограничением (tanh), чтобы громкие звуки
    не «клиппились» и не превращались в хрип.
    """
    if gain is None or abs(gain - 1.0) < 1e-3 or len(audio) == 0:
        return audio
    boosted = audio * float(gain)
    # Мягкий лимитер: пики плавно уходят в насыщение вместо резкого обрезания
    return np.tanh(boosted).astype(np.float32)


def noise_gate(audio: np.ndarray, threshold: float) -> np.ndarray:
    """
    Энергетический гейт: обнуляет окно 20 мс, если его RMS ниже порога.
    Именно это «шумодав» и вырезает — и именно это подсвечивается в GUI.
    """
    if threshold <= 0 or len(audio) == 0:
        return audio
    win = max(int(SAMPLE_RATE * 0.02), 1)
    out = audio.copy()
    for start in range(0, len(out), win):
        seg = out[start:start + win]
        if rms(seg) < threshold:
            out[start:start + win] = 0.0
    return out


def reduce_noise(audio: np.ndarray, noise_profile: Optional[np.ndarray],
                 prop_decrease: float = 0.8) -> np.ndarray:
    """Спектральное шумоподавление по записанному профилю шума (noisereduce)."""
    if noise_profile is None or len(noise_profile) == 0 or len(audio) == 0:
        return audio
    try:
        import noisereduce as nr

        return nr.reduce_noise(
            y=audio,
            sr=SAMPLE_RATE,
            y_noise=noise_profile,
            prop_decrease=prop_decrease,
            n_fft=1024,
        ).astype(np.float32)
    except ImportError:
        return audio
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Шумодав не сработал: {e}", file=sys.stderr)
        return audio


def clean_audio(audio: np.ndarray, noise_profile: Optional[np.ndarray],
                controller: "SensitivityController", runtime) -> np.ndarray:
    """
    Полный тракт очистки фразы перед распознаванием.

    Порядок шагов (он важен):
      1. АРУ — подтягиваем тихий микрофон к рабочему уровню.
      2. Энергетический гейт — рубим шумовые окна ниже порога шумового пола.
      3. Спектральное шумоподавление по профилю шума.
      4. Мягкий подъём остатка и нормализация пика.

    Каждый шаг можно выключить тумблерами в GUI.
    """
    if len(audio) == 0:
        return audio

    # 1. АРУ (автоматическая регулировка усиления) — только в авто-режиме
    if runtime.feature("auto_vad"):
        audio = apply_gain(audio, controller.gain)

    # 2. Энергетический гейт
    if runtime.feature("noise_reduction"):
        audio = noise_gate(audio, controller.gate_threshold())

    # 3. Спектральное шумоподавление по профилю
    if runtime.feature("noise_reduction"):
        audio = reduce_noise(audio, noise_profile)

    # 4. Нормализация пика (не даём клиппинга, но и не оставляем тишину)
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if 0 < peak < 0.9:
        audio = (audio / peak * 0.9).astype(np.float32)

    return audio


# ---------------------------------------------------------------------------
# Распознавание речи (STT)
# ---------------------------------------------------------------------------
# Используется лёгкая модель Kairos (см. kairos_asr). Она быстрая, работает
# на CPU и не требует тяжёлых зависимостей — этого достаточно для коротких
# фраз экскурсовода.


class RAGClient:
    """
    Клиент WebRAgent.

    Работает через **серверный чат** WebRAgent (``/chat/<chat_id>/query``):
    WebRAgent сам хранит историю вопросов/ответов в MongoDB и подставляет её
    как контекст в каждый следующий запрос. Это проверенно надёжнее, чем
    передавать историю вручную через ``conversation_context``: серверный чат
    использует «правильный» путь и модель действительно помнит предыдущие
    реплики.

    ``query()`` оставлен для обратной совместимости (прямой ``/query`` без
    контекста).
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

    def login(self, retries: int = 20, delay: float = 3.0) -> bool:
        """
        Вход в WebRAgent С ПОВТОРАМИ.

        При старте всего стека контейнер WebRAgent (/run_all.sh web) поднимается
        на 10–40 секунд позже пульта. Одиночная попытка входа падала с
        «Connection refused», ассистент умирал с «Не удалось войти в WebRAgent»
        и сам больше не поднимался: пульт показывал статус «Ошибка». Теперь ждём
        сервис и повторяем.
        """
        for attempt in range(1, retries + 1):
            if self._try_login():
                return True
            if attempt < retries:
                print(
                    f"⏳ WebRAgent ещё не готов (попытка {attempt}/{retries}) — "
                    f"жду {delay:.0f} с…",
                    file=sys.stderr,
                )
                time.sleep(delay)
        print("❌ WebRAgent так и не поднялся — ассистент не сможет отвечать.",
              file=sys.stderr)
        return False

    def _try_login(self) -> bool:
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

    # ------------------------------------------------------------------
    # Прямой /query (без сохранения контекста) — резервный путь
    # ------------------------------------------------------------------
    def query(
        self,
        query_text: str,
        collection_id: Optional[str] = None,
        use_agent_search: bool = False,
        use_web_search: bool = False,
        max_results: int = 4,
        conversation_context: Optional[list] = None,
        use_deep_search: bool = False,
        original_query: str = "",
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
            "use_deep_search": "on" if use_deep_search else "",
            "agent_strategy": "direct",
            "max_results": str(max(1, min(max_results, 10))),
        }
        if collection_id:
            data["collection_id"] = collection_id
        if conversation_context:
            import json
            data["conversation_context"] = json.dumps(conversation_context)
        if original_query and original_query != query_text:
            data["original_query"] = original_query

        query_url = f"{self.base_url}/query"
        try:
            response = self.session.post(query_url, data=data, timeout=300)
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
            response = self.session.get(url, timeout=60)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ошибка получения коллекций: статус {response.status_code}"
                )
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Ошибка соединения при получении коллекций: {e}")

    def get_llm_provider(self) -> str:
        """
        Возвращает активный LLM-провайдер WebRAgent ('ollama', 'deepseek'…).

        Зачем это знать ассистенту: ответ ищет и формулирует САМ WebRAgent.
        Если там выбрана облачная модель (DeepSeek), её ответ уже точный и
        короткий — пересказывать его локальной 3B-моделью ВРЕДНО: именно
        пересказ портил факты и имена («Санислав Кудж», «МИФИ»). Поэтому при
        облачном провайдере пересказ пропускается.

        Результат кэшируется на минуту, чтобы не дёргать сервер на каждую фразу.
        """
        import time as _time

        now = _time.time()
        cached = getattr(self, "_provider_cache", None)
        if cached and (now - cached[0]) < 60:
            return cached[1]

        if not self._logged_in:
            return "ollama"
        try:
            response = self.session.get(
                f"{self.base_url}/admin/api/models/active", timeout=10
            )
            if response.status_code == 200:
                provider = str(response.json().get("llm_provider") or "").lower()
                self._provider_cache = (now, provider)
                return provider
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ Не удалось узнать провайдер LLM: {e}", file=sys.stderr)

        # Не смогли узнать — считаем локальной (поведение по умолчанию).
        self._provider_cache = (now, "ollama")
        return "ollama"

    def direct_answer(
        self,
        question: str,
        max_tokens: int = 120,
        history: Optional[list] = None,
        hint: str = "",
    ) -> str:
        """
        Отвечает на вопрос БЕЗ поиска по базе знаний — только моделью.

        Нужно для вопросов о самом роботе («ты кто», «как тебя зовут»,
        «Лера, ты работаешь?»). В базе знаний про личность робота ничего нет:
        если отправить такой вопрос в поиск, робот получает случайные чанки и
        отвечает «Этого я не знаю» — или, что хуже, находит реального человека
        с похожим именем (например, телеведущую Валерию Кудрявцеву).

        Использует активного провайдера WebRAgent, поэтому работает и с
        локальной моделью, и с DeepSeek.

        Возвращает пустую строку, если сервер недоступен — вызывающий код
        тогда отвечает заготовкой.
        """
        if not self._logged_in:
            return ""
        url = f"{self.base_url}/direct"
        # Контекст разговора и подсказка — необязательные, но именно они
        # спасают от «не разобрал» на уточняющих репликах.
        payload = {"question": question, "max_tokens": max_tokens}
        if history:
            payload["history"] = history
        if hint:
            payload["hint"] = hint
        try:
            response = self.session.post(
                url,
                json=payload,
                timeout=90,
            )
            if response.status_code != 200:
                return ""
            return (response.json().get("response") or "").strip()
        except requests.RequestException as e:
            print(f"⚠️ Прямой ответ недоступен: {e}", file=sys.stderr)
            return ""



    # ------------------------------------------------------------------
    # Серверный чат WebRAgent — ОСНОВНОЙ путь (контекст на сервере)
    # ------------------------------------------------------------------
    def create_chat(self) -> Optional[str]:
        """Создаёт новый чат на сервере и возвращает его chat_id."""
        if not self._logged_in:
            raise RuntimeError("Необходимо сначала выполнить login()")

        url = f"{self.base_url}/chat/new"
        try:
            response = self.session.post(url, timeout=60)
            if response.status_code != 200:
                print(
                    f"❌ Ошибка создания чата: статус {response.status_code}",
                    file=sys.stderr,
                )
                return None
            return response.json().get("chat_id")
        except requests.RequestException as e:
            print(f"❌ Ошибка соединения при создании чата: {e}", file=sys.stderr)
            return None

    def chat_exists(self, chat_id: str) -> bool:
        """Проверяет, что чат ещё существует на сервере (например, после перезапуска БД)."""
        if not chat_id:
            return False
        try:
            resp = self.session.get(
                f"{self.base_url}/chat/{chat_id}/messages",
                params={"limit": 1},
                timeout=30,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def chat_query(
        self,
        chat_id: str,
        query_text: str,
        collection_id: Optional[str] = None,
        use_agent_search: bool = False,
        use_web_search: bool = False,
        max_results: int = 4,
        use_deep_search: bool = False,
        conversation_context: Optional[list] = None,
        original_query: str = "",
    ) -> dict:
        """
        Отправляет вопрос в существующий серверный чат.

        WebRAgent сохраняет исходную реплику и использует переданный контекст.
        Когда клиент не передаёт историю, сервер берёт её из чата.
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
            "use_deep_search": "on" if use_deep_search else "",
            "agent_strategy": "direct",
            "max_results": str(max(1, min(max_results, 10))),
        }
        if collection_id:
            data["collection_id"] = collection_id
        if conversation_context is not None:
            data["conversation_context"] = json.dumps(conversation_context, ensure_ascii=False)
        if original_query and original_query != query_text:
            data["original_query"] = original_query

        url = f"{self.base_url}/chat/{chat_id}/query"
        try:
            response = self.session.post(url, data=data, timeout=300)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ошибка запроса чата: статус {response.status_code}, "
                    f"ответ: {response.text[:200]}"
                )
            return response.json()
        except requests.RequestException as e:
            raise RuntimeError(f"Ошибка соединения при запросе к чату: {e}")

    def chat_query_stream(self, chat_id: str, query_text: str, collection_id: str, max_results: int = 4,
                          *, conversation_context: Optional[list] = None, original_query: str = ""):
        """
        Потоковый вариант chat_query: генератор событий NDJSON от /chat/<id>/query_stream.
        События: {"t":"meta"}, {"t":"d","x":"кусок"}, {"t":"done","response":…} или {"t":"err"}.
        """
        if not self._logged_in:
            raise RuntimeError("Необходимо сначала выполнить login()")
        data = {"query": query_text, "collection_id": collection_id,
                "max_results": str(max(1, min(max_results, 10)))}
        if conversation_context is not None:
            data["conversation_context"] = json.dumps(conversation_context, ensure_ascii=False)
        if original_query and original_query != query_text:
            data["original_query"] = original_query
        resp = self.session.post(f"{self.base_url}/chat/{chat_id}/query_stream", data=data,
                                 stream=True, timeout=(10, 120))
        if resp.status_code != 200:
            raise RuntimeError(f"Ошибка потокового запроса: статус {resp.status_code}")
        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    continue
        finally:
            resp.close()

    def get_chat_messages(self, chat_id: str, limit: int = 50):
        """Возвращает сообщения чата (для отладки/восстановления истории)."""
        if not self._logged_in:
            raise RuntimeError("Необходимо сначала выполнить login()")

        url = f"{self.base_url}/chat/{chat_id}/messages"
        try:
            response = self.session.get(url, params={"limit": limit}, timeout=60)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ошибка получения сообщений чата: статус {response.status_code}"
                )
            data = response.json()
            # Сервер может вернуть {"chat": ..., "messages": [...]}
            if isinstance(data, dict):
                return data.get("messages", [])
            return data
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

# Второй набор — студенческое самоуправление: Студсоюз и его подразделения
# (студсоветы, профком, волонтёрский центр, отряды, СНО, ССК…). Вопросы про
# них — это тоже вопросы про РТУ МИРЭА, поэтому ищем их в ТОЙ ЖЕ коллекции,
# а не в интернете. Слова добавлены во всех формах склонения, потому что
# поиск сравнивает слова буквально.
STUDSOYUZ_ALIASES = [
    # Студсоюз — слитно и с вариантами распознавания речи
    "студсоюз", "студсоюза", "студсоюзу", "студсоюзе", "студсоюзом",
    "студсоюзы", "студсоюзов",
    "стуцсоюз", "стутсоюз", "студсоюс", "студ-союз",
    # Полное название организации
    "студенческий союз", "студенческого союза", "студенческому союзу",
    "студенческом союзе", "студенческим союзом",
    # Студенческие советы и самоуправление
    "студсовет", "студсовета", "студсовету", "студсовете", "студсоветы",
    "студсоветов", "студсоветам",
    "студенческий совет", "студенческого совета", "студенческом совете",
    "самоуправление", "самоуправления", "самоуправлению", "самоуправлением",
    "староста", "старосты", "старосте", "старост", "старостат", "старостата",
    # Профсоюз, профком, профбюро
    "профком", "профкома", "профкому", "профкоме",
    "профсоюз", "профсоюза", "профсоюзу", "профсоюзе", "профсоюзы", "профсоюзов",
    "профсоюзный", "профсоюзная", "профсоюзное", "профсоюзные",
    "профсоюзного", "профсоюзной", "профсоюзному", "профсоюзном",
    "профсоюзную", "профсоюзных", "профсоюзным", "профсоюзными",
    "профорганизация", "профорганизации", "профорганизацию", "профорганизаций",
    "профбюро", "профорг", "профорга", "профорги",
    # Клубы, отряды, волонтёры, СНО, ССК, медиа
    "волонтёрский", "волонтерский",
    "волонтёрский центр", "волонтерский центр",
    "волонтёр", "волонтер", "волонтёры", "волонтеры",
    "штаб мсо", "студотряд", "студотряда", "студотряды", "студотрядов",
    "студенческий отряд", "студенческие отряды", "поисковый отряд",
    "сно", "студенческое научное общество",
    "сск", "киберкружочек", "медиацентр", "студофис", "студофиса",
    "sumirea",
]

# Ключевые слова, при которых поиск идёт по коллекции (а не по интернету).
# Включает МИРЭА-алиасы + слова про Студсоюз + прочие слова про РТУ МИРЭА.
COLLECTION_KEYWORDS = MIREA_ALIASES + STUDSOYUZ_ALIASES + [
    "мегалаборатория", "мегалаборатории", "мегалабораторий",
    "лаборатория", "лаборатории", "лабораторий",
    "институт", "института", "институту", "институте", "институтом",
    "институты", "институтов", "институтам", "институтах",
    "вуз", "вуза", "вузе", "вузов",
    "университет", "университета", "университету", "университете",
    "университетом", "университеты", "университетов",
    "колледж", "колледжа", "колледже", "колледжи",
    "кафедра", "кафедры", "кафедре", "кафедру", "кафедрой", "кафедр",
    "кафедрам", "кафедрами", "кафедрах", "базовая кафедра",
    "факультет", "факультета", "факультету", "факультете", "факультетом",
    "факультеты", "факультетов", "факультетам", "факультетах",
    "деканат", "деканата", "деканату", "деканате", "деканаты", "деканатов",
    "диплом", "диплома", "дипломы", "поступлени",
    "поступить", "поступать", "поступил", "поступит", "поступлю",
    "абитуриент", "абитуриента", "абитуриенту", "абитуриенты", "абитуриентов",
    "стипендия", "стипендии", "стипендию", "стипендией", "стипендий",
    "общежитие", "общежития", "общежитию", "общежитии", "общежитий",
    "ректор", "ректора", "ректору", "ректоре",
    "баллы", "баллов", "егэ", "проходной", "проходные", "проходн",
    # Поступление и приём: вопрос «сколько мест бюджетных» тоже про вуз.
    "бюджет", "бюджета", "бюджетный", "бюджетная", "бюджетное",
    "бюджетные", "бюджетных", "бюджетным", "бюджетными",
    "платн", "платное", "платные", "контракт", "контрактное",
    "приём", "прием", "приёма", "приема", "приёме", "приеме",
    "приёмная", "приемная", "приёмную", "комиссия", "комиссии",
    "зачислени", "зачислен", "конкурс", "конкурсный", "аттестат",
    "направлени", "направления", "специальность", "специальности",
    "программа", "программы", "программе", "магистратура", "магистратуры",
    "бакалавриат", "бакалавриата", "специалитет", "аспирантура",
    "количество", "мест", "места", "сроки", "срок", "подача", "документы",
    "испытания", "испытаний", "испытаниях",
    "учитесь", "учёба", "учёбы", "учеба", "учебы",
    "тхт",
    "стромынка", "стромынки",
    "вернадка", "вернадки", "вернадку", "вернадке", "вернадкой",
    "рту",
]


# Основы слов, по которым вопрос уходит в базу МИРЭА (любое окончание).
# Раньше «поступление», «мегалабораторию», «день открытых дверей», «метро»
# в список не попадали и такие вопросы шли в интернет/болталку.
KB_STEMS = [
    "поступлени", "зачислен", "проходн", "абитуриент", "ректор", "кафедр",
    "общежит", "стипенд", "институт", "университет", "колледж", "магистратур",
    "бакалавр", "аспирантур", "мегалаборатор", "лаборатор", "робототехн",
    "мехатрон", "гуманоид", "манипулятор", "юнитри", "unitree", "кампус",
    "студсоюз", "профком", "профсоюз", "приёмн", "приемн",
    "дн\\w+\\s+открыт", "день\\s+открыт", "дней\\s+открыт",
    "добрат", "доехат", "проехат", "метро", "экскурси", "ВУЦ",
    # Экскурсия по лаборатории (раздел 13 базы)
    "полигон", "kuka", "youbot", "gofa", "iiwa", "хакатон", "стартап", "мультиагент",
    "автономн", "антропоморф", "четырёхног", "четырехног",
]
_KB_STEMS_RE = re.compile(r"(?<![\w])(?:" + "|".join(KB_STEMS) + r")\w*", re.IGNORECASE)


def normalize_mirea(text: str) -> str:
    """Заменяет слова близкие к МИРЭА на 'МИРЭА' (по границам слов)."""
    import re

    pattern = r"\b(" + "|".join(MIREA_ALIASES) + r")\b"
    return re.sub(pattern, "МИРЭА", text, flags=re.IGNORECASE)


_TECH_ASR_FIXES = {
    # Ошибки распознавания речи в технических словах
    "гейхаб": "гитхаб", "гидхаб": "гитхаб", "гитхап": "гитхаб", "гидхап": "гитхаб",
    "гейтхаб": "гитхаб", "шидхап": "гитхаб", "гитхаб": "гитхаб",
}


def normalize_tech_terms(text: str) -> str:
    """Исправляет типичные ослышки технических слов («гейхаб» → «гитхаб»)."""
    import re as _re

    pattern = _re.compile(r"\b(" + "|".join(_TECH_ASR_FIXES) + r")\b", _re.IGNORECASE)
    return pattern.sub(lambda m: _TECH_ASR_FIXES[m.group(1).lower()], text)


def normalize_studsouz(text: str) -> str:
    """
    Приводит варианты «Студсоюз» (в том числе ошибки распознавания речи)
    к словарной форме — так поиск по базе надёжнее находит раздел 8.3.

    Многословные варианты («студенческий союз») не трогаем: они уже есть в
    базе в правильном виде.
    """
    import re

    variants = [
        "студсоюс", "стуцсоюз", "стутсоюз", "студ-союз", "студтсоюз",
        "студсоюзс", "статсоюз",
    ]
    pattern = r"\b(" + "|".join(map(re.escape, variants)) + r")\b"
    return re.sub(pattern, "Студсоюз", text, flags=re.IGNORECASE)


def kb_match(text: str) -> Optional[str]:
    """
    Возвращает слово, по которому вопрос отнесён к базе РТУ МИРЭА
    (МИРЭА, Студсоюз и т.п.), или ``None``, если ни одно не найдено.
    """
    import re

    pattern = r"\b(" + "|".join(COLLECTION_KEYWORDS) + r")\b"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    # Основы слов: ловят любые падежи («поступление», «мегалабораторию»,
    # «робототехнике»), которые точный список выше пропускает.
    stem = _KB_STEMS_RE.search(text)
    return stem.group(0) if stem else None


def is_mirea_related(text: str) -> bool:
    """Есть ли в тексте упоминание МИРЭА, Студсоюза или ключевых слов."""
    return kb_match(text) is not None


# ---------------------------------------------------------------------------
# РАЗГОВОРНЫЕ РЕМАРКИ (не вопросы)
# ---------------------------------------------------------------------------
# Гость часто говорит не вопрос, а ремарку: «Не знаю, по сетям там
# что-нибудь надо», «Адреса она выдумывает какие-то. Ладно», «Я сам с собой
# разговариваю». В базе знаний такого нет, поэтому поиск подсовывал случайный
# чанк, и робот отвечал не по теме (например, про Ставропольский филиал).
#
# Такие реплики надо отвечать напрямую моделью, с учётом истории разговора.
_CONV_REMARK_MARKERS = (
    "не знаю", "ладно", "хватит", "перестань", "перестаньте",
    "сам с собой", "сама с собой", "разговарива",
    "повтори", "я сказал", "я говорю", "я думал", "я думала",
    "кажется", "похоже", "наверное", "короче", "в общем",
    "ты меня", "меня не", "мне не", "мне надо", "мне нужно",
    "давай", "подожди", "секунду", "погоди",
)


def looks_conversational_remark(text: str) -> bool:
    """
    Похоже ли, что это ремарка гостя, а не вопрос по существу.

    Условия ЖЁСТКИЕ, чтобы не увести в «болталку» настоящий вопрос:
    * нет вопросительного знака (вопрос — всегда ищем в базе/интернете);
    * в реплике есть разговорный маркер (см. _CONV_REMARK_MARKERS);
    * либо это вообще бессмыслица от распознавания речи.
    """
    if not text:
        return False
    low = text.lower().strip()
    if "?" in low:
        return False
    # Есть ключевое слово вуза — это вопрос по существу, пусть ищет в базе.
    if kb_match(text):
        return False
    if is_gibberish(low):
        return True
    if len(low.split()) < 2:
        return False
    return any(m in low for m in _CONV_REMARK_MARKERS)


# ---------------------------------------------------------------------------
# Разговор о вузе — ТОЛЬКО про РТУ МИРЭА
# ---------------------------------------------------------------------------
# Если речь про вуз вообще или про вузовские организации (Студсоюз, профком,
# студсовет, деканат, кафедра, ректор, общежитие, стипендия…), Валера рассказывает
# ТОЛЬКО про РТУ МИРЭА. Про другие университеты и их студенческие организации
# он не говорит ничего — иначе он начнёт выдумывать чужие факты.
#
# Правило дублируется в промпте (модель не всегда слушается) и в системном
# сообщении модели — см. Modelfile / Modelfile-FAST.
VUZ_ONLY_RULE = (
    "Про вуз и вузовские организации (Студсоюз, профком, студсовет, деканат, "
    "кафедра, ректор, общежитие, стипендия, клубы, отряды, комиссии) рассказывай "
    "ТОЛЬКО про РТУ МИРЭА. Про другие университеты, институты, колледжи и их "
    "студенческие организации ничего не говори — отвечай, что знаешь только "
    "про РТУ МИРЭА, и предложи спросить о нём. "
)

# Чужие вузы: если человек спрашивает про них, честно говорим, что знаем только
# про МИРЭА (нужно для лога и для подсказки поведения модели).
OTHER_UNIVERSITY_NAMES = [
    "мгу", "ломоносов", "бауманк", "бауманск", "мфти", "физтех",
    "вшэ", "вышк[аи]", "мифи", "мисис", "маи\\b", "мгту", "мэи",
    "миэт", "итмо", "спбг", "политех", "сеченов", "плеханов",
    "ранхигс", "финансовый университет", "станькин", "политехническ",
    r"государственн\w+ университет", "стэнфорд", "гарвард", "оксфорд",
    "кембридж", "сорбонн",
]


def _other_university_hit(text: str) -> Optional[str]:
    """Название чужого вуза в вопросе — или ``None``, если его нет."""
    if not text:
        return None
    import re

    pattern = r"\b(" + "|".join(OTHER_UNIVERSITY_NAMES) + r")"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1) if match else None


def posture_rule() -> str:
    """
    Строка про то, как сегодня расположен робот (сидит / стоит).

    Берётся из ``robot.posture`` в ``valera_config.json``. Нужна, чтобы робот
    не утверждал лишнего: сидя на стойке экскурсовода, он не должен говорить
    «я стою».
    """
    try:
        from valera_runtime import get_runtime

        where = str(get_runtime().get("robot", "posture", default="") or "").strip()
    except Exception:  # noqa: BLE001
        where = ""
    if not where:
        return ""
    return (
        f"Сегодня ты работаешь {where}: на вопросы отвечаешь {where}, "
        f"позицию не комментируй и не утверждай, что стоишь или ходишь. "
    )


# ---------------------------------------------------------------------------
# Запретные темы: политика и военная тематика
# ---------------------------------------------------------------------------
# Робот стоит в публичном месте (холл вуза), поэтому политические и военные
# темы не обсуждаем вовсе — это нейтральность и безопасность. Проверяем и в
# промпте, и КОДОМ (модель не всегда слушается).
#
# Шаблоны подобраны по началу слова и требуют границы слова, чтобы не ловить
# обычные русские слова. Например «\bдум[аеы]\b» поймает «дума/думе», но НЕ
# «думаю»; «\bсво\b» поймает аббревиатуру, но НЕ «своё».
FORBIDDEN_TOPIC_PATTERNS = [
    # --- политика ---
    r"\bполитик", r"\bполитич",
    r"\bпрезидент",
    r"\bпутин", r"\bзеленск", r"\bнавальн", r"\bмишустин",
    r"\bправительств", r"\bпарламент", r"\bсенат",
    r"\bгосдум", r"\bдум[аеы]\b",
    r"\bпарти[яию]\b", r"\bоппозици",
    r"\bсанкци", r"\bреферендум", r"\bмитинг", r"\bпротест",
    r"\bголосовани", r"\bвыбор(ы|ах|ов|ам)\b", r"\bкандидат",
    r"\bгосударств\w*\s+дум",
    # --- военная тематика / СВО ---
    r"\bсво\b", r"\bспецопераци", r"\bспециальн\w+\s+военн",
    r"\bвоенн\w+\s+операци", r"\bвоенн\w+\s+действи",
    r"\bвойн", r"\bвоюем", r"\bвоевать",
    r"\bукраин", r"\bмобилизаци", r"\bмобилиз", r"\bфронт\b",
    r"\bобстрел", r"\bбпла\b", r"\bдрон", r"\bснаряд",
    r"\bнацизм", r"\bнацист", r"\bфашизм", r"\bфашист",
    # --- религия и национальность (спорное в публичном месте) ---
    r"\bрелиги", r"\bцерков", r"\bнациональност", r"\bнационализм",
    r"\bбог[ауеом]?\b", r"\bгоспод[ьи]\b", r"\bмолит", r"\bкрещени",
]

# Вежливый отказ без нотаций — робот не спорит и не оценивает.
FORBIDDEN_REFUSAL = (
    "Эту тему я не обсуждаю. Спросите, пожалуйста, что-нибудь другое — с радостью отвечу."
)

# Варианты короткого ответа для отключённого тумблера.
# BACKCHANNEL_REPLIES = [
#     "Так, слушаю.",
#     "Хорошо.",
#     "Да, продолжаю.",
#     "Да, я здесь.",
#     "Понял, слушаю дальше.",
# ]


def is_forbidden_topic(text: str) -> bool:
    """
    Политика / военная тематика / религия / национальность — не обсуждаем.

    Возвращает True, если вопрос попадает в запретную тему.
    """
    if not text:
        return False
    import re

    low = text.lower()
    return any(re.search(p, low) for p in FORBIDDEN_TOPIC_PATTERNS)


# ---------------------------------------------------------------------------
# «Поддакивания» (backchannel)
# ---------------------------------------------------------------------------
# Короткие реплики-подтверждения («угу», «ага», «да», «ясно»).
BACKCHANNEL_WORDS = {
    "угу", "ага", "да", "нет", "ну", "так", "ок", "окей", "окэй", "ладно",
    "понятно", "понял", "ясно", "хм", "мм", "ммм", "эм", "эээ", "хорошо",
    "точно", "верно", "конечно", "спасибо", "благодарю", "ясненько",
    "ух", "ой", "ого", "вау", "ничего", "да-да", "ага-ага", "угу-угу",
}


def is_backchannel(text: str) -> bool:
    """
    Проверяет, состоит ли реплика только из поддакиваний («угу», «ага, да»).

    Требуем, чтобы ВСЕ слова были поддакиваниями и их было не больше трёх:
    «ну да, ясно» → да; «да, расскажи про МИРЭА» → нет (это уже запрос).
    """
    if not text:
        return False
    import re

    words = re.findall(r"[а-яёa-z\-]+", text.lower())
    if not words or len(words) > 3:
        return False
    return all(w in BACKCHANNEL_WORDS for w in words)


_SPEECH_REQUEST_RE = re.compile(
    r"^[\s,]*(?:(?:валера|лера)[\s,]+)?(?:пожалуйста[\s,]+)?"
    r"(скажи(?:те|-ка)?|произнеси(?:те)?|повтори(?:те)?(?:\s+за\s+мной)?|"
    r"мож(?:ешь|ете)\s+(?:сказать|произнести|повторить)(?:\s+за\s+мной)?)"
    r"[\s,.:—-]+(.+)$",
    re.IGNORECASE,
)


def requested_speech(text: str) -> str:
    """Возвращает текст явной команды на произнесение."""
    request = _SPEECH_REQUEST_RE.fullmatch(text)
    if not request:
        return ""
    verb, target = request.groups()
    target = re.sub(r"(?i)^пожалуйста[\s,]+|[\s,]+пожалуйста[.!?]*$", "", target).strip()
    quoted = re.fullmatch(r'[«"„“](.+?)[»"“”][.!?]*', target)
    named_text = re.fullmatch(r"(?i)(?:(?:букву|звук)\s+([а-яёa-z])|слово\s+([а-яёa-z-]+))[.!?]*", target)
    if quoted:
        target = quoted.group(1).strip()
    elif named_text:
        target = named_text.group(1) or named_text.group(2)
    elif not re.search(r"\bза\s+мной$", verb, re.IGNORECASE):
        word = target.rstrip(".!?…").lower()
        if word in {"как", "кто", "что", "где", "чья", "чьё"}:
            return ""
        if not (
            re.fullmatch(r"[а-яёa-z]{1,3}|([а-яёa-z])(?:[\s,.-]*\1)+", word)
            or is_backchannel(word)
            or GREET_RE.fullmatch(word)
        ):
            return ""

    if not any(char.isalnum() for char in target):
        return ""
    return target if target.endswith((".", "!", "?", "…")) else target + "."



def arecord_command(extra_args=None):
    """Базовая команда захвата через arecord (16 кГц, моно, S16_LE, raw)."""
    # Устройство берём из рантайм-настроек (его можно сменить в GUI/веб на лету),
    # а если там пусто — из переменной окружения/дефолта.
    try:
        rt = get_runtime()
        device = (rt.get("audio", "input_device", default=None)
                  or rt.get("audio", "device", default=AUDIO_DEVICE)
                  or AUDIO_DEVICE)
    except Exception:  # noqa: BLE001
        device = AUDIO_DEVICE
    cmd = [
        "arecord", "-D", str(device),
        "-r", str(SAMPLE_RATE),
        "-c", "1",
        "-f", "S16_LE",
        "-t", "raw",
        "-q",
    ]
    if extra_args:
        cmd += extra_args
    return cmd


def resolve_output_device() -> str:
    """
    Возвращает устройство ВОСПРОИЗВЕДЕНИЯ (колонки) из рантайм-настроек.

    Пусто/«auto» → системное по умолчанию (``default``). Список доступных
    устройств можно посмотреть командой ``aplay -L``.
    """
    try:
        dev = get_runtime().get("audio", "output_device", default="default")
    except Exception:  # noqa: BLE001
        dev = "default"
    if not dev or dev == "auto":
        return "default"
    return str(dev)


def play_audio(audio_data: bytes, sample_rate: int = TTS_SAMPLE_RATE) -> bool:
    """
    Проигрывает сырой PCM (S16_LE, моно) на выбранных колонках через ``aplay``.

    Нужно, чтобы направлять звук на конкретное устройство (например, DIN-выход
    к роботу Unitree H1), а не только на системное по умолчанию. Библиотека
    simpleaudio так не умеет — она всегда играет в «default».

    Возвращает True при успехе.
    """
    device = resolve_output_device()
    cmd = [
        "aplay", "-D", device,
        "-r", str(sample_rate),
        "-c", "1",
        "-f", "S16_LE",
        "-t", "raw",
        "-q",
    ]
    try:
        proc = subprocess.run(cmd, input=audio_data, timeout=120,
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", "ignore")[:200]
            print(f"⚠️ aplay ({device}) вернул код {proc.returncode}: {err}",
                  file=sys.stderr)
            return False
        return True
    except FileNotFoundError:
        print("⚠️ aplay не найден — озвучка через Python (simpleaudio).",
              file=sys.stderr)
        return False
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Ошибка вывода звука ({device}): {e}", file=sys.stderr)
        return False


def leading_silence_bytes(sample_rate: int = TTS_SAMPLE_RATE,
                          ms: int = TTS_LEADING_SILENCE_MS) -> bytes:
    """
    PCM-буфер тишины (S16_LE, моно), который ставим В НАЧАЛО речи.

    Колонки/усилитель открываются не мгновенно, поэтому первые ~200-300 мс
    звука «проглатывались». Короткая пауза тишины в начале даёт устройству
    время открыться — начало фразы больше не срезается.
    """
    if ms <= 0:
        return b""
    n_samples = int(sample_rate * ms / 1000)
    return b"\x00\x00" * n_samples


def split_into_sentences(text: str) -> list:
    """
    Разбивает текст на предложения для ПОТОКОВОЙ озвучки.

    Синтез всей простыни целиком даёт долгую паузу перед первым словом.
    Если резать по предложениям и озвучивать по мере готовности, первые
    слова слышны заметно раньше: пока играет первое предложение, в фоне
    уже синтезируется второе.
    """
    import re as _re

    if not text:
        return []
    text = text.strip()
    # Режем ПОСЛЕ знака конца предложения (. ! ? …), сохраняя знак в начале фразы.
    parts = _re.split(r"(?<=[.!?…])\s+", text)
    out: list = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Метка эмоции относится к ПРЕДЫДУЩЕМУ предложению. Разбивка идёт по
        # «.!?…», поэтому «Привет! [эмоция:радость]» распадается на «Привет!»
        # и «[эмоция:радость]». Возвращаем метку туда, где она стояла, иначе
        # эмоция достанется следующей фразе.
        _m = _EMOTION_RE.match(p)
        if _m and out:
            out[-1] = f"{out[-1]} {_m.group(0).strip()}"
            p = p[_m.end():].strip()
            if not p:
                continue
        # Совсем короткие обрывки приклеиваем к предыдущему предложению,
        # чтобы не озвучивать «А.» отдельным куском.
        if out and len(p) < 3:
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out or [text]


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
    chat_id: Optional[str] = None,
    source: str = "",
    search_query: str = "",
) -> None:
    """
    Записывает заданный вопрос (и ответ) в журнал LOG_FILE.

    Одна строка — один JSON: время, что услышали с микрофона, что ушло в ИИ,
    куда ушло (база МИРЭА / интернет), ответ и id чата, в котором хранится
    контекст. Нужен для отладки и аудита: всегда видно, что именно спросили.
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
        "answer": parse_emotion(answer)[0],
        "chat_id": chat_id,
        "source": source,   # куда ушёл запрос: chat | query | direct | trigger
    }
    if search_query and search_query != query_text:
        entry["search_query"] = search_query
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


def _default_llm_model() -> str:
    """
    Имя LLM-модели Ollama.

    Приоритет: переменная окружения ``REPHRASE_MODEL`` → поле ``llm.model`` в
    ``valera_config.json`` → быстрая модель ``true_valera_fast:latest``.

    Раньше здесь жёстко стоял ``valera_live:latest`` — старая 8B-модель, из-за
    чего «активная» модель из настроек игнорировалась, а ответы были медленнее.
    """
    env = os.getenv("REPHRASE_MODEL")
    if env:
        return env
    try:
        from valera_runtime import get_runtime

        cfg_model = get_runtime().get("llm", "model", default=None)
        if cfg_model:
            return str(cfg_model)
    except Exception:  # noqa: BLE001
        pass
    return "true_valera_fast:latest"


REPHRASE_MODEL = _default_llm_model()


# ---------------------------------------------------------------------------
# Очистка английских слов из ответа
# ---------------------------------------------------------------------------
# Модель (особенно лёгкая 3B) иногда вставляет латинские слова: «robot»,
# «ownjoke», «live». Синтезатор речи читает их с чудовищным акцентом или
# вообще пропускает, поэтому чистим текст ПЕРЕД озвучкой — это гарантия,
# не зависящая от того, послушалась ли модель промпта.
_EN_WORDS = {
    "robot": "робот", "robots": "роботы", "robo": "робот",
    "joke": "шутка", "funny": "смешно", "live": "вживую", "life": "жизнь",
    "ok": "хорошо", "okay": "хорошо", "hello": "привет", "hi": "привет",
    "hey": "привет", "sorry": "извини", "please": "пожалуйста",
    "well": "ну", "actually": "вообще", "own": "свой",
    "guide": "гид", "museum": "музей", "university": "университет",
    "student": "студент", "students": "студенты", "thanks": "спасибо",
    "thank": "спасибо", "bye": "пока", "goodbye": "пока", "yes": "да",
    "no": "нет", "wow": "ого", "cool": "класс", "super": "супер",
    "human": "человек", "ai": "искусственный интеллект",
    "it": "ай-ти", "h1": "Аш-один", "unitree": "Юнитри",
    "kuka": "Кука", "abb": "А-Б-Б", "iiwa": "и-и-вэ-а", "gofa": "Го-Фа",
    "youbot": "ю-Бот", "ros": "рос", "dds": "ди-ди-эс",
    "rtu": "Р-Т-У", "mirea": "МИРЭА", "world": "мир", "people": "люди",
    "time": "время", "work": "работа", "good": "хорошо",
    "great": "отлично", "nice": "приятно", "man": "друг", "friend": "друг",
}

# Разрешённые латинские обозначения, которые НЕ трогаем (единицы измерения,
# модели техники), — их допустимо произносить как есть.
_EN_KEEP = {"v2", "3d", "5g", "usb", "gpt"}


_SENT_END_RE = re.compile(r"[.!?…]+(?=\s)")


_ABBR = {"стр", "ул", "пр", "просп", "им", "г", "гг", "д", "к", "т", "тел", "руб", "тыс", "млн", "млрд",
         "др", "см", "рис", "табл", "напр", "проф", "доц", "акад", "корп", "пом", "ауд", "каб"}


def split_complete_sentences(buf: str):
    """
    Отделяет от буфера законченные предложения; хвост без конца остаётся в буфере.

    Граница — «.!?…» + пробел + заглавная буква/кавычка следующего предложения. Не режем после
    сокращений («стр.», «ул.», «г.», «т. е.»), перед строчной буквой/цифрой и пока следующий
    символ ещё не пришёл.
    """
    out, start = [], 0
    for m in _SENT_END_RE.finditer(buf):
        end = m.end()
        tail = buf[end:]
        rest = tail.lstrip()
        if not rest:                      # следующий символ не пришёл — граница не подтверждена
            break
        # Эмоция ставится в конце КАЖДОГО предложения: «Привет! [эмоция:радость]».
        # Метка сразу после знака конца относится к ЭТОМУ предложению, а не
        # продолжает фразу. Без этой ветки «[» не считался заглавной буквой,
        # предложения склеивались и на весь ответ оставалась одна эмоция.
        emo = _EMOTION_RE.match(rest)
        if emo:
            label_end = end + (len(tail) - len(rest)) + emo.end()
            out.append(buf[start:label_end].strip())
            start = label_end
            continue
        if not (rest[0].isupper() or rest[0] in "«\"—-("):
            continue                      # продолжение предложения («2026 г. по будням»)
        word = re.findall(r"[\w-]+$", buf[start:m.start()])
        if word and m.group(0) == "." and word[0].lower() in _ABBR:
            continue                      # «стр.», «ул.», «г.»
        out.append(buf[start:end].strip())
        start = end
    return out, buf[start:]


def clean_english(text: str) -> str:
    """
    Убирает английские слова и оставшуюся латиницу из ответа.

    Сначала заменяет известные слова на русские (с сохранением заглавной
    буквы), затем выкидывает прочие латинские токены. В конце — схлопывает
    повторы вроде «Привет! привет», которые появляются после замены.
    """
    if not text:
        return text
    import re as _re

    def _repl(match: "_re.Match") -> str:
        word = match.group(0)
        low = word.lower()
        if low in _EN_KEEP:
            return word
        if low in _EN_WORDS:
            ru = _EN_WORDS[low]
            if word[:1].isupper():
                ru = ru[:1].upper() + ru[1:]
            return ru
        return ""  # прочие латинские слова (в т.ч. ownjoke) просто убираем

    cleaned = _re.sub(r"(?<![A-Za-z0-9])[A-Za-z0-9]*[A-Za-z][A-Za-z0-9']*", _repl, text)
    # Схлопываем одинаковые слова, разделённые пробелами и знаками препинания.
    cleaned = _re.sub(r"\b([А-Яа-яЁё]+)([,.!?;:]*)\s+\1\b", r"\1\2", cleaned,
                      flags=_re.IGNORECASE)
    # Убираем всё, кроме русского: лёгкая модель иногда «срывается» в чужие
    # алфавиты (тайский, китайский, арабский) — для синтезатора это мусор.
    cleaned = "".join(
        ch for ch in cleaned
        if ch.isdigit()
        or "А" <= ch <= "я" or ch in "Ёё"
        or ch in " \t\n.,!?;:—–-«»\"'()"
    )
    cleaned = _re.sub(r"\s{2,}", " ", cleaned)
    cleaned = _re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    # Убираем задвоенную пунктуацию, которая остаётся после выкинутого слова:
    # «Привет!, я …» → «Привет, я …»
    cleaned = _re.sub(r"([!?])\s*[,;:]+", r"\1", cleaned)
    cleaned = _re.sub(r"[,;:]+\s*([!?])", r"\1", cleaned)
    cleaned = _re.sub(r"([,.!?;:])\1+", r"\1", cleaned)
    # Пустые скобки после удаления латиницы: «ИКБ (ICB)» → «ИКБ ( )» → «ИКБ».
    cleaned = _re.sub(r"\(\s*\)", "", cleaned)
    cleaned = _re.sub(r"\(\s*[,;]*\s*\)", "", cleaned)
    cleaned = _re.sub(r"\s+([,.)!?;:])", r"\1", cleaned)
    cleaned = _re.sub(r"\(\s+", "(", cleaned)
    cleaned = _re.sub(r"\s{2,}", " ", cleaned)
    cleaned = _re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    return cleaned.strip()


def limit_words(text: str, max_words: int = ANSWER_MAX_WORDS) -> str:
    """
    Обрезает текст до ``max_words`` слов — гарантия короткого ответа.

    Модель не всегда слушается лимита из промпта, поэтому подрезаем кодом.
    Обрезка идёт по ГРАНИЦЕ ПРЕДЛОЖЕНИЯ, если она попадает в лимит: так ответ
    не обрывается на полуслове. Если предложение одно и слишком длинное —
    режем по последнему знаку препинания или пробелу.
    """
    if not text:
        return text
    import re as _re

    # Нормализуем пробелы и считаем слова.
    text = _re.sub(r"\s{2,}", " ", text).strip()
    words = text.split()
    if len(words) <= max_words:
        return text

    limited = " ".join(words[:max_words])
    # Ищем ПОСЛЕДНИЙ конец предложения внутри лимита — обрезаем красиво.
    matches = list(_re.finditer(r"[.!?…]+(?=\s|$)", limited))
    if matches:
        cut = matches[-1].end()
        # Не даём обрезать в самое начало (меньше половины смысла).
        if len(limited[:cut].split()) >= max(3, max_words // 3):
            return limited[:cut].strip()
    # Иначе — последняя запятая внутри лимита, либо просто конец с «…».
    comma = limited.rfind(",")
    if comma > 0 and len(limited[:comma].split()) >= max(3, max_words // 3):
        return limited[:comma].strip() + "."
    return limited.rstrip(" ,;:—-") + "…"


# «Вода» и дежурные вступления, которые модель любит вставлять. Убираем их
# КОДОМ: лёгкая 3B-модель почти не слушается правила «без вступлений» из промпта,
# поэтому надёжнее отрезать. Проверяем только НАЧАЛО ответа.
_FILLER_OPENERS = [
    # длинные — проверяем раньше коротких
    "о, отличный вопрос!", "о, отличный вопрос", "отличный вопрос!",
    "отличный вопрос", "хороший вопрос!", "хороший вопрос",
    "интересный вопрос!", "интересный вопрос",
    "давай разберёмся", "давай разберемся", "ну что ж", "ну что же",
    "конечно же", "конечно,", "конечно!",
    "итак,", "итак", "слушай,", "смотри,", "знаешь,",
    "да, я знаю", "ооо!", "ого!", "ой!",
    "как ии-модель,", "как ии модель,", "согласно данным,",
    "в предоставленной базе,", "хм,",
]


def fix_name(text: str) -> str:
    """
    Исправляет неправильные краткие формы имени робота.

    Лёгкая модель любит сокращать «Валера» до «Лера» — но так робота зовут
    неправильно (да и «Лерой» зовут девочек). Заменяем на «Валера» в нужном
    падеже. Заменяем только ОТДЕЛЬНЫЕ слова, чтобы не испортить «Валера».
    """
    if not text:
        return text
    import re as _re

    # Падежные формы «Лера» → соответствующая форма «Валера».
    forms = {
        "лера": "Валера", "леры": "Валеры", "лере": "Валере",
        "леру": "Валеру", "лерой": "Валерой", "лерою": "Валерою",
        # Лёгкая модель портит имена руководства — а их произносят вслух.
        "санислав": "Станислав", "станислова": "Станислава",
        "алексевеч": "Алексеевич", "кужд": "Кудж",
        # «Куж» — самая частая потеря буквы «д» в фамилии ректора.
        "куж": "Кудж",
        # Модель «досочиняет» фамилию до похожего слова — тоже правим,
        # иначе робот вслух называет несуществующего человека.
        "кудышев": "Кудж", "куджев": "Кудж", "куджеш": "Кудж",
        "куджевич": "Кудж", "куджа": "Куджа",
        "сигав": "Сигов", "сигово": "Сигова",
    }
    pattern = _re.compile(r"\b(" + "|".join(forms) + r")\b", _re.IGNORECASE)

    def _sub(match):
        src = match.group(1)
        dst = forms[src.lower()]
        # Сохраняем регистр: в начале предложения — с заглавной.
        if src[0].isupper():
            return dst
        return dst[0].lower() + dst[1:]

    return pattern.sub(_sub, text)


def answer_guard(text: str) -> str:
    """
    Страховка на СЛУЧАЙ, если модель всё-таки заговорила о политике/войне
    (вопрос мог проскочить мимо проверки). Заменяем такой ответ на отказ.

    Список здесь КОРОЧЕ, чем для вопросов: берём только однозначные слова,
    чтобы не портить нормальные ответы (например, про военную кафедру).
    """
    if not text:
        return text
    import re as _re

    # Уже отказ — второй раз не трогаем.
    low = text.lower()
    if "не обсуждаю" in low or "не буду обсуждать" in low:
        return text

    strict_patterns = [
        r"\bпутин", r"\bзеленск", r"\bнавальн", r"\bлавров", r"\bмишустин",
        r"\bспецопераци", r"\bсво\b", r"\bукраин", r"\bмобилизаци",
        r"\bнато\b", r"\bбпла\b", r"\bобстрел",
        r"\bгосдум", r"\bкремл", r"\bоппозици", r"\bмитинг",
        r"\bфашизм", r"\bнацизм",
    ]
    if any(_re.search(p, low) for p in strict_patterns):
        return FORBIDDEN_REFUSAL
    return text


def strip_filler(text: str) -> str:
    """
    Убирает «воду» и дежурные вступления в НАЧАЛЕ ответа.

    Пример: «О, отличный вопрос! МИРЭА — это вуз.» → «МИРЭА — это вуз.»
    Лёгкая модель почти всегда начинает с такой фразы, а для озвучки это
    лишние секунды и «вода». Чистим кодом — надёжнее, чем просьбой в промпте.
    """
    if not text:
        return text
    import re as _re

    out = text.strip()
    changed = True
    # Повторяем, пока в начале есть дежурные фразы (их может быть несколько).
    while changed:
        changed = False
        low = out.lower()
        for filler in _FILLER_OPENERS:
            if low.startswith(filler):
                out = out[len(filler):].lstrip(" ,;:—-–.!?»«\"'")
                # Вернём заглавную букву в начало.
                if out and out[0].islower():
                    out = out[0].upper() + out[1:]
                changed = True
                break
    return out.strip()


# ---------------------------------------------------------------------------
# Служебная («мета»-)речь, которую НЕЛЬЗЯ произносить вслух
# ---------------------------------------------------------------------------
# Промпт поиска требует опираться на документы, и модель честно отвечает
# «Согласно источникам, указанным в контексте…», «В предоставленной информации
# нет…». Для живого экскурсовода это звучит как отчёт и сразу выдаёт робота.
# Поэтому вырезаем такие обороты КОДОМ: промпт — просьба, код — гарантия.
#
# Убираем только служебные формулировки, а не смысл: «Согласно источникам,
# ректор — Кудж» → «Ректор — Кудж».
_META_SPEECH_PATTERNS = [
    r"согласно\s+(?:предоставленн\w*\s+|имеющ\w*\s+)?(?:источник\w*|контекст\w*|"
    r"базе\s+знаний|документ\w*|информации|данн\w*)",
    # «…указанным/приведённым в контексте» — довесок к «согласно источникам».
    # Без него оставалось сломанное «Указанным текущим ректором является…».
    r"(?:указанн\w*|приведенн\w*|приведённ\w*|описанн\w*|представленн\w*|"
    r"названн\w*)\s+в\s+(?:контекст\w*|источник\w*|документ\w*|базе\s+знаний)",
    # «Судя по предоставленной информации, …» — лёгкая модель так начинает
    # почти каждый ответ. ВАЖНО: этот шаблон должен стоять ДО общего «по
    # предоставленной информации», иначе тот съест хвост, и висеть останется
    # «Судя, в настоящее время…».
    r"судя\s+по\s+(?:предоставленн\w*|имеющ\w*|полученн\w*)?\s*(?:информации|данн\w*|контекст\w*|всему)",
    r"исходя\s+из\s+(?:предоставленн\w*|имеющ\w*)?\s*(?:информации|данн\w*|контекст\w*)",
    r"по\s+(?:имеющ\w*|предоставленн\w*|полученн\w*)\s+(?:данн\w*|информации|сведени\w*)",
    r"в\s+(?:предоставленн\w*|полученн\w*)\s+(?:контекст\w*|информации|данн\w*|материал\w*)",
    r"в\s+(?:контекст\w*|базе\s+знаний|документ\w*|источник\w*)",
    r"(?:как\s+)?указано\s+в\s+(?:источник\w*|контекст\w*|документ\w*)",
    r"(?:источник\w*|документ\w*|база\s+знаний)\s+(?:указыва\w+|говор\w+|сообща\w+|содерж\w+)",
    r"на\s+основании\s+(?:предоставленн\w*\s+)?(?:информации|данн\w*|контекст\w*)",
    r"по\s+(?:информации|данн\w*)\s+из\s+(?:базы|источник\w*|документ\w*)",
    # Остаётся после удаления «В базе знаний …»: «Указано, что бюджетных мест 3 088».
    r"(?:указано|отмечено|сказано|написано|сообщается|говорится)\s*,\s*что",
    # Метки цитирования: [Source 1], [Источник 1], [1], [И234], [1, 2].
    # В ответе вслух они не звучат, а лишь мусорят текст.
    r"\[\s*(?:source|источник|и)?\s*\d+(?:\s*[,;/-]\s*\d+)*\s*\]",
    # Модель любит начинать ответ служебной меткой из-за инструкции «сначала
    # факт»: «Главный факт: ректор — Кудж». Вслух такая метка не нужна.
    r"^(?:главн\w*\s+факт|важн\w*\s+факт|коротк\w*\s+ответ|ответ|итог|итого|"
    r"суть|кратко|справка|информация)\s*[:\-–—]\s*",
]

# Ответы-«отписки» о том, что данных нет. Вместо канцелярского «нет данных о
# стоимости» живой робот должен коротко признать, что не знает.
_NO_ANSWER_RE = (
    r"(?i)^\s*(?:"
    r"нет\s+(?:данн\w*|информации|сведени\w*|инфо)"
    r"|данн\w*\s+(?:нет|отсутству\w+|не\s+найден\w*)"
    r"|информации\s+(?:нет|отсутству\w+)"
    r"|(?:не\s+удалось|невозможно)\s+найти"
    r"|отсутству\w+\s+информаци\w*"
    r")"
)
NO_ANSWER_REPLY = "Этого я не знаю."

# ---------------------------------------------------------------------------
# ЗАЩИТА ОТ «АНАЛИЗА ГОСТЯ» ВМЕСТО ОТВЕТА
# ---------------------------------------------------------------------------
# На реплику «Уверен, что ты это хочешь? Я хотел рассказать.» модель выдала
# не ответ, а разбор поведения собеседника: «Суть ситуации — ваша реплика это
# не запрос на информацию, а коммуникативный сигнал: вы мягко указываете, что
# в предыдущем обмене роли перепутались». Это грубейшая ошибка для робота-
# экскурсовода: он должен ОТВЕЧАТЬ, а не анализировать гостя.
#
# Промпты это запрещают, но локальные модели и облачные иногда «плывут»,
# поэтому есть второй рубеж — проверка готового ответа.
_ANALYSIS_MARKERS = (
    "коммуникативн",
    "суть ситуации",
    "ваша реплика",
    "твоя реплика",
    "вы мягко",
    "ты мягко",
    "роли перепутал",
    "не запрос на информацию",
    "а не запрос на информацию",
    "в предыдущем обмене",
    "в предыдущей реплике",
    "судя по формулировке",
    "данная фраза",
    "эта фраза означает",
    "ваш вопрос можно",
    "твой вопрос можно",
    "интерпретир",
    "коммуникативный сигнал",
    "анализируя",
    "психологическ",
)

# Живая реакция на непонятную/битую фразу. Это НЕ «я не знаю» (то про факты),
# а честное «не разобрал» — обычно причина в распознавании речи.
UNRECOGNIZED_REPLY = "Повторите, пожалуйста."


def looks_like_analysis(text: str) -> bool:
    """
    Похоже ли, что вместо ответа модель разбирает поведение гостя.

    Срабатывает при ДВУХ и более маркерах-анализе: одно случайное слово
    («фраза») ещё не приговор, а набор «суть ситуации / ваша реплика /
    коммуникативный сигнал» — точно разбор, а не ответ.
    """
    if not text:
        return False
    low = text.lower()
    hits = sum(1 for m in _ANALYSIS_MARKERS if m in low)
    return hits >= 2


def strip_analysis(text: str) -> str:
    """
    Убирает «Ответ на вопрос: …» и заменяет разбор поведения на живую фразу.

    Модель часто начинает ответ с «Ответ на вопрос: «…»» и только потом
    переходит к сути (или к разбору). Такой префикс вслух звучит как робот-
    диктофон, поэтому вырезаем его и отвечаем по-человечески.
    """
    if not text:
        return text
    import re as _re

    out = text.strip()

    # «Ответ на вопрос: «...» Суть ситуации …» → выбрасываем служебную шапку.
    out = _re.sub(
        r'(?is)^\s*ответ\s+на\s+вопрос\s*:?\s*[«"“][^»"”]{0,300}[»"”]\s*',
        "",
        out,
    )
    out = _re.sub(r"(?is)^\s*ответ\s+на\s+вопрос\s*:\s*", "", out)
    out = out.strip()

    # Остался чистый разбор поведения — значит, отвечать было нечем: фраза
    # гостя, скорее всего, не распознана. Просим повторить.
    if looks_like_analysis(out):
        return UNRECOGNIZED_REPLY

    return out


# ---------------------------------------------------------------------------
# «НЕ РАЗОБРАЛ» — ТОЛЬКО ДЛЯ НАСТОЯЩЕЙ БЕССМЫСЛИЦЫ
# ---------------------------------------------------------------------------
# Правило «фраза не понятна → переспроси» модели поняли СЛИШКОМ буквально и
# начали сбегать от ответа: на нормальную человеческую речь («Не знаю, по
# сетям там что-нибудь надо», «Адреса она выдумывает какие-то») робот отвечал
# «Прости, я не разобрал — повтори, пожалуйста?». Гость видит поломку.
#
# Теперь «не разобрал» разрешено ТОЛЬКО когда в реплике нет осмысленных слов.
# Проверяем это кодом, а не доверяем модели: если слова есть — переспрашиваем
# по существу или просим модель ответить ещё раз.
_UNRECOGNIZED_RE = re.compile(
    r"(?i)(?:"
    r"не\s+(?:разобрал|расслышал)"
    r"|повтори(?:те)?,?\s+пожалуйста"
    r"|не\s+понял,?\s+(?:что\s+(?:ты|вы|имеется\s+в\s+виду)|о\s+ч[её]м)"
    r")"
)


def is_unrecognized_reply(text: str) -> bool:
    """Ответ модели — это отписка «я не разобрал»?"""
    return bool(text) and bool(_UNRECOGNIZED_RE.search(text))


def is_gibberish(text: str) -> bool:
    """
    Похоже ли, что распознавание речи выдало бессмыслицу.

    Бессмыслица — это когда осмысленных слов (3+ буквы) почти нет, или когда
    одно и то же слово повторяется («бал-бал-бал-бал»). Всё остальное — живая
    речь, пусть и сбивчивая: на неё надо отвечать, а не просить повторить.
    """
    if not text:
        return True
    # Слова от 2 букв: «да», «ну», «ок» — осмысленные короткие ответы,
    # их нельзя считать мусором.
    words = re.findall(r"[А-Яа-яЁёA-Za-z]{2,}", text)
    if not words:
        return True
    low = [w.lower() for w in words]
    # Одно и то же слово/слог подряд — типичный сбой распознавания
    # («бал-бал-бал»), в том числе одно растянутое «аааа».
    if len(low) >= 3 and len(set(low)) == 1:
        return True
    if len(low) == 1 and len(set(low[0])) == 1:
        return True
    # Меньше двух осмысленных слов на длинную реплику — тоже мусор.
    return False


# Уточняющие вопросы: их говорим вместо «не разобрал», когда слова понятны.
CLARIFY_MIREA = (
    "Уточните, пожалуйста: Вас интересует поступление, институты, "
    "общежития или что-то ещё?"
)
CLARIFY_GENERIC = "Уточните, пожалуйста — я не совсем понял, что Вы имеете в виду."


def clarify_reply(mirea_related: bool = False) -> str:
    """Короткий уточняющий вопрос вместо отписки «не разобрал»."""
    return CLARIFY_MIREA if mirea_related else CLARIFY_GENERIC


def normalize_name(text: str) -> str:
    """
    Нормализует обращения по имени робота в тексте ВОПРОСА.

    Гости часто называют робота «Лера» (сокращают «Валера») или путают имя.
    В базе знаний такого человека нет, и поиск уходил искать посторонних
    людей: на «Лера, ты работаешь?» модель уверенно рассказывала про
    телеведущую Валерию Кудрявцеву. Поэтому в запросе заменяем короткие
    формы на полное имя — дальше робот говорит про себя.
    """
    if not text:
        return text
    import re as _re

    aliases = {
        "лера": "Валера", "леры": "Валеры", "лере": "Валере",
        "леру": "Валеру", "лерой": "Валерой", "лерою": "Валерою",
        "лерка": "Валера",
        # Варианты распознавания речи
        "валека": "Валера", "валераа": "Валера", "валела": "Валера",
    }
    pattern = _re.compile(r"\b(" + "|".join(aliases) + r")\b", _re.IGNORECASE)

    def _sub(m):
        src = m.group(1)
        dst = aliases[src.lower()]
        return dst if src[0].isupper() else dst[0].lower() + dst[1:]

    return pattern.sub(_sub, text)


# Вопросы О САМОМ РОБОТЕ (не про вуз). Их нельзя отправлять в поиск по базе
# знаний: там про личность робота ничего нет, поэтому поиск подсовывал
# случайные чанки, и модель отвечала «Этого я не знаю» либо находила
# постороннего человека с похожим именем.
_SELF_QUESTION_RE = re.compile(
    r"(?i)(?:"
    r"\bты\s+(?:кто|робот|человек|живой|настоящий|умный)\b"
    r"|\bкто\s+(?:ты|вы)\b"
    r"|\bкак\s+(?:тебя|вас)\s+зовут\b"
    r"|\b(?:тво[её]|ваше)\s+имя\b"
    r"|\bты\s+(?:работаешь|работаете|здесь|тут|на\s+работе|готов)\b"
    r"|\bчто\s+ты\s+(?:умеешь|можешь|знаешь)\b"
    r"|\bрасскажи\s+(?:о\s+себе|про\s+себя)\b"
    r"|\bты\s+(?:меня|нас)\s+(?:слышишь|понимаешь|видишь)\b"
    r"|\b(?:ты|вы)\s+(?:бо[тт]|робот)\b"
    r"|\b(?:Валера|Лера)\s*,\s*"
    r"|\bтебя\s+зовут\b"
    r"|\b(?:какой|какого)\s+(?:у\s+тебя\s+)?(?:рост|вес)\b"
    r"|\bсколько\s+(?:ты|вы)\s+(?:весишь|весите|стоишь|стоите|ходишь|бегаешь)\b"
    r"|\bсколько\s+тебе\s+лет\b"
    r"|\bгде\s+ты\s+(?:живёшь|живешь|находишься|работаешь|стоишь)\b"
    r"|\bкто\s+тебя\s+(?:создал|сделал|собрал|придумал|построил)\b"
    r"|\bиз\s+чего\s+ты\b"
    r"|\bкак\s+ты\s+(?:устроен|ходишь|видишь|двигаешься|работаешь)\b"
    r"|\bкакой\s+ты\s+(?:робот|модели)\b"
    r")"
)
_INTRO_REQUEST_RE = re.compile(
    r"\bпредставь(?:ся|тесь)(?:[\s,]+пожалуйста)?[.!?…\s]*$", re.IGNORECASE,
)
_VERSION_RE = re.compile(r"\b(?:верси\w*|сборк\w*|ревизи\w*|релиз\w*|редакци\w*)\b", re.IGNORECASE)
_FIRMWARE_RE = re.compile(r"\b(?:прошив?к\w*|firmware)\b", re.IGNORECASE)
_PROMPT_TOPIC_RE = re.compile(
    r"\b(?:пром(?:пт?|т)(?:а|у|ом|е|ы|ов|ам|ами|ах)?|системн\w+\s+инструкци\w*)\b",
    re.IGNORECASE,
)
_PROMPT_DEFINITION_RE = re.compile(
    r"\b(?:что\s+(?:такое|значит|означает)|объясни(?:те)?|зачем\s+нужен)\b", re.IGNORECASE
)
_PROMPT_VERSION_RE = re.compile(r"^Версия системного промпта — v(\d+(?:\.\d+)+)\.$", re.MULTILINE)
_VERSION_COMPONENT_RE = re.compile(
    r"\b(?:gemma|джемм\w*|ollama|оллам\w*|"
    r"kairos|кайрос\w*|silero|силеро|xtts|икс[ -]?ти[ -]?ти[ -]?эс|coqui|коки|"
    r"ros\s*2?|рос(?:\s*(?:два|2))?|unitree\s+sdk|пакет\w*\s+юнитри|"
    r"распознаван\w+\s+речи|синтез\w*\s+речи|языков\w+\s+модел\w*)\b",
    re.IGNORECASE,
)
_VERSION_SELF_RE = re.compile(
    r"\b(?:(?:у|на)\s+(?:тебя|тебе|вас|робота|роботе|валеры)|"
    r"(?:тво|сво)(?:й|я|[её]|и|его|ей|ему|им|ими|их|ю|[её]м)|"
    r"ваш(?:а|е|и|его|ей|ему|им|ими|их|у|ем)?|"
    r"(?:ты|вы)\s+(?:какой|какая|на|использу\w*|работа\w*)|"
    r"какой\s+(?:ты|вы))\b",
    re.IGNORECASE,
)
_VERSION_REQUEST_RE = re.compile(
    r"\b(?:какая|какой|какую|какие|каком|каким|что(?:\s+у\s+(?:тебя|вас))?\s+за|"
    r"как\s+называется|номер|назови(?:те)?|"
    r"скажи(?:те)?|подскажи(?:те)?|установлен\w*|стоит)\b",
    re.IGNORECASE,
)
_BARE_VERSION_RE = re.compile(
    r"[\s,]*(?:(?:а|ну|да|тогда)[\s,]+)*(?:какая\s+(?:(?:у\s+(?:тебя|вас)|сейчас)\s+)*версия|"
    r"что\s+за\s+версия|версия\s+какая|текущая\s+версия)(?:\s+у\s+(?:тебя|вас))?[?!.\s]*",
    re.IGNORECASE,
)
_BARE_FIRMWARE_RE = re.compile(
    r"[\s,]*(?:(?:а|ну)\s+)?(?:(?:какая\s+(?:сейчас\s+)?(?:версия\s+)?|"
    r"что\s+за\s+|(?:на|с)\s+какой\s+|номер\s+)?прошив?к\w*"
    r"(?:\s+какая)?)(?:\s+робота)?(?:\s+сейчас)?[?!.\s]*",
    re.IGNORECASE,
)
_FIRMWARE_OTHER_REQUEST_RE = re.compile(
    r"\b(?:обнов\w*|переустанов\w*|скача\w*|перепроши\w*|"
    r"(?:как|где)\s+(?:узнать|посмотреть|проверить|найти)|и\s+(?:какая|какой|как|где|что))\b",
    re.IGNORECASE,
)


def asks_robot_firmware_version(text: str) -> bool:
    """Проверяет, спрашивает ли гость именно номер прошивки Валеры."""
    return bool(
        (_BARE_FIRMWARE_RE.fullmatch(text) or (
            _FIRMWARE_RE.search(text) and _VERSION_SELF_RE.search(text)
            and (_VERSION_RE.search(text) or _VERSION_REQUEST_RE.search(text))
        ))
        and not _VERSION_COMPONENT_RE.search(text)
        and not _PROMPT_TOPIC_RE.search(text)
        and not _FIRMWARE_OTHER_REQUEST_RE.search(text)
    )


def is_self_question(text: str) -> bool:
    """
    Спрашивают ли О САМОМ РОБОТЕ (имя, «ты кто», «ты работаешь»).

    Такие вопросы отвечаем напрямую из системной инструкции личности, без
    поиска по базе знаний — иначе робот говорит «Этого я не знаю» или
    выдумывает посторонних людей.
    """
    if not text:
        return False
    return bool(
        _SELF_QUESTION_RE.search(text)
        or _INTRO_REQUEST_RE.search(text)
        or _BARE_VERSION_RE.fullmatch(text)
        or _BARE_FIRMWARE_RE.fullmatch(text)
        or (_VERSION_RE.search(text) and (
            _VERSION_SELF_RE.search(text) or _VERSION_COMPONENT_RE.search(text)
            or _FIRMWARE_RE.search(text) or _PROMPT_TOPIC_RE.search(text)
        ))
        or (_FIRMWARE_RE.search(text) and _VERSION_REQUEST_RE.search(text))
    )


def prompt_version_answer(question: str, previous_question: str) -> str:
    """Отвечает на вопрос о версии промпта по Modelfile."""
    if not _VERSION_RE.search(question):
        return ""
    if _FIRMWARE_RE.search(question) or _VERSION_COMPONENT_RE.search(question):
        return ""
    mentions_prompt = bool(_PROMPT_TOPIC_RE.search(question))
    if not mentions_prompt and not (
        _BARE_VERSION_RE.fullmatch(question) and _PROMPT_TOPIC_RE.search(previous_question)
    ):
        return ""

    try:
        version = _PROMPT_VERSION_RE.search(Path(__file__).with_name("Modelfile").read_text(encoding="utf-8"))
    except OSError:
        version = None
    if version is None:
        return "Номер версии системного промпта не указан. [эмоция:спокойствие]"

    from num2words import num2words

    spoken = " точка ".join(num2words(int(part), lang="ru") for part in version.group(1).split("."))
    answer = f"Версия моего системного промпта — {spoken}. [эмоция:уверенность]"
    if mentions_prompt and _PROMPT_DEFINITION_RE.search(question):
        answer = ("Системный промпт — это правила, которые задают мою роль и то, как я отвечаю. "
                  "[эмоция:спокойствие] " + answer)
    return answer


# Неверные расшифровки аббревиатуры МИРЭА, которые модель иногда выдумывает.
# Реально МИРЭА — Московский институт радиотехники, электроники и автоматики
# (с 2015 г. — РТУ МИРЭА). А «Московский инженерно-физический университет» —
# это МИФИ, ДРУГОЙ вуз: лёгкая модель путает похожие аббревиатуры и уверенно
# произносит чужое название вслух. Заменяем кодом, потому что промпт это не
# гарантирует.
_WRONG_VUZ_NAME_PATTERNS = [
    r"Московск\w+\s+инженерно[\s-]физическ\w+\s+университет\w*",
    r"инженерно[\s-]физическ\w+\s+университет\w*",
    r"Московск\w+\s+физико[\s-]техническ\w+\s+институт\w*",
    r"Московск\w+\s+государственн\w+\s+университет\w*",
]


def fix_vuz_name(text: str) -> str:
    """
    Исправляет выдуманные расшифровки «МИРЭА».

    Модель регулярно подставляет чужой вуз (МИФИ — «Московский
    инженерно-физический университет»), и робот вслух называет неправильное
    имя. Заменяем на верное и короткое «РТУ МИРЭА».
    """
    if not text:
        return text
    import re as _re

    out = text
    for pat in _WRONG_VUZ_NAME_PATTERNS:
        out = _re.sub(pat, "РТУ МИРЭА", out, flags=_re.IGNORECASE)
    # «РТУ МИРЭА (РТУ МИРЭА)» — убираем дубль в скобках после замены.
    out = _re.sub(r"РТУ\s+МИРЭА\s*\((?:РТУ\s+МИРЭА|МИРЭА)\)", "РТУ МИРЭА", out,
                  flags=_re.IGNORECASE)
    return out


def strip_meta_speech(text: str) -> str:
    """
    Вырезает служебные обороты о «контексте», «источниках» и «базе знаний».

    Ответ читает синтезатор речи, поэтому такие вставки нельзя произносить:
    они превращают живого робота в диктофон для отчёта. Обороты могут стоять
    как в начале («Согласно источникам, ректор...»), так и в середине ответа,
    поэтому заменяем их везде, а не только в начале.
    """
    if not text:
        return text
    import re as _re

    out = text
    for pat in _META_SPEECH_PATTERNS:
        out = _re.sub(rf"(?i)(?<![\w]){pat}\s*[,:]?\s*", "", out)
    # Чистим следы удаления: двойные пробелы, пробел перед знаком, висячие
    # запятые («Ректор — Кудж, согласно документам.» → «…Кудж,»).
    out = _re.sub(r"\s{2,}", " ", out)
    out = _re.sub(r"[,;:]+\s*([.!?])", r"\1", out)
    out = _re.sub(r"([,.!?;:])\1+", r"\1", out)
    out = _re.sub(r"\s+([.,!?;:])", r"\1", out)
    out = _re.sub(r"[\s,;:]+$", "", out)
    out = _re.sub(r"^[\s,;:\-–—]+", "", out)
    out = out.strip()
    # Если в итоге осталась только «отписка» про отсутствие данных — заменяем
    # её на живую фразу.
    #
    # ВАЖНО: заменяем ТОЛЬКО когда от ответа действительно ничего не осталось
    # (короткая фраза). Раньше условие срабатывало по одному началу, и
    # содержательный ответ с оговоркой «в данных нет точной цифры, но известно,
    # что…» целиком превращался в «Этого я не знаю» — гость не получал ничего.
    if out and _re.match(_NO_ANSWER_RE, out) and len(out.split()) <= 7:
        return NO_ANSWER_REPLY
    # Точка в конце, если текст остался без завершения.
    if out and out[-1] not in ".!?…":
        out += "."
    # Возвращаем заглавную букву в начало.
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out


# ---------------------------------------------------------------------------
# «РТУ МИРЭА» НЕ ЧЕРЕЗ СЛОВО И НИКАКОГО «СТОЮ В ХОЛЛЕ»
# ---------------------------------------------------------------------------
# Модель любила через слово повторять название вуза и вставлять «стою в холле»,
# «жду гостей», «я Валера, робот-экскурсовод» в ответ на любой вопрос. Промпт
# просит этого не делать, но 8B-модель слушается не всегда, поэтому чистим кодом.
_VUZ_NAME_RE = re.compile(
    r"(?i)(?:\b(в|во|из|от|про|об|о|для|со|с|к)\s+)?[«\"]?(?:РТУ\s+)?МИРЭА[»\"]?(?![\w-])"
)
_HALL_SENT_RE = re.compile(
    r"(?i)[^.!?…]*\b(?:стою\s+(?:тут\s+|здесь\s+|прямо\s+)?в\s+холле|жду\s+гостей|стою\s+в\s+холле)\b[^.!?…]*[.!?…]*\s*"
)
_INTRO_SENT_RE = re.compile(r"(?i)^\s*(?:привет[!,. ]*|здравствуйте[!,. ]*)?\s*я\s+валера[^.!?…]*робот[^.!?…]*[.!?…]+\s*")
_ans_ctx = {"vuz_in_question": False, "prev_named": False, "intro_ok": False, "prev_question": False}

_VUZ_REPL = {
    "в": "у нас", "во": "у нас", "из": "у нас", "от": "от нас",
    "про": "про наш университет", "о": "о нашем университете", "об": "о нашем университете",
    "для": "для нашего университета", "со": "с нашим университетом", "с": "с нашим университетом",
    "к": "к нашему университету",
}


def set_answer_context(question: str, intro_ok: bool = False) -> None:
    """Запоминает, что известно про реплику: назван ли вуз гостем и уместно ли представление."""
    _ans_ctx["vuz_in_question"] = bool(_VUZ_NAME_RE.search(question or ""))
    _ans_ctx["intro_ok"] = intro_ok


def note_answer_given(answer: str) -> None:
    """После ответа: был ли в нём назван вуз (тогда следующий ответ уже не называет)."""
    done = finalize_answer(answer) or ""
    _ans_ctx["prev_named"] = bool(_VUZ_NAME_RE.search(done))
    _ans_ctx["prev_question"] = done.rstrip().endswith("?")


def limit_vuz_mentions(text: str) -> str:
    """Оставляет название вуза не чаще одного раза — и ни разу, если оно уже звучало."""
    if not text:
        return text
    allowed = 0 if (_ans_ctx["vuz_in_question"] or _ans_ctx["prev_named"]) else 1
    seen = 0

    def _sub(m: "re.Match") -> str:
        nonlocal seen
        seen += 1
        if seen <= allowed:
            return m.group(0)
        prep = (m.group(1) or "").lower()
        before = text[:m.start()].rstrip()
        at_start = (not before) or before[-1] in ".!?…"
        if prep:
            rep = _VUZ_REPL.get(prep, "у нас")
            return rep[:1].upper() + rep[1:] if at_start else rep
        if re.search(r"(?i)(?:экскурсовод|ассистент|робот)$", before):
            return ""
        if at_start:
            return "Наш университет"
        return "нашего университета"

    out = _VUZ_NAME_RE.sub(_sub, text)
    return re.sub(r"\s{2,}", " ", re.sub(r"\s+([,.!?;:])", r"\1", out)).strip()


_TRAIL_Q_RE = re.compile(r"(?s)^(.*?[.!…])\s+([^.!?…]*\?)\s*$")


def limit_trailing_question(text: str) -> str:
    """
    Не заканчивает КАЖДЫЙ ответ встречным вопросом («Хотите узнать подробнее?»): если прошлый
    ответ уже закончился вопросом, завершающий вопрос в этом отбрасывается.
    """
    if not text or not _ans_ctx["prev_question"]:
        return text
    m = _TRAIL_Q_RE.match(text)
    if m and len(m.group(1).split()) >= 3:
        return m.group(1).strip()
    return text


def strip_hall_talk(text: str) -> str:
    """Убирает «стою в холле», «жду гостей» и непрошеное «Я Валера, робот-экскурсовод…»."""
    if not text:
        return text
    if _ans_ctx["intro_ok"]:      # гость сам спросил «кто ты / где ты» — это уместно
        return text
    out = _HALL_SENT_RE.sub("", text).strip()
    out = _INTRO_SENT_RE.sub("", out).strip()
    return out or "Слушаю Вас."


# ---------------------------------------------------------------------------
# ЭМОЦИИ: модель возвращает метку, голос её ОТРАЖАЕТ
# ---------------------------------------------------------------------------
# Просить модель «говорить эмоционально» бесполезно — ответ читает синтезатор,
# эмоция должна быть параметром синтеза. Поэтому модель в КОНЦЕ ответа ставит
# служебную метку вида ``[эмоция:радость]``, а код:
#   1) вырезает метку из текста (вслух она не звучит и в пульт не попадает);
#   2) меняет темп речи и громкость через XTTS (``speed``) — это и есть
#      «смена тона», доступная без отдельной эмоциональной модели.
# Метка необязательна: если модель её не поставила, голос звучит как обычно.
_EMOTION_RE = re.compile(
    r"[\[(]\s*(?:эмоция|emotion)\s*[:：]\s*([^\])]{2,24})\s*[\])]\s*",
    re.IGNORECASE,
)
_INCOMPLETE_EMOTION_RE = re.compile(
    r"[\[(]\s*(?:эмоция|emotion)\s*[:：][^\])]*$", re.IGNORECASE
)

# Темп по эмоции. Значения подобраны на слух: живая речь в радости быстрее,
# разговор о серьёзном (стипендии, экзамены) — медленнее и ровнее.
# ``volume`` — множитель громкости (1.0 = как обычно).
EMOTION_STYLE = {
    "радость":       {"speed": 1.08, "volume": 1.00, "pitch": 1.050},
    "воодушевление": {"speed": 1.10, "volume": 1.02, "pitch": 1.065},
    "восторг":       {"speed": 1.12, "volume": 1.03, "pitch": 1.075},
    "дружелюбие":    {"speed": 1.04, "volume": 1.00, "pitch": 1.020},
    "тепло":         {"speed": 1.02, "volume": 1.00, "pitch": 1.015},
    "спокойствие":   {"speed": 1.00, "volume": 1.00, "pitch": 0.995},
    "уверенность":   {"speed": 1.01, "volume": 1.01, "pitch": 0.990},
    "серьёзность":   {"speed": 0.94, "volume": 1.00, "pitch": 0.962},
    "сочувствие":    {"speed": 0.96, "volume": 0.98, "pitch": 0.970},
    "задумчивость":  {"speed": 0.97, "volume": 0.98, "pitch": 0.982},
    "удивление":     {"speed": 1.06, "volume": 1.02, "pitch": 1.055},
    # Английские варианты: Gemma иногда отвечает «[emotion: joy]».
    "joy":           {"speed": 1.08, "volume": 1.00, "pitch": 1.050},
    "excited":       {"speed": 1.10, "volume": 1.02, "pitch": 1.065},
    "friendly":      {"speed": 1.04, "volume": 1.00, "pitch": 1.020},
    "warm":          {"speed": 1.02, "volume": 1.00, "pitch": 1.015},
    "calm":          {"speed": 1.00, "volume": 1.00, "pitch": 0.995},
    "confident":     {"speed": 1.01, "volume": 1.01, "pitch": 0.990},
    "serious":       {"speed": 0.94, "volume": 1.00, "pitch": 0.962},
    "sympathy":      {"speed": 0.96, "volume": 0.98, "pitch": 0.970},
    "thoughtful":    {"speed": 0.97, "volume": 0.98, "pitch": 0.982},
    "surprised":     {"speed": 1.06, "volume": 1.02, "pitch": 1.055},
    "neutral":       {"speed": 1.00, "volume": 1.00, "pitch": 1.000},
}
DEFAULT_EMOTION = "дружелюбие"


def parse_emotion(text: str) -> tuple:
    """
    Вырезает метку эмоции из ответа.

    Возвращает ``(чистый_текст, эмоция|None)``. Понимает варианты написания:
    ``[эмоция:радость]``, ``(эмоция: радость)``, ``[Emotion: joy]`` (модель
    иногда отвечает по-английски). Неизвестную эмоцию возвращает как есть —
    вызывающий код просто не найдёт её в ``EMOTION_STYLE``.
    """
    if not text:
        return text, None
    found = None

    def _grab(match):
        nonlocal found
        if found is None:
            found = match.group(1).strip().lower().rstrip(".!?,;")
        return ""

    clean = _EMOTION_RE.sub(_grab, text)
    clean = _INCOMPLETE_EMOTION_RE.sub("", clean)
    clean = re.sub(r"\s{2,}", " ", clean).strip()
    return clean, found


def emotion_style(emotion: Optional[str]) -> dict:
    """
    Параметры синтеза для эмоции.

    Если метки нет (модель обрезалась по лимиту токенов, ответ короткий или
    это служебная фраза), берём ``DEFAULT_EMOTION`` — дружелюбие. Голос тогда
    чуть живее нейтрального, но без риска «переиграть». Плоским он не станет
    ни при каких условиях.
    """
    if not emotion:
        return dict(EMOTION_STYLE.get(DEFAULT_EMOTION, {"speed": 1.0, "volume": 1.0, "pitch": 1.0}))
    return EMOTION_STYLE.get(emotion, dict(EMOTION_STYLE.get(DEFAULT_EMOTION, {"speed": 1.0, "volume": 1.0, "pitch": 1.0})))


def split_answer_for_tts(text: str) -> list:
    """
    Делит ответ на предложения и достаёт эмоцию ИЗ КАЖДОГО.

    Возвращает список пар ``(текст_предложения, стиль_синтеза)``. Метки
    ``[эмоция:…]`` убираются из текста — вслух они звучать не должны — но
    остаются привязанными к своей фразе, поэтому интонация может меняться
    по ходу ответа (факт — спокойно, хорошая новость — радостно). Если у
    фразы метки нет, берётся ``DEFAULT_EMOTION``: голос останется живым,
    но не «переиграет».
    """
    pairs = []
    for part in split_into_sentences(text or ""):
        clean, name = parse_emotion(part)
        clean = finalize_answer(clean)
        if clean:
            pairs.append((clean, emotion_style(name)))
    return pairs


_INCOMPLETE_ANSWER_END_RE = re.compile(r"\b(?:и|или|но|к|из|с|у|о|об|в|на|для|по|про)[.!?…\s]*$", re.I)
_TRANSITION_START_RE = re.compile(
    r"^(?:(?:давайте\s+)?(?:переходим|перейд[её]м|начн[её]м|продолжим|"
    r"посмотрим)|следующ\w+\s+(?:остановк\w+|раздел))\b",
    re.I,
)
_OFFER_QUESTION_RE = re.compile(r"^(?:хотите\s+(?:узнать|посмотреть)|рассказать\s+(?:вам|подробнее|про|о|об)).*\?$", re.I)


def is_stalled_continuation(text: str, previous: str) -> bool:
    """Проверяет, появился ли в продолжении содержательный ответ."""
    plain, _ = parse_emotion(text or "")
    previous, _ = parse_emotion(previous or "")
    words = re.findall(r"\w+", plain.casefold())
    if not words or words == re.findall(r"\w+", previous.casefold()) or _INCOMPLETE_ANSWER_END_RE.search(plain):
        return True
    parts = split_into_sentences(plain)
    return all(
        _OFFER_QUESTION_RE.fullmatch(part)
        or (_TRANSITION_START_RE.match(part) and not re.search(r"[,;:]", part))
        for part in parts
    )


def finalize_answer(
    text: str, max_words: Optional[int] = None, keep_emotion: bool = False
) -> str:
    """
    Приводит ответ к финальному виду для озвучки: вырезает служебную речь
    («согласно источникам»), поправляет имя, убирает английские слова и «воду»,
    затем подрезает до лимита слов. Единая функция для всех путей, чтобы ответ
    всегда был чистым и коротким.

    Метку эмоции ``[эмоция:…]`` по умолчанию вырезаем: она служебная, вслух
    звучать не должна. Но если ответ ещё будет резаться на предложения для
    потоковой озвучки, метки нужно СОХРАНИТЬ (``keep_emotion=True``) — иначе
    после этой функции уже не понять, какая интонация к какому предложению
    относилась. Тогда метки снимает вызывающий код, по одной на фразу.
    """
    max_words = max_words or current_answer_words()
    plain, _ = parse_emotion(text)
    if keep_emotion and _EMOTION_RE.search(text or ""):
        checked = answer_guard(strip_meta_speech(strip_analysis(plain)))
        if checked in (FORBIDDEN_REFUSAL, UNRECOGNIZED_REPLY):
            return finalize_answer(checked, max_words)

        # Проверяем повтор названия на всём ответе, пока предложения ещё вместе.
        parts = []
        remaining = max_words
        for part in split_into_sentences(limit_vuz_mentions(text)):
            sentence, emotion = parse_emotion(part)
            if not sentence:
                continue
            clean = finalize_answer(sentence, max_words)
            if not clean or (clean == "Слушаю Вас." and sentence != clean):
                continue
            words = len(clean.split())
            if words > remaining:
                break
            parts.append((clean, emotion))
            remaining -= words

        if parts:
            clean_answer = " ".join(sentence for sentence, _ in parts)
            if len(parts) > 1 and limit_trailing_question(clean_answer) != clean_answer:
                parts.pop()
            return " ".join(
                f"{sentence} [эмоция:{emotion}]" if emotion in EMOTION_STYLE else sentence
                for sentence, emotion in parts
            )

    return limit_words(
        strip_filler(
            clean_english(limit_trailing_question(limit_vuz_mentions(strip_hall_talk(fix_vuz_name(fix_name(
                answer_guard(strip_meta_speech(strip_analysis(plain)))
            ))))))
        ),
        max_words,
    )


# ---------------------------------------------------------------------------
# ЗАЩИТА ОТ ЗАВИСАНИЯ ЛОКАЛЬНОЙ МОДЕЛИ
# ---------------------------------------------------------------------------
# Наблюдаемая картина (Ollama 0.32.5, llama-server):
#   • /tokenize и /completion самого runner'а отвечают мгновенно;
#   • /api/chat и /api/generate на промптах от ~600 символов иногда виснут
#     навсегда (в логе: `chat prompt error … /tokenize: context canceled`);
#   • после первого таймаута слот генерации НЕ освобождается, и все следующие
#     запросы висят тоже — выглядит как «робот завис».
#
# Поэтому мало поднять таймаут: нужен «предохранитель». Если локальная модель
# не ответила за LOCAL_LLM_TIMEOUT, помечаем её сломанной на
# LOCAL_LLM_COOLDOWN секунд и больше не пытаемся — отвечаем через WebRAgent
# (DeepSeek). Гость получает ответ за секунды вместо тишины на минуту.
LOCAL_LLM_TIMEOUT = float(os.getenv("LOCAL_LLM_TIMEOUT", "15"))
LOCAL_LLM_COOLDOWN = float(os.getenv("LOCAL_LLM_COOLDOWN", "120"))

# Максимальный размер промта для локальной модели. Замерено на этой машине:
# до ~500 символов Ollama отвечает за 0.2–1 с, а на 600+ символов виснет
# навсегда (в логе: `/tokenize: context canceled`, затем `500` через таймаут).
# Причина — включённый в llama-server prompt cache с чекпоинтами
# (`context checkpoints enabled, min spacing = 8192` при n_ctx 4096).
# Поэтому промт ОБРЕЗАЕМ: лучше короткий живой ответ, чем вечная тишина.
# Лимит поднят 30.10.2026: проверено, что true_valera отвечает на промпт в 3000
# символов за ~0,6 с; страховка от зависания остаётся (LOCAL_LLM_TIMEOUT + карантин).
LOCAL_LLM_MAX_PROMPT = int(os.getenv("LOCAL_LLM_MAX_PROMPT", "4000"))
DIALOGUE_HISTORY_MESSAGES = 6

# Сколько символов черновика кладём в промт пересказа. Держим маленьким,
# чтобы весь промт уложился в LOCAL_LLM_MAX_PROMPT: модель виснет на
# длинных промптах, а обрезка середины съела бы инструкцию.
LOCAL_LLM_DRAFT_CHARS = int(os.getenv("LOCAL_LLM_DRAFT_CHARS", "1500"))

_local_llm_down_until = 0.0


def _fit_local_prompt(prompt: str, limit: int = None) -> str:
    """
    Обрезает промт под безопасный размер локальной модели.

    Режем СЕРЕДИНУ, сохраняя начало и конец: в наших промтах в начале стоит
    роль и вопрос, а в конце — метка «Ответ:», без которой модель не начнёт
    отвечать.
    """
    limit = limit or LOCAL_LLM_MAX_PROMPT
    if len(prompt) <= limit:
        return prompt
    head = int(limit * 0.35)
    tail = limit - head - 1
    return prompt[:head].rstrip() + "…" + prompt[-tail:].lstrip()


def local_llm_available() -> bool:
    """Не в «карантине» ли локальная модель после таймаута."""
    return time.time() >= _local_llm_down_until


def _mark_local_llm_down(reason: str = "") -> None:
    """Ставит локальную модель в карантин — чтобы не ждать её снова."""
    global _local_llm_down_until
    _local_llm_down_until = time.time() + LOCAL_LLM_COOLDOWN
    print(
        f"🛑 Локальная модель не отвечает ({reason}) — "
        f"{LOCAL_LLM_COOLDOWN:.0f} с работаю без неё.",
        file=sys.stderr,
    )


def ollama_think_off(model: str = "") -> dict:
    """
    Всегда просим Ollama НЕ «думать» (think=false).

    Gemma 4 / Qwen 3 по умолчанию тратят весь лимит токенов на «размышления» и отдают пустой
    content — робот молчит («Ollama не ответила (пустой ответ)»). Моделям без режима размышлений
    параметр безвреден (проверено на llama3.2), поэтому шлём его всем и не гадаем по имени модели
    (наша true_valera — это Gemma 4 под другим именем).
    """
    return {"think": False}


def _ollama_chat(
    prompt: str,
    model: str,
    num_predict: int,
    temperature: float,
    timeout: float = LOCAL_LLM_TIMEOUT,
    history: Optional[list[dict[str, str]]] = None,
) -> str:
    """
    Один запрос к Ollama через ``/api/chat``. Возвращает текст ответа.

    ПОЧЕМУ НЕ ``/api/generate``: он возвращает ещё и поле ``context`` — все
    токены промпта числами. При промпте ~2500 токенов это ~15 КБ, а VPN на
    loopback обрывает ответы крупнее ~16 КБ, и клиент висит до таймаута.
    ``/api/chat`` отдаёт только текст.

    Таймаут по умолчанию 15 секунд: живая модель отвечает за 0.3–2 с, и если
    за 15 с ответа нет — она заклинена, ждать дальше бессмысленно. Повтор
    делаем ТОЛЬКО при обрыве соединения (сервер перезапускается), но НЕ при
    таймауте: после таймаута слот Ollama остаётся занятым, и повтор снова
    провисит те же 15 с.

    Тело собираем через ``json.dumps(..., ensure_ascii=False)``: ``requests``
    с ``json=`` экранирует кириллицу в ``\\uXXXX`` и раздувает тело втрое.
    """
    import json as _json

    if not local_llm_available():
        return ""

    # Обрезаем промт под безопасный размер (см. LOCAL_LLM_MAX_PROMPT).
    prompt = _fit_local_prompt(prompt)
    remaining = LOCAL_LLM_MAX_PROMPT - len(prompt)
    recent = []
    for message in reversed((history or [])[-DIALOGUE_HISTORY_MESSAGES:]):
        if len(message["content"]) > remaining:
            break
        recent.append(message)
        remaining -= len(message["content"])
    if recent and recent[-1]["role"] == "assistant":
        recent.pop()  # Не начинаем историю с ответа без вопроса.

    payload = {
        "model": model,
        "messages": [*reversed(recent), {"role": "user", "content": prompt}],
        "stream": False,
        "keep_alive": -1,  # модель всегда прогрета (не выгружается в простое)
        "options": {"num_predict": num_predict, "temperature": temperature},
        **ollama_think_off(model),
    }
    body = _json.dumps(payload, ensure_ascii=False).encode("utf-8")

    last_error = None
    for attempt in (1, 2):
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                data=body,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}"
                # 500 — раннер перезагружается: короткая пауза помогает.
                if attempt == 1:
                    time.sleep(1.0)
                    continue
                break
            data = resp.json()
            text = ((data.get("message") or {}).get("content") or "").strip()
            if text:
                return text
            last_error = "пустой ответ"
            break
        except requests.Timeout as e:
            # Таймаут = модель заклинена. Повтор не поможет, только съест
            # ещё 15 секунд тишины.
            last_error = e
            _mark_local_llm_down("таймаут")
            break
        except requests.RequestException as e:
            last_error = e
            if attempt == 1:
                time.sleep(1.0)
                continue
            break

    print(f"⚠️ Ollama /api/chat не ответила ({last_error})", file=sys.stderr)
    return ""


def rephrase_answer_with_ai(
    question: str,
    raw_answer: str,
    max_words: Optional[int] = None,
) -> str:
    """
    Пересказывает «сырой» ответ RAG (со ссылками/сниппетами) через локальную
    модель Ollama в короткий разговорный ответ, пригодный для озвучки.

    Убирает URL, форматирование и избыточность — оставляет суть как живой
    ответ ассистента. Если Ollama недоступна — возвращает исходный текст.
    """
    max_words = max_words or current_answer_words()
    if not raw_answer or len(raw_answer.strip()) < 5:
        return raw_answer

    draft = raw_answer.strip().replace("\n", " ")[:LOCAL_LLM_DRAFT_CHARS]
    prompt = (
        f"Вопрос гостя: {question}\n"
        f"Черновик ответа: {draft}\n"
        f"Перескажи черновик своими словами, живо и коротко, "
        f"не более {max_words} слов. Сохрани имена, числа, даты и оговорки. "
        f"Не добавляй факты. Ответ:"
    )
    try:
        text = _ollama_chat(
            prompt,
            model=REPHRASE_MODEL,
            num_predict=tokens_for_words(max_words),
            temperature=0.5,
            timeout=LOCAL_LLM_TIMEOUT,
        )
        # Падаем обратно на черновик, если модель вернула пусто
        if not text:
            return raw_answer
        return finalize_answer(text, max_words, keep_emotion=True)
    except Exception as e:
        print(f"⚠️ Не удалось пересказать ответ через ИИ: {e}", file=sys.stderr)
        return raw_answer


def ask_ollama_direct(
    question: str,
    max_words: Optional[int] = None,
    *,
    history: Optional[list[dict[str, str]]] = None,
    hint: str = "",
) -> str:
    """
    Отвечает на вопрос напрямую через Ollama (без веб-поиска).
    Используется как fallback, когда поиск ничего не нашёл (0 контекстов) —
    например, для разговорных реплик («Здесь ты», «Привет», «Что ты умеешь»).

    Возвращает пустую строку при ошибке.
    """
    max_words = max_words or current_answer_words()
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Europe/Moscow"))
    except Exception:
        now = datetime.now()
    now_str = now.strftime("%d.%m.%Y %H:%M")

    context = f"\n{hint}" if hint else ""
    prompt = (
        f"Сегодня {now_str}.\n"
        f"Вопрос гостя: {question}{context}\n"
        f"Ответь не более чем {max_words} словами."
    )
    try:
        text = _ollama_chat(
            prompt,
            model=REPHRASE_MODEL,
            num_predict=tokens_for_words(max_words),
            temperature=0.6,
            timeout=LOCAL_LLM_TIMEOUT,
            history=history,
        )
        if not text:
            return ""
        return finalize_answer(text, max_words, keep_emotion=True)
    except Exception as e:
        print(f"⚠️ Прямой вопрос к LLM не удался: {e}", file=sys.stderr)
        return ""


def warmup_ollama(model: str = REPHRASE_MODEL, timeout: int = 240) -> None:
    """
    Прогревает LLM-модель в Ollama коротким вызовом, чтобы она загрузилась
    в VRAM до первого реального запроса (иначе первый ответ ждёт cold start).

    Таймаут по умолчанию 240 с: холодная загрузка 26B на двух картах занимает
    ~50–90 с (плюс возможная догрузка весов с диска). Раньше стояло 20 с —
    этого не хватало, и запрос ОБРЫВАЛ загрузку уже на середине
    («client connection closed before llama-server finished loading, aborting
    load»), после чего модель вообще оставалась не загруженной.
    """
    try:
        # Через /api/chat (см. _ollama_chat): у /api/generate ответ содержит
        # массив context, из-за которого VPN обрывал ответ.
        text = _ollama_chat(
            "Привет",  # короткий прогрев
            model=model,
            num_predict=5,
            temperature=0.6,
            timeout=timeout,
        )
        if text:
            print(f"🔥 LLM прогрев: {model} готова")
        else:
            print(f"⚠️ LLM прогрев: {model} ответила пусто", file=sys.stderr)
    except Exception as e:
        print(f"⚠️ LLM прогрев не удался: {e}", file=sys.stderr)


@dataclass(frozen=True)
class DialogueContext:
    history: list[dict[str, str]]
    summary: str = ""

    def search_history(self) -> list[dict[str, str]]:
        if not self.summary:
            return self.history
        return [
            {"role": "user", "content": "Краткая сводка более ранней части этого разговора."},
            {"role": "assistant", "content": self.summary},
            *self.history,
        ]


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
        runtime=None,
        controller: Optional[SensitivityController] = None,
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

        # --- рантайм-состояние (общее с GUI) -------------------------------
        # Если не передали явно — берём синглтон, чтобы CLI и GUI работали
        # с одними и теми же настройками и буферами визуализации.
        self.runtime = runtime if runtime is not None else get_runtime()
        cfg = self.runtime.snapshot()
        self._controller = controller or SensitivityController(
            base_threshold=vad_threshold,
            margin=cfg.get("vad", {}).get("margin", 2.5),
            target_rms=cfg.get("audio", {}).get("target_rms", 0.15),
            max_gain=cfg.get("audio", {}).get("max_gain", 8.0),
        )
        # Синхронизируем флаги CLI с рантайм-конфигом при старте: явно
        # переданные аргументы командной строки имеют приоритет.
        if no_confirm:
            self.runtime.set_feature("confirm_before_answer", False)
        if barge_in:
            self.runtime.set_feature("barge_in", True)

        # Флаг прерывания озвучки (barge-in): при речи во время ответа
        # озвучка будет прервана, чтобы ассистент сразу слушал новый вопрос.
        self._interrupt_tts = False

        # TTS-поток: при barge-in озвучка идёт в фоне, не блокируя обработку.
        self._tts_thread: Optional[threading.Thread] = None

        # Эмоция текущего ответа (темп/громкость синтеза). Ставится в _speak
        # по метке [эмоция:…] из ответа модели, сбрасывается после озвучки.
        self._tts_emotion: Optional[dict] = None

        # STT-модель: грузим лениво и переиспользуем
        self._asr = None          # выбранная модель распознавания речи

        self.is_recording = False
        self.noise_profile: Optional[np.ndarray] = None
        self._capture_proc: Optional[subprocess.Popen] = None
        self._vad_buffer = np.array([], dtype=np.float32)
        self._collection_id: Optional[str] = None  # кэш ID коллекции для МИРЭА

        # Чат WebRAgent: контекст вопросов/ответов хранится на сервере (MongoDB).
        # Создаём чат при старте, далее каждый запрос идёт через /chat/<id>/query,
        # а WebRAgent сам передаёт модель последние сообщения как контекст.
        self._chat_id: Optional[str] = None
        self._memory_chat_id = uuid.uuid4().hex

        # Локальная история (резерв) — держим последние 10 обменов
        self._conversation_history: list = []
        # Долговременная память: обмены переживают смену чата и перезапуск.
        self.memory = ConversationMemory()

        # Гейт проверяет конец фразы, выделяет предложенную тему и ведёт сводку.
        gate_cfg = cfg.get("gate", {}) or {}
        self.gate = PhraseGate(
            ollama_url=(gate_cfg.get("ollama_url")
                        or cfg.get("llm", {}).get("ollama_url")
                        or "http://localhost:11434"),
            model=gate_cfg.get("model") or cfg.get("llm", {}).get("model") or REPHRASE_MODEL,
            wait_sec=float(gate_cfg.get("wait_sec", 5.0)),
            timeout=float(gate_cfg.get("timeout", 10.0)),
            enabled=bool(gate_cfg.get("enabled", True)),
        )
        # Незаконченная (по смыслу) фраза: пока гость не договорит, держим её
        # аудио и текст, чтобы при следующей реплике склеить в одну.
        self._pending_raw: list = []
        self._pending_text = ""
        self._pending_since = 0.0
        # Когда флешируем незаконченную фразу по таймауту — не переспрашиваем гейт.
        self._force_phrase = False

        self._last_mirea = False
        self._last_effective = ""
        self._last_topic = ""
        self._pending_followup = None
        # Откуда пришёл последний ответ: "chat" (серверный чат с контекстом),
        # "query" (прямой запрос) или "direct" (быстрый ответ через LLM).
        self._last_query_source: str = ""

        # Флаг занятости: пока идёт обработка (STT/RAG/TTS),
        # речь с микрофона игнорируется и новые запросы не отправляются
        self._busy = False

        # Состояние VAD (только из потока захвата)
        self._current_phrase: list = []
        self._last_speech_time = 0.0
        self._speech_started = False
        self._phrase_started_at = 0.0

        # Очередь готовых фраз (16кГц аудио) для обработки в фоне
        self._phrase_queue: queue.Queue = queue.Queue()

        # Блокировка, чтобы не обрабатывать параллельно несколько фраз
        self._processing_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Актуальные пороги (учитывают ручной/авто режим и правки из GUI)
    # ------------------------------------------------------------------
    def _current_vad_threshold(self) -> float:
        """Порог VAD: авто (по шумовому полу) или ручной из GUI."""
        if self.runtime.feature("auto_vad"):
            return self._controller.auto_vad_threshold()
        return float(self.runtime.get("vad", "threshold", default=self._vad_threshold))

    def _current_silence_timeout(self) -> float:
        return float(self.runtime.get("vad", "silence_timeout", default=self.silence_timeout))

    def _current_min_speech(self) -> float:
        return float(self.runtime.get("vad", "min_speech", default=self.min_speech_duration))

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

            # Авто-чувствительность: обновляем шумовой пол/усиление по каждому чанку
            chunk_rms = rms(chunk)
            threshold = self._current_vad_threshold()
            is_speech = prob > threshold
            if self.runtime.feature("auto_vad"):
                self._controller.update(chunk_rms, is_speech=is_speech)

            current_time = time.time()

            # Телеметрия для GUI (уровни + пороги) — дёшево, только счётчики
            self.runtime.set_metrics(
                mic_rms=chunk_rms,
                vad_prob=prob,
                vad_threshold=threshold,
                gate_threshold=self._controller.gate_threshold(),
                noise_rms=self._controller.noise_rms or 0.0,
                speech_rms=self._controller.speech_rms or 0.0,
                gain=self._controller.gain,
            )

            if is_speech:
                # Если идёт озвучка ответа и включён barge-in — прерываем её
                if (self.runtime.feature("barge_in") and self._tts_thread
                        and self._tts_thread.is_alive()):
                    self._interrupt_tts = True
                if not self._speech_started:
                    self._speech_started = True
                    self._current_phrase = []
                    self._phrase_started_at = current_time
                self._current_phrase.append(chunk)
                self._last_speech_time = current_time

                # Защита от «залипания»: слишком длинная фраза принудительно закрывается
                if current_time - self._phrase_started_at > MAX_PHRASE_DURATION_SEC:
                    self._finalize_phrase()
                    self._speech_started = False
                    self._current_phrase = []
            else:
                if self._speech_started:
                    # Держим немного «хвоста» тишины — он нужен, чтобы
                    # не срезать последний слог (паддинг фразы).
                    self._current_phrase.append(chunk)
                    if current_time - self._last_speech_time >= self._current_silence_timeout():
                        self._finalize_phrase()
                        self._speech_started = False
                        self._current_phrase = []

            # Живой осциллограф: сырой сигнал с микрофона
            self.runtime.push_wave(raw=chunk)
            # Шкала уровня: микрофон → после гейта (что выжило после шумодава).
            # Именно эти две линии рисует GUI на нижнем графике вместе с порогом.
            gate = self._controller.gate_threshold()
            gated_rms = chunk_rms if (not self.runtime.feature("noise_reduction")
                                      or chunk_rms >= gate) else 0.0
            self.runtime.push_level(chunk_rms, gated_rms, gated_rms)

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
            self.runtime.log_event("error", f"arecord не запустился: {e}")
            self.is_recording = False
            return

        chunk_bytes = 512 * 2  # 512 сэмплов * 2 байта (S16_LE)
        received_any = False
        silent_reads = 0
        while self.is_recording:
            data = self._capture_proc.stdout.read(chunk_bytes)
            if not data:
                break
            received_any = True
            try:
                self._feed_pcm(data)
            except Exception as e:
                print(f"❌ Ошибка VAD: {e}", file=sys.stderr)
                break

            # Признак «микрофон молчит»: подряд идут нулевые сэмплы.
            # Это не ошибка — микрофон может быть просто не подключён.
            if not np.any(np.frombuffer(data, dtype=np.int16)):
                silent_reads += 1
            else:
                silent_reads = 0
            if silent_reads == 64 and not getattr(self, "_warned_silent", False):
                self._warned_silent = True
                msg = ("Микрофон молчит (все сэмплы нулевые). Проверьте, что он "
                       "подключён и выбран в «Устройства и сеть».")
                print(f"🔇 {msg}", file=sys.stderr)
                self.runtime.log_event("error", msg)

        # Добиваем остаток при остановке
        if self._speech_started:
            self._finalize_phrase()
        if self._capture_proc:
            try:
                self._capture_proc.terminate()
            except Exception:
                pass

        # Если не пришло ни одного байта — микрофон недоступен
        if self.is_recording and not received_any:
            msg = (f"Микрофон не отдал данных (устройство "
                   f"«{arecord_command()[2]}»). Подключите микрофон и выберите его "
                   f"в «Устройства и сеть», затем перезапустите ассистента.")
            print(f"❌ {msg}", file=sys.stderr)
            self.runtime.log_event("error", msg)
            self.runtime.set_status("Микрофон недоступен")

    def _finalize_phrase(self):
        if not self._current_phrase:
            return
        audio_np = np.concatenate(self._current_phrase)
        duration = len(audio_np) / SAMPLE_RATE
        self._current_phrase = []

        # Обрезаем лишнюю тишину по краям (оставляя небольшой паддинг)
        pad = int(PHRASE_PADDING_SEC * SAMPLE_RATE)
        gate = self._controller.gate_threshold()
        if audio_np.size:
            active = np.nonzero(np.abs(audio_np) > gate)[0]
            if active.size:
                start = max(0, int(active[0]) - pad)
                end = min(len(audio_np), int(active[-1]) + pad)
                audio_np = audio_np[start:end]
                duration = len(audio_np) / SAMPLE_RATE

        if duration < self._current_min_speech():
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
                # Флеш незаконченной фразы: гость так и не продолжил в течение
                # gate.wait_sec — обрабатываем то, что накопилось, по принуждению
                # (гейт уже не переспрашиваем).
                if (self._pending_raw and not self._busy
                        and self.gate.enabled
                        and time.time() - self._pending_since >= self.gate.wait_sec):
                    pending = self._pending_raw
                    self._pending_raw = []
                    self._pending_text = ""
                    self._force_phrase = True
                    print("⏱️ Фраза не продолжилась — обрабатываю как есть.")
                    self._handle_phrase(np.concatenate(pending))
                    self._force_phrase = False
                continue
            if self._busy:
                continue  # не обрабатываем новые фразы, пока заняты
            self._handle_phrase(audio_np)

    def _get_asr(self):
        """
        Возвращает модель распознавания речи (Kairos).
        Грузится лениво при первом использовании и переиспользуется.
        """
        if self._asr is not None:
            return self._asr

        print("[STT] Инициализация Kairos-модели...")
        from kairos_asr import KairosASR

        self._asr = KairosASR(device="auto")
        return self._asr

    def _warmup_all(self):
        """
        Полный прогрев всех ИИ-компонентов перед началом работы:
        STT (Kairos) + LLM (Ollama) + TTS (Silero). Вызывается всегда при
        старте, чтобы первый вопрос/ответ не ждал загрузку моделей.

        Прогрев TTS особенно важен для ПОТОКОВОЙ озвучки: первый синтез без
        прогрева занимает ~1.8с (инициализация CUDA), а после прогрева —
        ~0.5с. Именно от него зависит, как быстро зазвучат первые слова.
        """
        print("🔥 Полный прогрев ИИ-моделей...")
        # 1. Прогрев STT
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

        # 3. Прогрев TTS — синтезируем короткую фразу, звук НЕ проигрываем.
        try:
            speaker = self.runtime.get(
                "tts", "speaker", default=self._tts_speaker
            )
            if self._synthesize_pcm_bytes("Прогрев.", speaker):
                print("   ✅ TTS готова")
            else:
                # Модель может быть ещё не нужна — просто отмечаем.
                print("   ⚠️ Прогрев TTS не дал данных (пропускаю).")
        except Exception as e:
            print(f"   ⚠️ Прогрев TTS не удался: {e}", file=sys.stderr)

        print("🔥 Прогрев завершён.")

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

    @staticmethod
    def is_conversational(text: str) -> bool:
        """
        Определяет, является ли фраза разговорной репликой (не требующей
        веб-поиска): «Где ты это вычитал?», «Здесь ты», «Привет», «Ага» и т.п.

        Такие фразы не должны уходить в агентный/веб-поиск — их обрабатывает
        LLM напрямую (ask_ollama_direct), чтобы не зависать и отвечать быстро.
        """
        import re
        low = text.lower().strip()

        # Короткие междометия/обращения (не информационные вопросы)
        # «как работает X», «что такое X» — НЕ считаем разговорными, их надо искать.
        if re.fullmatch(r"(ага|угу|да|нет|ну|так|ок|окей|ладно|понятно|спасибо|привет|здравствуй|пока|давай|приветствую)", low):
            return True
        if re.fullmatch(r"(привет|здравствуй|здравствуйте|пока|давай|спасибо|ок|окей|ладно|ага|угу)\b[\s\S]*", low) and len(low.split()) <= 3:
            return True

        # Фразы про «где ты взял/вычитал/нашёл» — разговорные упрёки/вопросы к ассистенту
        conversation_patterns = [
            r"где ты\s+(это вообще|вообще|это|так|же)?\s*(взял|вычитал|нашёл|нашел|узнал|услышал|прочитал)",
            r"ты (чего|зачем|почему|откуда|как посмел)",
            r"как ты (узнал|понял|догадался|смог)",
            r"это (ты|валера|робот)",
            r"ты здесь|здесь ты|ты тут|ты рядом",
            r"хватит|отстань|замолчи|перестань",
            r"продолжай|расскажи ещё|а дальше|что дальше|и что",
            # Жалобы и реплики В АДРЕС САМОГО РОБОТА («Загрузись уже
            # нормально», «Ты тормозишь», «Алло, ты тут?»). Это не вопрос про
            # вуз и не тема для веб-поиска: поиск ничего не находил, и робот
            # отвечал «К сожалению, я не смог найти ответ на ваш вопрос».
            # Отвечаем напрямую, по-человечески.
            r"загруз\w*|зависа\w*|тормоз\w*|тупишь|глючишь|барахлишь",
            r"(ты|робот)\s+(?:не\s+)?(?:работаешь|слышишь|понимаешь|отвечаешь|спишь)",
            r"\bалло\b|\bау\b|\bэй\b|слышишь меня|долго\s+(?:думаешь|молчишь)",
            r"нормально\s+(?:работай|отвечай|говори)",
        ]
        for pat in conversation_patterns:
            if re.search(pat, low):
                return True

        return False

    # ------------------------------------------------------------------
    # Запрос к WebRAgent: серверный чат (с контекстом) → фолбэк на /query
    # ------------------------------------------------------------------
    def _dialogue_context(self) -> DialogueContext:
        summary = self.gate.digest[:600] if self.gate and self.gate.enabled else ""
        return DialogueContext(list(self._conversation_history[-DIALOGUE_HISTORY_MESSAGES:]), summary)

    def _resolve_offer_confirmation(self, text: str) -> Optional[FollowUp]:
        if not is_offer_confirmation(text) or not self.gate or not self.gate.enabled or self._pending_followup:
            return None
        rows = self.memory.recent(self._memory_chat_id, limit=1)
        if not rows or time.time() - rows[0]["ts"] > FOLLOWUP_WINDOW_SEC:
            return None
        topic = self.gate.detect_offered_topic(rows[0]["a"])
        if not topic or is_forbidden_topic(topic):
            return None
        return self.memory.resolve_followup(
            text, self._memory_chat_id, has_own_topic=False, offered_topic=topic,
        )

    def _direct_llm(self, question: str, hint: str = "", *,
                    context: Optional[DialogueContext] = None, search_query: str = "") -> str:
        """
        Быстрый ответ БЕЗ поиска по базе — через активную модель WebRAgent.

        ПОЧЕМУ ТАК, А НЕ СРАЗУ ЧЕРЕЗ OLLAMA.
        Раньше этот путь всегда шёл в локальную Ollama, и это была главная
        причина «зависаний»: на промпт ~2500 токенов ответ Ollama либо не
        доходил из-за обрыва на loopback, либо раннер модели падал, и клиент
        висел до таймаута (ровно 45–60 с, после чего в логе `500 | 1m0s`).
        Гость всё это время слышал тишину, а пульт показывал «Отвечаю LLM…».

        WebRAgent (DeepSeek) отвечает за 1.5–2.5 с и не зависит ни от GPU,
        ни от загрузки модели. Поэтому сначала пробуем его, и только если он
        недоступен — отвечаем локальной моделью.

        ``history`` (последние реплики) передаём ОБЯЗАТЕЛЬНО: без него модель
        не понимает уточнений и отвечает «не разобрал».

        Возвращает пустую строку, если ответить не удалось ни там, ни там.
        """
        if _INTRO_REQUEST_RE.search(question):
            return ("Я Валера, гуманоидный робот-экскурсовод. [эмоция:дружелюбие] "
                    "Рассказываю о лаборатории и помогаю с вопросами о робототехнике. [эмоция:дружелюбие]")
        if asks_robot_firmware_version(question):
            return "Точная версия моей прошивки мне неизвестна. [эмоция:спокойствие]"

        # Недавние реплики помогают понять, к чему относится уточнение.
        context = context if context is not None else self._dialogue_context()
        history = context.history
        previous_question = next((msg["content"] for msg in reversed(history) if msg["role"] == "user"), "")
        version_answer = prompt_version_answer(question, previous_question)
        if version_answer:
            return version_answer
        notes = []
        if context.summary:
            notes.append(f"Краткая сводка более раннего разговора: {context.summary}")
        if search_query and search_query != question:
            notes.append(f"Уточнённый смысл текущего вопроса: {search_query}")
        style_question = question
        if search_query and is_offer_confirmation(question):
            style_question = search_query
            notes.append("Продолжи рассказ по этой теме. Начни с конкретного факта.")
        notes.extend(filter(None, (style_hint(style_question), hint)))
        hint = "\n".join(notes)
        if self.rag is not None:
            answer = self.rag.direct_answer(question, history=history, hint=hint)
            if answer:
                return answer
            print("⚠️ WebRAgent /direct не ответил — пробую локальную модель.",
                  file=sys.stderr)
        return ask_ollama_direct(question, history=history, hint=hint)

    def _finish_continuation(self, answer: str, question: str, context: DialogueContext,
                             search_query: str) -> str:
        """Даёт модели одну попытку дополнить повторный или оборванный ответ."""
        previous = next(
            (message["content"] for message in reversed(context.history) if message["role"] == "assistant"),
            "",
        )
        answer = finalize_answer(answer, keep_emotion=True)
        if not is_stalled_continuation(answer, previous):
            return answer
        hint = (
            "Гость просит продолжить рассказ. Сразу дай новый конкретный факт по теме. "
            "Закончи мысль. Если подходящих сведений нет, скажи об этом и ничего не выдумывай."
        )
        retry = self._direct_llm(question, hint=hint, context=context, search_query=search_query)
        retry = finalize_answer(retry, keep_emotion=True)
        if is_stalled_continuation(retry, previous):
            return "Что именно Вы хотите узнать об этом?"
        return retry

    def _answer_said_unrecognized(
        self, answer: str, question: str, mirea_related: bool, *,
        context: Optional[DialogueContext] = None, search_query: str = "",
    ) -> str:
        """
        Проверяет ответ на «сбежавшее» «не разобрал» и исправляет его.

        Модель любит отвечать «Прости, я не разобрал — повтори, пожалуйста?»
        на любую сбивчивую речь. Для живого робота это выглядит как поломка:
        гость сказал нормальные слова, а его просят повторить.

        Логика:
        * в реплике НЕТ осмысленных слов → честно просим повторить;
        * слова ЕСТЬ → просим модель ответить ещё раз (с контекстом и явной
          подсказкой), а если она снова отказывается — задаём короткий
          уточняющий вопрос по теме.
        """
        if not is_unrecognized_reply(answer):
            return answer

        if is_gibberish(question):
            return UNRECOGNIZED_REPLY

        print("🔁 Модель сказала «не разобрал» — прошу ответить иначе.")
        hint = (
            "Ответь по существу, опираясь на предыдущие реплики. "
            "Если неясен предмет вопроса, попроси уточнить его."
        )
        retry = self._direct_llm(question, hint=hint, context=context, search_query=search_query)
        if retry and not is_unrecognized_reply(retry):
            return retry

        # Модель упёрлась — отвечаем сами, без канцелярита.
        return clarify_reply(mirea_related)

    def _stream_enabled(self) -> bool:
        """
        Потоковая озвучка: тумблер включён, озвучка включена, есть серверный чат.

        Пересказ (rephrase_answers) при потоке не нужен: ответ по базе уже формулируется
        промптом Валеры, а пересказ вторым вызовом модели только добавлял задержку.
        """
        if not self.runtime.feature("stream_tts", default=True) or not self.runtime.feature("tts", default=True):
            return False
        return bool(self._chat_id) and hasattr(self.rag, "chat_query_stream")

    def _stream_answer(self, rag_text: str, collection_id: str, *,
                       context: Optional[DialogueContext] = None, original_query: str = "") -> Optional[str]:
        """
        Ответ по базе знаний потоком: первое предложение озвучивается, пока модель пишет остальное.

        Возвращает озвученный текст, либо None — если поток не удался до первого слова
        (тогда вызывающий код делает обычный запрос без потока).
        """
        import queue as _queue

        speaker = self.runtime.get("tts", "speaker", default=self._tts_speaker)
        budget = current_answer_words()
        sent_q: "_queue.Queue" = _queue.Queue()
        pcm_q: "_queue.Queue" = _queue.Queue(maxsize=3)
        stop = threading.Event()
        spoken: list = []
        state = {"words": 0, "limit": False}
        orig_prev_named = _ans_ctx["prev_named"]

        def _producer():
            while True:
                s = sent_q.get()
                if s is None:
                    pcm_q.put(None)
                    return
                if stop.is_set():
                    continue
                # В очередь кладём пару (текст, стиль эмоции): так стиль
                # привязан к своей фразе и не «перетекает» на соседние.
                if isinstance(s, tuple):
                    s_text, s_emo = s
                else:
                    s_text, s_emo = s, None
                pcm_q.put(self._synthesize_pcm_bytes(s_text, speaker, s_emo))

        def _player():
            while True:
                pcm = pcm_q.get()
                if pcm is None:
                    return
                if self._interrupt_tts:
                    stop.set()
                    continue
                if pcm and not stop.is_set():
                    play_audio(pcm, sample_rate=TTS_SAMPLE_RATE)

        prod = threading.Thread(target=_producer, daemon=True)
        play = threading.Thread(target=_player, daemon=True)
        prod.start()
        play.start()

        def _emit(sentence: str) -> None:
            if state["limit"] or not sentence.strip():
                return
            # Эмоция стоит в конце КАЖДОГО предложения, поэтому кусок приходит
            # от split_complete_sentences уже вместе со своей меткой — сразу
            # и синтезируем _(текст, стиль)_. Если метки нет, emotion_style
            # вернёт «дружелюбие», то есть живую интонацию, а не плоскую.
            sentence, _emo_name = parse_emotion(sentence)
            _emo_style = emotion_style(_emo_name)
            if _emo_name:
                print(f"🎭 Эмоция: {_emo_name} (темп ×{_emo_style['speed']:.2f})")
            clean = finalize_answer(sentence, budget)
            if not clean or (clean == "Слушаю Вас." and sentence.strip() != clean):
                return
            if clean.endswith("?") and _ans_ctx["prev_question"] and spoken:
                return                                    # не заканчиваем каждый ответ вопросом
            n = len(clean.split())
            if spoken and state["words"] + n > budget:
                state["limit"] = True                     # потолок длины по вопросу
                return
            state["words"] += n
            spoken.append(clean)
            if _VUZ_NAME_RE.search(clean):
                _ans_ctx["prev_named"] = True             # дальше в этом ответе вуз не называем
            sent_q.put((clean, _emo_style))

        buf, failed = "", None
        try:
            self._last_query_source = "chat-stream"
            context = context if context is not None else self._dialogue_context()
            for ev in self.rag.chat_query_stream(
                self._chat_id, rag_text, collection_id, max_results=3,
                conversation_context=context.search_history(), original_query=original_query,
            ):
                t = ev.get("t")
                if t == "err":
                    raise RuntimeError(ev.get("message") or "ошибка потока")
                if t == "d":
                    buf += ev.get("x", "")
                    done, buf = split_complete_sentences(buf)
                    for s in done:
                        _emit(s)
                    if state["limit"]:
                        break
            if buf.strip() and not state["limit"]:
                _emit(buf.strip())
        except Exception as e:  # noqa: BLE001
            failed = e
            print(f"⚠️ Потоковый ответ прервался: {e}", file=sys.stderr)
        finally:
            sent_q.put(None)
            prod.join(timeout=60)
            play.join(timeout=120)
            _ans_ctx["prev_named"] = orig_prev_named
        if not spoken:
            return None if failed else ""
        return " ".join(spoken)

    def _remember_exchange(self, question: str, answer: str, persist: bool = True, *,
                           use_as_topic: bool = True) -> None:
        """
        Дописывает обмен в ЛОКАЛЬНУЮ историю (последние 10 обменов).

        Нужна для двух случаев:
        * быстрые ответы через ``ask_ollama_direct`` идут мимо WebRAgent,
          поэтому их надо запомнить самим;
        * запасной путь ``/query`` (когда серверный чат недоступен) берёт
          контекст именно отсюда.

        Основной путь (серверный чат) историю хранит сам, поэтому дублировать
        её не нужно — она используется только как подстраховка.
        """
        answer, _ = parse_emotion(answer)
        self._conversation_history.append({"role": "user", "content": question})
        self._conversation_history.append({"role": "assistant", "content": answer})
        self._conversation_history = self._conversation_history[-20:]  # 10 обменов
        note_answer_given(answer)
        if persist:
            # В долговременную память — с id чата и темой (вуз/не вуз).
            self.memory.add(
                question, answer, chat_id=self._memory_chat_id,
                mirea=self._last_mirea if use_as_topic else False,
                eq=(self._last_effective or None) if use_as_topic else None,
                topic=(self._last_topic or None) if use_as_topic else None,
                use_as_topic=use_as_topic,
            )
        # Смысловой учёт всего сказанного: быстрая модель сжимает весь диалог
        # в короткую сводку. Обновляем в фоне, чтобы не задерживать озвучку.
        if self.gate and use_as_topic and question and answer:
            self.gate.enqueue_digest(question, answer)

    def _rag_query(
        self,
        query_text: str,
        collection_id: Optional[str] = None,
        use_web_search: bool = False,
        use_agent_search: bool = False,
        max_results: int = 4,
        *,
        context: Optional[DialogueContext] = None,
        original_query: str = "",
    ) -> dict:
        """
        Отправляет вопрос в WebRAgent.

        Основной путь — ``/chat/<chat_id>/query``, где WebRAgent сохраняет диалог.

        Если чат пропал (например, перезапустили MongoDB) — создаём новый и
        повторяем запрос. Запасной ``/query`` получает тот же контекст.
        """
        context = context if context is not None else self._dialogue_context()
        search_history = context.search_history()

        def _params(cid: Optional[str]) -> dict:
            return dict(
                query_text=query_text,
                collection_id=cid,
                use_web_search=use_web_search,
                use_agent_search=use_agent_search,
                max_results=max_results,
                conversation_context=search_history,
                original_query=original_query,
            )

        if self._chat_id:
            try:
                self._last_query_source = "chat"
                return self.rag.chat_query(chat_id=self._chat_id, **_params(collection_id))
            except Exception as e:
                # Чат мог исчезнуть — пересоздаём и пробуем ещё раз
                print(f"♻️  Чат недоступен ({e.__class__.__name__}) — создаю новый...")
                self._chat_id = None

        new_chat = self.rag.create_chat()
        if new_chat:
            self._chat_id = new_chat
            self._save_chat_id(new_chat)
            self.runtime.set_current_chat(new_chat, "")
            print(f"✅ Новый чат с контекстом: {new_chat}")
            try:
                self._last_query_source = "chat"
                return self.rag.chat_query(chat_id=new_chat, **_params(collection_id))
            except Exception as e:
                print(f"⚠️  Запрос к чату не удался: {e}", file=sys.stderr)

        print("⚠️  Серверный чат недоступен — передаю локальный контекст в /query.")
        self._last_query_source = "query"
        try:
            return self.rag.query(
                query_text=query_text,
                collection_id=collection_id,
                use_web_search=use_web_search,
                use_agent_search=use_agent_search,
                max_results=max_results,
                conversation_context=search_history,
                original_query=original_query,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"⚠️ Поиск недоступен ({type(exc).__name__}) — пробую прямой ответ.")
            return {"response": "", "contexts": []}

    def _handle_phrase(self, audio_np: np.ndarray):
        """Полный пайплайн: Шумодав → STT → триггеры → RAG → TTS"""
        self._busy = True
        try:
            # Авто-новый чат после длительной тишины: если гость давно ушёл,
            # начинаем разговор с чистого контекста (см. rag.auto_new_chat...).
            self._maybe_auto_new_chat()
            self._handle_phrase_inner(audio_np)
            self.runtime.touch_activity()
        finally:
            self._busy = False

    def _maybe_auto_new_chat(self) -> None:
        """
        Начинает новый чат, если с прошлой реплики прошло слишком много тишины.

        Порог задаётся в ``valera_config.json`` →
        ``rag.auto_new_chat_after_silence_sec`` (0 = выключено), а также
        переключателем «Авто-новый чат» в GUI/веб-интерфейсе.
        """
        if not self.runtime.feature("auto_new_chat", default=True):
            return
        if not self.runtime.should_auto_new_chat():
            return

        idle = self.runtime.seconds_since_activity()
        limit = self.runtime.get(
            "rag", "auto_new_chat_after_silence_sec", default=0.0
        )
        print(f"🕐 Тишина {idle:.0f}с (порог {limit:.0f}с) — начинаю новый чат.")
        self.runtime.log_event(
            "info", f"Авто-новый чат после {idle:.0f}с тишины"
        )
        self.new_conversation()

    def new_conversation(self) -> Optional[str]:
        """Начинает отдельный разговор даже при недоступном WebRAgent."""
        with self._processing_lock:
            self._chat_id = self.rag.create_chat()
            self._memory_chat_id = self._chat_id or uuid.uuid4().hex
            self._conversation_history.clear()
            self._pending_raw = []
            self._pending_text = ""
            self._pending_followup = None
            self._last_mirea = False
            self._last_effective = self._last_topic = self._last_query_source = ""
            _ans_ctx.update(dict.fromkeys(_ans_ctx, False))
            if self.gate:
                self.gate.reset()
            self._save_chat_id(self._chat_id)
            self.runtime.set_current_chat(self._chat_id, "")
            self.runtime.mark_new_chat_started()
            return self._chat_id

    def _speak(self, text: str, *, literal: bool = False) -> None:
        """
        Озвучивает текст. При включённом barge-in — в отдельном потоке,
        чтобы можно было прервать озвучку новым вопросом.

        Текст озвучивается ПОТОКОВО, по одному предложению: первое
        предложение начинает звучать, пока в фоне синтезируется второе.
        Это заметно ускоряет первые слова (не надо ждать синтеза всего
        ответа). Плюс в начало каждого куска добавляется короткая тишина,
        чтобы колонки успели открыться и начало не «проглатывалось».

        Тумблеры «Озвучивать ответы» и «Прерывать озвучку» читаются из
        рантайм-состояния на каждом вызове, поэтому применяются без перезапуска.
        """
        if not text:
            return
        # Ответ озвучивается ПОТОКОВО, по предложениям, а метка эмоции стоит в
        # конце КАЖДОГО предложения. Поэтому метки разбираем здесь же, вместе с
        # разбивкой: у каждой фразы будет своя интонация, а не одна на всё.
        pairs = [(text, emotion_style(None))] if literal else split_answer_for_tts(text)
        if not pairs:
            return
        # Интонация для отката на синтез всего текста целиком: берём стиль
        # первой фразы. Так даже односоставный ответ («Кто ректор?» — одно
        # предложение) не потеряет свою эмоцию на запасном пути.
        self._tts_emotion = pairs[0][1]
        if not self.runtime.feature("tts", default=True):
            print("🔇 Озвучка выключена (тумблер в GUI).")
            return

        speaker = self.runtime.get("tts", "speaker", default=self._tts_speaker)
        plain = " ".join(t for t, _ in pairs)

        def _do_speak():
            try:
                # 1) Пробуем потоковую озвучку по предложениям на выбранном
                #    устройстве (aplay). Первые слова звучат почти сразу.
                if TTS_PARALLEL_SENTENCES:
                    if len(pairs) > 1 and self._stream_sentences(pairs, speaker):
                        return
                # 2) Откат: целиком через устройство (aplay)…
                played = self._speak_via_device(plain, speaker)
                if not played:
                    # 3) …или встроенный проигрыватель NeuralSpeaker (simpleaudio).
                    self._speak_whole_fallback(plain, speaker)
            except Exception as e:
                print(f"❌ Ошибка синтеза речи: {e}", file=sys.stderr)
                self.runtime.log_event("error", f"TTS: {e}")
            finally:
                self._interrupt_tts = False
                # Эмоцию сбрасываем: следующий ответ может быть без метки.
                self._tts_emotion = None

        if self.runtime.feature("barge_in"):
            # Ждём завершения предыдущей озвучки (быстрое прерывание флагом)
            if self._tts_thread and self._tts_thread.is_alive():
                self._interrupt_tts = True
                self._tts_thread.join(timeout=2.0)
            self._interrupt_tts = False
            self._tts_thread = threading.Thread(target=_do_speak, daemon=True)
            self._tts_thread.start()
        else:
            _do_speak()

    def _speak_whole_fallback(self, text: str, speaker: str) -> None:
        """
        Озвучивает текст целиком встроенным проигрывателем NeuralSpeaker.

        Добавляем в начало короткую тишину в WAV, чтобы начало фразы не
        срезалось при открытии устройства (та же причина, что и в aplay-пути).
        """
        try:
            data = self._synthesize_pcm_bytes(text, speaker)
            if data:
                # Проигрываем WAV напрямую через simpleaudio — с тишиной в начале.
                import wave
                import io
                import simpleaudio as _sa

                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(TTS_SAMPLE_RATE)
                    wf.writeframes(data)
                buf.seek(0)
                with wave.open(buf, "rb") as wf:
                    play_obj = _sa.WaveObject(
                        wf.readframes(wf.getnframes()),
                        wf.getnchannels(), wf.getsampwidth(), wf.getframerate(),
                    ).play()
                    play_obj.wait_done()
                return
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ simpleaudio-откат не удался: {e}", file=sys.stderr)
        # Последний рубеж — штатный NeuralSpeaker (играет в default).
        self.tts.speak(
            text, speaker=speaker, save_file=False, sample_rate=TTS_SAMPLE_RATE
        )

    def _stream_sentences(self, sentences: list, speaker: str) -> bool:
        """
        Озвучивает предложения конвейером: пока играет текущее, в фоне уже
        синтезируется следующее. Возвращает True, если всё проигралось через
        aplay; False — если вывод не удался (нужен откат на другой способ).

        ``sentences`` — пары ``(текст, стиль_эмоции)``: у каждой фразы своя
        интонация, потому что метка эмоции стоит в конце каждого предложения.
        Для обратной совместимости принимает и просто строки.
        """
        import queue as _queue

        q: "_queue.Queue" = _queue.Queue(maxsize=3)
        interrupted = threading.Event()

        def _producer():
            for item in sentences:
                if interrupted.is_set() or self._interrupt_tts:
                    break
                s, style = item if isinstance(item, tuple) else (item, None)
                if style:
                    print(f"🎭 Эмоция: темп ×{style['speed']:.2f}, тон ×{style['pitch']:.2f}")
                # Тишина в начало уже добавляется внутри _synthesize_pcm_bytes.
                pcm = self._synthesize_pcm_bytes(s, speaker, style)
                q.put((s, pcm))
            q.put(None)  # признак конца

        th = threading.Thread(target=_producer, daemon=True)
        th.start()

        played_any = False
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                _s, pcm = item
                if self._interrupt_tts:
                    interrupted.set()
                    break
                if not pcm:
                    continue
                if not play_audio(pcm, sample_rate=TTS_SAMPLE_RATE):
                    # Устройство не приняло звук — прерываем и просим откат.
                    interrupted.set()
                    return False
                played_any = True
        finally:
            interrupted.set()
        return played_any

    def _speak_via_device(self, text: str, speaker: str) -> bool:
        """
        Озвучивает текст на выбранном устройстве вывода.

        ``NeuralSpeaker`` всегда играет в «default», поэтому для конкретного
        устройства мы синтезируем PCM сами и отправляем его в ``aplay -D <dev>``.

        Возвращает False, если вывод на конкретное устройство не удался —
        тогда ``_speak()`` откатится на встроенный проигрыватель.
        """
        device = resolve_output_device()
        # 'default' → можно не усложнять, играем как раньше (simpleaudio)
        #if device == "default": return False
        try:
            data = self._synthesize_pcm_bytes(text, speaker)
            if not data:
                return False
            return play_audio(data, sample_rate=TTS_SAMPLE_RATE)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ Озвучка на устройство {device} не удалась: {e}", file=sys.stderr)
            return False

    def _apply_emotion_gain(self, pcm: bytes, volume: float) -> bytes:
        """
        Множит громкость PCM-потока (эмоциональный акцент).

        XTTS умеет менять только темп, поэтому «воодушевление»/«восторг»
        дополнительно звучат громче. Мягкое ограничение (tanh) не даёт
        клиппинга на громких слогах — иначе на колонках слышны щелчки.
        """
        if abs(volume - 1.0) < 0.01:
            return pcm
        try:
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            audio = np.tanh(audio * volume) * 0.98
            return (audio * 32767).astype(np.int16).tobytes()
        except Exception:  # noqa: BLE001
            return pcm

    def _apply_emotion_pitch(self, pcm: bytes, pitch: float) -> bytes:
        """
        Сдвигает ТОН голоса ресемплингом PCM (эмоциональная интонация).

        Зачем отдельно от ``speed``: XTTS меняет темп, но сама генерация
        недетерминирована, и на слух разница между «спокойно» и «радостно»
        смазывается (замер: разброс длительности ±20 % при одном и том же
        ``speed``). Ресемплинг даёт ГАРАНТИРОВАННЫЙ сдвиг: тот же поток
        проигрывается с другой частотой — радость выше и быстрее, серьёзность
        ниже и медленнее. Это и есть «смена тона», минимально достаточная
        для эмоциональной речи.

        Коэффициент держим в диапазоне 0.95–1.08: сильнее — голос начинает
        звучать неестественно («эффект бурундука»).
        """
        if abs(pitch - 1.0) < 0.005:
            return pcm
        pitch = max(0.94, min(1.09, pitch))
        try:
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
            n_out = max(1, int(round(len(audio) / pitch)))
            # Линейная интерполяция по новой сетке индексов.
            idx = np.linspace(0, len(audio) - 1, n_out)
            stretched = np.interp(idx, np.arange(len(audio)), audio)
            return stretched.astype(np.int16).tobytes()
        except Exception:  # noqa: BLE001
            return pcm

    def _synthesize_pcm_bytes(
        self, text: str, speaker: str, emotion: Optional[dict] = None
    ) -> Optional[bytes]:
        """
        Синтезирует речь в сырые PCM-байты (S16_LE, моно, 48 кГц).

        Обходим ``NeuralSpeaker.speak()`` (он сразу играет звук) и обращаемся к
        внутренней модели напрямую — так мы получаем данные и сами решаем,
        на какое устройство их отправить. Повторяем ту же предподготовку
        текста (транслит + числа прописью), что и оригинальный ``speak()``.

        ``emotion`` — стиль {speed, volume, pitch}. Если не задан, берётся
        ``self._tts_emotion`` (обычный путь через ``_speak``). Стриминг передаёт
        его явно: там предложения идут очередью и общий атрибут состояния
        гоняется между потоками.
        """
        # Основной голос — XTTS-v2 (отдельный сервис на GPU). Если он не поднят
        # или упал, клиент вернёт None и ниже сработает запасной Silero.
        if self.runtime.get("tts", "backend", default="xtts") == "xtts":
            # Базовая скорость из настроек умножается на темп эмоции:
            # радость — быстрее, серьёзный разговор — медленнее. Это и есть
            # «смена тона», доступная без отдельной эмоциональной модели.
            base_speed = float(self.runtime.get("tts", "speed", default=1.0) or 1.0)
            emo = emotion if emotion is not None else (getattr(self, "_tts_emotion", None) or {})
            speed = base_speed * float(emo.get("speed", 1.0))
            pcm = get_xtts_client().synthesize(
                text,
                speaker=self.runtime.get("tts", "xtts_speaker", default="Damien Black"),
                sample_rate=TTS_SAMPLE_RATE,
                speed=round(speed, 3),
            )
            if pcm and len(pcm) % 2 == 0 and np.any(np.frombuffer(pcm, dtype=np.int16)):
                pcm = self._apply_emotion_gain(pcm, float(emo.get("volume", 1.0)))
                pcm = self._apply_emotion_pitch(pcm, float(emo.get("pitch", 1.0)))
                return leading_silence_bytes(TTS_SAMPLE_RATE) + pcm

        try:
            import re as _re

            from num2words import num2words
            from transliterate import translit

            model = getattr(self.tts, "_NeuralSpeaker__model", None)
            if model is None:
                return None

            def _num2words_ru(match):
                return num2words(match.group().replace(",", "."), lang="ru")

            prepared = translit(text, "ru")
            prepared = _re.sub(r"-?[0-9][0-9,._]*", _num2words_ru, prepared)
            if not any(char.isalnum() for char in prepared):
                return None

            audio = model.apply_tts(
                text=prepared, speaker=speaker, sample_rate=TTS_SAMPLE_RATE
            )
            audio = audio.numpy()
            peak = float(np.max(np.abs(audio))) or 1.0
            audio = (audio * 32767 / peak).astype(np.int16)
            # Ставим короткую тишину в начало — иначе первые ~200 мс речи
            # «проглатываются», пока колонки открываются.
            pcm = leading_silence_bytes(TTS_SAMPLE_RATE) + audio.tobytes()
            return pcm
        except Exception as e:  # noqa: BLE001
            print(f"⚠️ Не удалось синтезировать PCM: {e}", file=sys.stderr)
            return None

    def _handle_phrase_inner(self, audio_np: np.ndarray):
        with self._processing_lock:
            if self._pending_raw:
                audio_np = np.concatenate(self._pending_raw + [audio_np])
                self._pending_raw = []
                self._pending_text = ""
            raw_audio = audio_np.copy()

            # 1–2. Очистка аудио: АРУ → энергетический гейт → спектральный шумодав
            # (порядок и вкл/выкл каждого шага — в clean_audio, управляются GUI)
            audio_np = clean_audio(
                audio_np, self.noise_profile, self._controller, self.runtime
            )

            # Публикуем «склейку» фразы для GUI: что пришло с микрофона,
            # что вырезал шумодав и что ушло на распознавание.
            self.runtime.push_phrase(raw_audio, audio_np)
            self.runtime.set_status("Распознаю речь…")

            # Сохраняем во временный WAV
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp:
                    tmp_path = tmp.name
                sf.write(tmp_path, audio_np, SAMPLE_RATE, subtype="PCM_16")

                # 3. Распознаём речь (Kairos): лёгкая модель, работает на CPU/GPU
                asr = self._get_asr()
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
                self.runtime.set_status("Тишина")
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
                self.runtime.set_status("Текст пуст")
                return

            print(f"🗣️  Распознано: {clean_text}")
            self.runtime.set_transcript(clean_text)
            self.runtime.set_status(f"Услышал: {clean_text[:60]}")
            speech = requested_speech(clean_text)
            forbidden = is_forbidden_topic(clean_text)
            confirms_offer = is_offer_confirmation(clean_text)
            backchannel = is_backchannel(clean_text)
            offer_followup = self._resolve_offer_confirmation(clean_text) if confirms_offer and not forbidden else None

            # --- Смысловой гейт: договорил ли гость? -------------------------
            # VAD режет речь по паузе (~0.75 с), поэтому человек, задумавшийся
            # посреди предложения, обрезается на полуслове. Модель
            # смотрит на ТЕКСТ и решает по смыслу: фраза завершена или оборвана.
            # Если оборвана — не отвечаем, а держим аудио и ждём продолжения
            # (жёсткий потолок по времени всё равно закроет фразу).
            if (self.gate.enabled and not self._force_phrase
                    and not speech
                    and not confirms_offer
                    and not backchannel
                    and not forbidden
                    and not is_self_question(clean_text)):
                decision = self.gate.is_complete(clean_text)
                if not decision["complete"]:
                    self._pending_raw = [raw_audio]
                    self._pending_text = clean_text
                    self._pending_since = time.time()
                    print(f"⏸️ Фраза оборвана ({decision['reason']}) — слушаю дальше.")
                    self.runtime.set_status("Слушаю дальше…")
                    self.runtime.set_transcript(clean_text, "…")
                    return

            # Прежняя настройка реакции на поддакивания.
            # if not offer_followup and is_backchannel(clean_text):
            #     if self.runtime.feature("react_to_backchannel", default=False):
            #         reply = random.choice(BACKCHANNEL_REPLIES)
            #         print(f"💬 Поддакивание — короткий ответ: {reply}")
            #         self.runtime.set_transcript(clean_text, reply)
            #         self._speak(reply)
            #     else:
            #         print("🤫 Поддакивание — отвечать не нужно, слушаю дальше.")
            #         self.runtime.set_status("Слушаю…")
            #     return
            if not offer_followup and (confirms_offer or backchannel):
                reply = "О чём Вам рассказать дальше?" if confirms_offer else "Слушаю Вас."
                log_question(raw_text=text, query_text=clean_text, answer=reply,
                             chat_id=self._chat_id, source="clarification" if confirms_offer else "backchannel")
                self.runtime.set_transcript(clean_text, reply)
                self._speak(reply)
                self.runtime.set_status("Слушаю…")
                return

            # --- Запретные темы: политика, военная тематика --------------------
            # Робот стоит в публичном месте. Такие темы не обсуждаем: коротко
            # отказываемся и предлагаем другую тему. Проверяем ДО триггеров,
            # поиска и LLM — чтобы ничего лишнего не запускалось.
            if forbidden:
                print("🚫 Запретная тема — вежливый отказ.")
                self.runtime.log_event(
                    "info", f"Отказ (запретная тема): {clean_text[:80]}"
                )
                answer = FORBIDDEN_REFUSAL
                log_question(
                    raw_text=text,
                    query_text=clean_text,
                    mirea=False,
                    web_search=False,
                    answer=answer,
                    chat_id=self._chat_id,
                    source="refusal",
                )
                self._remember_exchange(clean_text, answer, use_as_topic=False)
                self.runtime.set_transcript(clean_text, answer)
                self._speak(answer)
                print("✅ Готово. Слушаю дальше...\n")
                self.runtime.set_status("Слушаю…")
                return

            if speech:
                log_question(raw_text=text, query_text=clean_text, answer=speech,
                             chat_id=self._chat_id, source="literal-speech")
                self._remember_exchange(clean_text, speech, use_as_topic=False)
                self.runtime.set_transcript(clean_text, speech)
                self._speak(speech, literal=True)
                self.runtime.set_status("Слушаю…")
                return

            # --- Триггеры «слово → сетевая команда» (робот Unitree H1) ---------
            # Проверяем ДО любого поиска: если фраза содержит ключевое слово,
            # сразу дёргаем заданную команду (например, curl к API робота).
            trigger_results = [] if offer_followup else self.runtime.fire_triggers(clean_text, raw_text=text)
            for res in trigger_results:
                if res.get("skipped"):
                    print(f"⏱️  Триггер пропущен ({res.get('reason')}): {res.get('command')}")
                elif res.get("ok"):
                    print(f"🤖 Триггер выполнен: {res.get('command')}")
                else:
                    print(f"⚠️  Триггер не удался: {res.get('stderr') or res.get('command')}",
                          file=sys.stderr)

            # Нормализация названий: близкие к «МИРЭА» слова → "МИРЭА",
            # варианты «студсоюз» → "Студсоюз", «Лера» → «Валера» (гости часто
            # сокращают имя робота, а в базе знаний такого человека нет — поиск
            # уходил искать посторонних людей). Проверяем тему уже по
            # нормализованному тексту — тогда опечатки распознавания
            # («статсоюз», «стуцсоюз») тоже уходят в базу вуза.
            query_text = fix_asr(normalize_tech_terms(normalize_studsouz(
                normalize_name(normalize_mirea(clean_text))
            )))
            kb_hit = kb_match(offer_followup.query if offer_followup else query_text)
            mirea_related = kb_hit is not None
            self._last_mirea = mirea_related
            self._last_effective = query_text
            self._last_topic = query_text
            # Потолок длины ответа — по вопросу (факт — коротко, «расскажи» — развёрнуто).
            budget = set_answer_budget(offer_followup.query if offer_followup else query_text)
            self_question = offer_followup is None and is_self_question(query_text)
            set_answer_context(query_text, intro_ok=self_question)
            print(f"📏 Длина ответа: до {budget} слов")
            if query_text != clean_text:
                print(f"🔄 Нормализация: {query_text}")
            context = self._dialogue_context()
            pending = self._pending_followup
            self._pending_followup = None

            # --- Память: «что было раньше», «о чём говорили», «повтори» -----------
            # Достаём прошлые обмены (в том числе из прошлых чатов) и просим модель
            # ответить по ним — без поиска по базе знаний.
            if not self_question and recall_kind(query_text):
                kind, mem_rows = self.memory.recall(query_text, current_chat=self._memory_chat_id)
                print(f"🧠 Вопрос о прошлом ({kind}) — достаю из памяти: {len(mem_rows)} обмен(ов).")
                self.runtime.set_status("Вспоминаю…")
                if not mem_rows:
                    answer = "Я не помню, о чём мы говорили раньше."
                else:
                    ctx = ConversationMemory.format_context(mem_rows)
                    if kind == "repeat":
                        hint = ("Гость просит повторить. Вот твой последний ответ — повтори его своими "
                                "словами, коротко:\n" + ctx)
                    else:
                        hint = ("Гость спрашивает, что было раньше в разговорах. Вот записи из твоей "
                                "памяти (от старых к новым):\n" + ctx + "\nОтветь коротко по этим записям, "
                                "ничего не выдумывай; если нужного там нет — скажи, что не помнишь.")
                    answer = self._direct_llm(query_text, hint=hint, context=context)
                    if not answer:
                        answer = mem_rows[-1]["a"] if kind == "repeat" else \
                            "Хм, не получилось вспомнить. Спросите ещё раз."
                answer = finalize_answer(answer, keep_emotion=True)
                log_question(
                    raw_text=text, query_text=query_text, mirea=False, web_search=False,
                    answer=answer, chat_id=self._chat_id, source="memory",
                )
                self._remember_exchange(query_text, answer, persist=(kind != "repeat"), use_as_topic=False)
                self.runtime.set_transcript(clean_text, parse_emotion(answer)[0])
                self._speak(answer)
                print("✅ Готово. Слушаю дальше...\n")
                self.runtime.set_status("Слушаю…")
                return

            # --- Уточнения: «а сколько стоит?», «подробнее», «а там метро есть?» ---
            # Такая реплика сама по себе ничего не говорит базе знаний, поэтому
            # для поиска дополняем её прошлым вопросом и сохраняем тему (вуз).
            rag_text = query_text
            followup = offer_followup
            if followup is None and not self_question:
                followup = self.memory.resolve_followup(
                    query_text, self._memory_chat_id, has_own_topic=kb_hit is not None, pending=pending,
                )
            clarification = followup.clarification if followup else ""
            if followup is None and not self_question and needs_topic(query_text, kb_hit is not None):
                clarification = "Уточните, пожалуйста, о чём именно Вы спрашиваете?"
            if clarification:
                self._pending_followup = followup
                log_question(raw_text=text, query_text=query_text, answer=clarification,
                             chat_id=self._chat_id, source="clarification")
                self._remember_exchange(query_text, clarification, use_as_topic=False)
                self.runtime.set_transcript(clean_text, clarification)
                self._speak(clarification)
                self.runtime.set_status("Слушаю…")
                return
            if followup is not None:
                rag_text = followup.query
                self._last_topic = followup.topic
                self._last_effective = rag_text
                if followup.mirea and not mirea_related:
                    mirea_related = True
                    kb_hit = "уточнение"
                    self._last_mirea = True
                print(f"🔗 Уточнение к «{followup.topic[:60]}» → поиск: {rag_text[:120]}")
            continuing = bool(offer_followup or (
                followup and CONTINUE_RE.fullmatch(query_text.strip(" ,.!?…"))
            ))

            # --- Вопросы О САМОМ РОБОТЕ («ты кто», «как зовут», «Лера, ты
            # работаешь?») ---------------------------------------------------
            # Их НЕЛЬЗЯ отправлять в базу знаний: там про личность робота ничего
            # нет. Раньше поиск подсовывал случайные чанки, и робот отвечал
            # «Этого я не знаю» — или находил реального человека с похожим
            # именем (телеведущую Валерию Кудрявцеву). Отвечаем напрямую из
            # системной инструкции личности, через активного провайдера.
            if self_question:
                print("🙋 Вопрос о себе — отвечаю без поиска по базе.")
                self.runtime.set_status("Отвечаю…")
                answer = self._direct_llm(query_text, context=context)
                answer = self._answer_said_unrecognized(
                    answer, query_text, mirea_related, context=context
                )
                if not answer:
                    # Сервер недоступен — честная заготовка вместо пустоты.
                    answer = ("Я Валера, робот-экскурсовод РТУ МИРЭА. "
                              "Отвечаю на вопросы о вузе.")
                answer = finalize_answer(answer, keep_emotion=True)
                log_question(
                    raw_text=text,
                    query_text=query_text,
                    mirea=False,
                    web_search=False,
                    answer=answer,
                    chat_id=self._chat_id,
                    source="direct-self",
                )
                self._remember_exchange(query_text, answer)
                self.runtime.set_transcript(clean_text, parse_emotion(answer)[0])
                self._speak(answer)
                print("✅ Готово. Слушаю дальше...\n")
                self.runtime.set_status("Слушаю…")
                return

            # Про вуз и вузовские организации (Студсоюз, профком, студсовет,
            # ректор, общежитие…) говорим ТОЛЬКО про РТУ МИРЭА. Если спросили про
            # ЧУЖОЙ вуз (МГУ, Бауманка, МФТИ…) — тоже уходим в базу МИРЭА: там
            # ответа про чужой вуз не будет, и Валера честно скажет, что знает
            # только про РТУ МИРЭА (правило есть и в промпте).
            other_vuz = _other_university_hit(clean_text)
            if other_vuz:
                print(f"🎓 Вопрос про другой вуз («{other_vuz}») — говорю только про МИРЭА")
                mirea_related = True
                if kb_hit is None:
                    kb_hit = other_vuz

            if mirea_related:
                print(f"🎯 Тема про вуз РТУ МИРЭА (слово «{kb_hit}») → база МИРЭА")

            # Разговорные реплики («Где ты это вычитал?», «Здесь ты», «Ага»)
            # не ищутся в интернете — отвечаем напрямую через LLM (быстро).
            #
            # Сюда же попадают РЕМАРКИ без вопроса («Не знаю, по сетям там
            # что-нибудь надо», «Адреса она выдумывает какие-то. Ладно»):
            # в базе знаний их нет, а поиск всё равно возвращал случайный
            # чанк, и робот отвечал не по теме.
            conversational = (
                self.is_conversational(query_text)
                or looks_conversational_remark(query_text)
            )
            if conversational and not mirea_related:
                print(f"💬 Разговорная реплика — отвечаю напрямую через LLM...")
                self.runtime.set_status("Отвечаю (диалог)…")
                direct = self._direct_llm(query_text, context=context, search_query=rag_text)
                if direct:
                    answer = (self._finish_continuation(direct, query_text, context, rag_text)
                              if continuing else finalize_answer(direct, keep_emotion=True))
                    log_question(
                        raw_text=text,
                        query_text=query_text,
                        mirea=False,
                        web_search=False,
                        answer=answer,
                        chat_id=self._chat_id,
                        source="direct",
                        search_query=rag_text,
                    )
                    # Быстрый ответ идёт мимо WebRAgent, поэтому сами
                    # дописываем обмен в историю — иначе следующая фраза
                    # «потеряет» эту реплику.
                    self._remember_exchange(
                        query_text, answer, use_as_topic=followup is not None or not self.is_conversational(query_text),
                    )
                    print(f"🤖 Ответ: {answer[:200]}...")
                    self.runtime.set_transcript(clean_text, parse_emotion(answer)[0])
                    self._speak(answer)
                    print("✅ Готово. Слушаю дальше...\n")
                    self.runtime.set_status("Слушаю…")
                    return

            # Подтверждение: спрашиваем, хочет ли пользователь узнать об этом
            # (флаг можно переключить тумблером в GUI на лету)
            if not offer_followup and self.runtime.feature("confirm_before_answer"):
                if not self._confirm(query_text):
                    print("⏭️  Пользователь отказался — не отправляю в ИИ.\n")
                    self.runtime.set_status("Слушаю…")
                    return

            # 4. Отправляем запрос в RAG (коллекция для МИРЭА, иначе интернет)
            already_spoken = False      # поток уже озвучил ответ — повторно не говорим
            try:
                want_mirea = self.runtime.feature("mirea_collection")
                want_web = self.runtime.feature("web_search")

                if mirea_related and want_mirea:
                    collection_id = self._get_collection_id()
                    if collection_id:
                        print(f"📚 Поиск по коллекции (МИРЭА): {collection_id}")
                        self.runtime.set_status("Поиск в базе МИРЭА…")
                        streamed = None
                        if not continuing and self._stream_enabled():
                            self.runtime.set_status("Отвечаю (поток)…")
                            streamed = self._stream_answer(
                                rag_text, collection_id, context=context, original_query=clean_text,
                            )
                        if streamed is not None and streamed != "":
                            rag_result = {"response": streamed, "contexts": [{"stream": True}]}
                            already_spoken = True
                        else:
                            rag_result = self._rag_query(
                                query_text=rag_text,
                                collection_id=collection_id,
                                use_web_search=False,
                                use_agent_search=False,
                                max_results=3,
                                context=context,
                                original_query=clean_text,
                            )
                    else:
                        print("⚠️ Нет коллекций — ищу в интернете.")
                        self.runtime.set_status("Поиск в интернете…")
                        rag_result = self._rag_query(
                            query_text=rag_text,
                            use_web_search=True,
                            use_agent_search=False,
                            max_results=3,
                            context=context,
                            original_query=clean_text,
                        )
                elif not want_web:
                    # Веб-поиск выключен тумблером в GUI: отвечаем напрямую через LLM
                    print("🌐 Веб-поиск выключен — отвечаю через LLM напрямую...")
                    self.runtime.set_status("Отвечаю (LLM)…")
                    self._last_query_source = "llm"
                    rag_result = {"response": "", "contexts": []}
                else:
                    # Адаптивно: простые вопросы — быстрый веб-поиск без декомпозиции,
                    # сложные (многословные / многосторонние) — агентный поиск.
                    use_agent = self._should_use_agent_search(query_text)
                    if use_agent:
                        print("🔍 Агентный поиск (сложный вопрос)...")
                        self.runtime.set_status("Агентный поиск…")
                    else:
                        self.runtime.set_status("Поиск в интернете…")
                    rag_result = self._rag_query(
                        query_text=rag_text,
                        use_web_search=True,
                        use_agent_search=use_agent,
                        max_results=5,
                        context=context,
                        original_query=clean_text,
                    )
                answer = clean_html(rag_result.get("response"))
                # Если поиск ничего не дал (0 контекстов или пустой/обрывочный
                # ответ типа «В данный момент я.») — отвечаем напрямую через LLM,
                # как на обычную реплику диалога (например, «Здесь ты», «Привет»).
                n_contexts = len(rag_result.get("contexts", []))
                if not already_spoken and (n_contexts == 0 or len(answer.strip()) < 10):
                    print(f"🔍 Поиск не дал контекстов ({n_contexts}), отвечаю через LLM напрямую...")
                    direct = self._direct_llm(query_text, context=context, search_query=rag_text)
                    if direct:
                        answer = direct
                # Пересказываем ответ через ИИ, чтобы озвучивать живой текст,
                # а не сырые ссылки/сниппеты из поиска (тумблер в GUI).
                #
                # НО: если в WebRAgent выбрана ОБЛАЧНАЯ модель (DeepSeek), её
                # ответ уже точный, чистый и нужной длины. Пересказ локальной
                # 3B-моделью в этом случае только портит факты и имена
                # («Санислав Кудж», «МИФИ») и добавляет задержку. Поэтому
                # пересказ применяем только к локальному провайдеру.
                provider = self.rag.get_llm_provider() if self.rag else "ollama"
                cloud_llm = provider in ("deepseek", "openai", "claude")
                if cloud_llm:
                    print(f"☁️ Ответ от облачной модели ({provider}) — пересказ не нужен.")
                elif not already_spoken and self.runtime.feature("rephrase_answers"):
                    answer_ai = rephrase_answer_with_ai(rag_text, answer)
                    if answer_ai and answer_ai.strip():
                        answer = answer_ai
                # Даже облачная модель иногда отказывается отвечать на
                # сбивчивую речь («не разобрал»). Слова понятны — отвечаем.
                if not already_spoken:
                    answer = self._answer_said_unrecognized(
                        answer, query_text, mirea_related, context=context, search_query=rag_text,
                    )
                # Обрезаем, чтобы не было слишком длинно для TTS
                answer = answer[:1500]
            except Exception as e:
                print(f"❌ Ошибка RAG-запроса: {e}", file=sys.stderr)
                self.runtime.log_event("error", f"Ошибка RAG: {e}")
                self._speak("Извините, произошла ошибка при обработке запроса.")
                self.runtime.set_status("Слушаю…")
                return

            if not answer or answer == "Ответ отсутствует":
                # Ничего не нашли и модель ничего не сказала. Если в реплике
                # были осмысленные слова — переспрашиваем по теме, а чистую
                # бессмыслицу просим повторить.
                answer = (UNRECOGNIZED_REPLY if is_gibberish(query_text)
                          else clarify_reply(mirea_related))

            if continuing:
                answer = self._finish_continuation(answer, query_text, context, rag_text)
            elif not already_spoken:
                answer = finalize_answer(answer, keep_emotion=True)
            print(f"🤖 Ответ: {parse_emotion(answer)[0][:200]}")
            self.runtime.set_transcript(clean_text, parse_emotion(answer)[0])

            # Логируем заданный вопрос и ответ (для отладки/аудита)
            log_question(
                raw_text=text,
                query_text=query_text,
                mirea=mirea_related,
                collection_id=self._collection_id if mirea_related else None,
                web_search=not mirea_related,
                answer=answer,
                chat_id=self._chat_id,
                source=getattr(self, "_last_query_source", ""),
                search_query=rag_text,
            )

            # Сохраняем обмен в локальную историю (подстраховка для /query).
            # Серверный чат WebRAgent ведёт свою историю сам.
            self._remember_exchange(query_text, answer)

            # 5. Озвучиваем ответ
            if already_spoken:
                print("🔊 Ответ уже озвучен потоком.")
            else:
                self.runtime.set_status("Отвечаю голосом…")
                print("🔊 Озвучиваю ответ...")
                self._speak(answer)

            print("✅ Готово. Слушаю дальше...\n")
            self.runtime.set_status("Слушаю…")

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
            # Для короткого ответа (да/нет) используем ту же модель Kairos
            result = self._get_asr().transcribe(wav_file=tmp_path)
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
            speaker=self.runtime.get("tts", "speaker", default=self._tts_speaker),
            save_file=False,
            sample_rate=TTS_SAMPLE_RATE,
        )
        print("🗣️  Подтверждение: 'Вы хотели бы узнать об %s?'" % clean_question_for_speech)
        print("    Скажите ДА или НЕТ...")
        answer = self._record_answer(seconds=3.0)
        print(f"    Ответ: {answer!r}")
        return is_yes_answer(answer)

    def _get_collection_id(self) -> Optional[str]:
        """
        Возвращает ID коллекции МИРЭА (кэшируется).

        Сначала ищем коллекцию по имени из рантайм-настроек (по умолчанию
        «МИРЭА» — её создаёт ``kb_ingest.py``), и только если не нашли —
        берём первую попавшуюся (совместимость со старым поведением).
        """
        if self._collection_id is not None:
            return self._collection_id
        want = (self.runtime.get("rag", "collection_name", default="МИРЭА") or "").strip()
        try:
            collections = self.rag.get_collections()
            if not collections:
                return None
            # Ищем точное совпадение по имени, затем частичное
            for col in collections:
                name = (col.get("name") or "").strip()
                if want and name.lower() == want.lower():
                    self._collection_id = col["id"]
                    print(f"📚 Коллекция «{name}»: {self._collection_id}")
                    return self._collection_id
            for col in collections:
                name = (col.get("name") or "").strip()
                if want and want.lower() in name.lower():
                    self._collection_id = col["id"]
                    print(f"📚 Коллекция «{name}» (частичное совпадение): {self._collection_id}")
                    return self._collection_id
            self._collection_id = collections[0]["id"]
            print(f"📚 Коллекция по умолчанию: {collections[0].get('name')} "
                  f"({self._collection_id})")
        except Exception as e:
            print(f"❌ Не удалось получить коллекции: {e}", file=sys.stderr)
        return self._collection_id

    def _load_chat_state(self) -> tuple[Optional[str], Optional[str]]:
        """Читает серверный чат и идентификатор локальной памяти."""
        try:
            saved = Path(CHAT_ID_FILE).read_text(encoding="utf-8").splitlines()
        except OSError:
            return None, None
        server_id = (saved[0].strip() or None) if saved else None
        memory_id = (saved[1].strip() or server_id) if len(saved) > 1 else server_id
        return server_id, memory_id

    def _save_chat_id(self, chat_id: Optional[str]) -> None:
        """Сохраняет границы разговора между перезапусками."""
        try:
            server_id = chat_id or ""
            state = server_id if chat_id == self._memory_chat_id else f"{server_id}\n{self._memory_chat_id}"
            Path(CHAT_ID_FILE).write_text(state, encoding="utf-8")
        except OSError as e:
            print(f"⚠️ Не удалось сохранить chat_id: {e}", file=sys.stderr)

    def start(self):
        if self.is_recording:
            return

        # Используем существующий чат (контекст сохраняется между запусками)
        # или создаём новый, если его нет / запрошен новый.
        if self._new_chat:
            self.new_conversation()
            self._new_chat = False
        elif not self._chat_id:
            saved, memory_id = self._load_chat_state()

            # Проверяем, что сохранённый чат ещё жив (MongoDB могла
            # перезапуститься и «потерять» его).
            if saved:
                if self.rag.chat_exists(saved):
                    self._chat_id = saved
                    print(f"💬 Продолжаю существующий чат: {self._chat_id}")
                else:
                    print(f"⚠️  Сохранённый чат {saved} не найден на сервере — создаю новый.")

            if not self._chat_id:
                print("💬 Создание нового чата в WebRAgent...")
                self._chat_id = self.rag.create_chat()
                if self._chat_id:
                    print(f"✅ Чат создан: {self._chat_id} "
                          f"(контекст вопросов/ответов хранится на сервере)")
                else:
                    print("⚠️ Чат WebRAgent недоступен — использую локальную память.")
            self._memory_chat_id = memory_id or self._chat_id or self._memory_chat_id
            self._save_chat_id(self._chat_id)
            self.runtime.set_current_chat(self._chat_id, "")
            if not self._conversation_history:
                for row in self.memory.recent(self._memory_chat_id):
                    self._conversation_history.extend((
                        {"role": "user", "content": row["q"]},
                        {"role": "assistant", "content": row["a"]},
                    ))

        # Шумовой профиль ДО запуска основного захвата.
        # Тумблер «Шумодав» в GUI читается здесь же — если он выключен,
        # профиль не записываем и не тратим время на старте.
        if self.runtime.feature("noise_reduction"):
            print(f"🔇 Запись шума в течение {NOISE_PROFILE_DURATION} сек...")
            self.runtime.set_status("Калибровка шума…")
            try:
                self.noise_profile = self._record_noise_profile()
                print("✅ Профиль шума записан")
            except Exception as e:
                self.noise_profile = None
                # Тишина в микрофоне — не ошибка, а скорее «микрофон не подключён»
                if "пустые данные" in str(e):
                    print("⚠️ Микрофон не отдал звук для калибровки — "
                          "шумодав будет работать без профиля шума "
                          "(проверьте устройство ввода во вкладке «Устройства и сеть»).")
                else:
                    print(f"⚠️ Не удалось записать шум, шумоподавление отключено: {e}")
                self.runtime.log_event("error", f"Калибровка шума: {e}")
        else:
            self.noise_profile = None
            print("🔇 Шумодав выключен — профиль шума не записывается.")

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

        self.runtime.set_status("Слушаю…")
        print(f"🎙️  Говорите... (пауза {self._current_silence_timeout():.2f} сек для завершения фразы)")
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


def build_assistant(
    stt: str = "kairos",
    no_confirm: bool = False,
    new_chat: bool = False,
    no_clean: bool = False,
    speaker: str = TTS_SPEAKER,
    vad_threshold: float = VAD_THRESHOLD,
    silence_timeout: float = SILENCE_TIMEOUT_SEC,
    min_speech: float = MIN_SPEECH_DURATION_SEC,
    barge_in: bool = False,
    runtime=None,
    base_url: str = None,
    username: str = None,
    password: str = None,
    skip_tts: bool = False,
) -> "VoiceAssistant":
    """
    Собирает ассистента со всеми зависимостями (RAG, TTS, VAD).

    Вынесено из ``main()``, чтобы GUI (``valera_gui.py``) запускал тот же самый
    пайплайн, а не свою копию: то, что видно в окне, — это реальный робот.

    ``skip_tts=True`` не грузит TTS-модель (полезно, если озвучка не нужна).
    """
    rt = runtime if runtime is not None else get_runtime()

    rag = RAGClient(
        base_url=base_url or rt.get("rag", "base_url", default=RAG_BASE_URL),
        username=username or rt.get("rag", "username", default=RAG_USERNAME),
        password=password or rt.get("rag", "password", default=RAG_PASSWORD),
    )
    if not rag.login():
        raise RuntimeError("Не удалось войти в WebRAgent (проверьте, что сервис запущен).")

    neural_speaker = NeuralSpeaker()

    vad = SileroVAD(sample_rate=SAMPLE_RATE, threshold=vad_threshold)

    return VoiceAssistant(
        rag_client=rag,
        neural_speaker=neural_speaker,
        vad=vad,
        silence_timeout=silence_timeout,
        min_speech_duration=min_speech,
        no_confirm=no_confirm,
        stt=stt,
        new_chat=new_chat,
        no_clean=no_clean,
        tts_speaker=speaker,
        vad_threshold=vad_threshold,
        preload=False,
        barge_in=barge_in,
        runtime=rt,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Голосовой ассистент (Микрофон → RAG → TTS)")
    parser.add_argument(
        "--no-confirm", action="store_true",
        help="Отключить подтверждение — отправлять запрос сразу в ИИ без вопроса пользователю"
    )
    parser.add_argument(
        "--stt", choices=["kairos"], default="kairos",
        help="Модель распознавания речи (доступна Kairos — лёгкая и быстрая)"
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
        "--vad-threshold", type=float, default=None,
        help="Порог VAD в ручном режиме (0..1, ниже = чувствительнее). "
             "Если не указан — берётся из valera_config.json / авто-режима"
    )
    parser.add_argument(
        "--silence-timeout", type=float, default=None,
        help="Пауза тишины перед завершением фразы (сек)"
    )
    parser.add_argument(
        "--min-speech", type=float, default=None,
        help="Минимальная длительность фразы (сек)"
    )
    parser.add_argument(
        "--speaker", default=None,
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
    parser.add_argument(
        "--no-auto-vad", action="store_true",
        help="Выключить автоматическую регулировку чувствительности VAD (ручной порог)"
    )
    parser.add_argument(
        "--proxy", default=None,
        help="HTTP-прокси для исходящих запросов, например http://127.0.0.1:12334"
    )
    parser.add_argument(
        "--web", action="store_true",
        help="Вместо консоли поднять веб-пульт (сайт на «невзрачном» порту 8137). "
             "Это рекомендуемый способ управления — полностью заменяет настольный GUI"
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="Поднять настольный графический пульт (valera_gui.py) вместо консоли"
    )
    parser.add_argument(
        "--web-port", type=int, default=None,
        help="Порт веб-пульта (по умолчанию 8137)"
    )
    parser.add_argument(
        "--web-host", default=None,
        help="Адрес веб-пульта (0.0.0.0 — доступ с других устройств)"
    )
    args = parser.parse_args()

    # Если задан прокси — применяем его ко всем исходящим запросам процесса
    # (requests в RAGClient/Ollama читают переменные окружения).
    if args.proxy:
        os.environ['HTTP_PROXY'] = args.proxy
        os.environ['HTTPS_PROXY'] = args.proxy
        os.environ['ALL_PROXY'] = args.proxy
        os.environ['NO_PROXY'] = os.environ.get(
            'NO_PROXY', 'localhost,127.0.0.1,::1,searxng,qdrant,mongodb,app'
        )
        print(f"  🌐 Прокси: {args.proxy}")

    # --- Режим веб-пульта: сайт вместо консоли (и вместо настольного GUI) ---
    if args.web:
        import valera_web

        argv = ["valera_web.py", "--stt", args.stt]
        if args.web_port:
            argv += ["--port", str(args.web_port)]
        if args.web_host:
            argv += ["--host", args.web_host]
        if args.new_chat:
            argv.append("--no-autostart")
        sys.argv = argv
        return valera_web.main()

    # --- Режим настольного GUI ---
    if args.gui:
        import valera_gui

        argv = ["valera_gui.py", "--stt", args.stt]
        if args.new_chat:
            argv.append("--no-autostart")
        sys.argv = argv
        return valera_gui.main()

    # Рантайм-состояние: единый источник настроек (то же, что правит GUI).
    rt = get_runtime()

    # Аргументы CLI, если переданы, перекрывают значения из конфига и
    # сохраняются в него — чтобы CLI и GUI не расходились.
    if args.vad_threshold is not None:
        rt.set("vad", "threshold", value=args.vad_threshold)
    if args.silence_timeout is not None:
        rt.set("vad", "silence_timeout", value=args.silence_timeout)
    if args.min_speech is not None:
        rt.set("vad", "min_speech", value=args.min_speech)
    if args.speaker:
        rt.set("tts", "speaker", value=args.speaker)
    if args.no_auto_vad:
        rt.set_feature("auto_vad", False)
    if args.no_confirm:
        rt.set_feature("confirm_before_answer", False)

    cfg = rt.snapshot()
    print("=" * 50)
    print("  Голосовой ассистент (Микрофон → RAG → TTS)")
    print(f"  STT-модель: {args.stt}")
    print(f"  VAD: {'авто' if rt.feature('auto_vad') else 'ручной'} "
          f"(порог {cfg['vad']['threshold']}) | "
          f"Таймаут: {cfg['vad']['silence_timeout']}с | Голос: {cfg['tts']['speaker']}")
    print(f"  Шумодав: {'вкл' if rt.feature('noise_reduction') else 'выкл'} | "
          f"Веб-поиск: {'вкл' if rt.feature('web_search') else 'выкл'} | "
          f"База МИРЭА: {'вкл' if rt.feature('mirea_collection') else 'выкл'}")
    if rt.feature("confirm_before_answer"):
        print("  ❓ Подтверждение вопроса включено")
    else:
        print("  ⚡ Без подтверждения (сразу ответ)")
    if args.new_chat:
        print("  🆕 Новый чат (контекст сброшен)")
    if args.no_clean:
        print("  🧽 Без очистки слов (имя/мат сохраняются в вопросе)")
    if args.barge_in:
        print("  ⏭️ Barge-in: озвучку можно прервать речью")
    print(f"  🤖 Триггеров «слово → curl»: {len(rt.triggers())}")
    print("=" * 50)

    print("\n[1/3] Подключение к WebRAgent...")
    try:
        assistant = build_assistant(
            stt=args.stt,
            no_confirm=args.no_confirm,
            new_chat=args.new_chat,
            no_clean=args.no_clean,
            speaker=cfg["tts"]["speaker"],
            vad_threshold=cfg["vad"]["threshold"],
            silence_timeout=cfg["vad"]["silence_timeout"],
            min_speech=cfg["vad"]["min_speech"],
            barge_in=args.barge_in,
            runtime=rt,
        )
    except RuntimeError as e:
        print(f"❌ {e}")
        sys.exit(1)
    print("✅ Подключено к WebRAgent")
    print("✅ Модель TTS загружена")
    print("✅ VAD готов")

    try:
        assistant.start()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n")
    finally:
        assistant.stop()
        assistant.rag.close()
        print("👋 Выход.")


if __name__ == "__main__":
    main()
