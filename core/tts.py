"""Синтез речи для ответов ассистента.

Бэкенд выбирается в `.env` параметром `VALERA_TTS_BACKEND`:

* ``russian_tts`` **(по умолчанию)** — Silero v3.1_ru из папки
  ``russian_text_to_speech/`` (файл ``model.pt``). Русский голос, считается на
  **CPU** и не занимает GPU. На Jetson это в разы быстрее: ~2 с на секунду речи
  против ~13 с у встроенного Talker'а.

* ``model`` — встроенный Talker модели Qwen2.5-Omni. Озвучка тем же стеком, что
  и распознавание, но каждый кадр аудио считается на GPU, поэтому долго.

Наружу отдаётся всегда частота ``settings.sample_rate`` (24 кГц) — единая для
всего проекта, поэтому остальной код (WAV, API, клиент) менять не нужно.
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from config import settings

#: Доступные голоса Silero
SILERO_VOICES = ["xenia", "eugene", "aidar", "baya", "kseniya", "random"]

#: Ограничение Silero на длину одной реплики (символов)
SILERO_MAX_CHARS = 500


class SileroTTS:
    """Обёртка над Silero v3.1_ru (папка russian_text_to_speech)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        speaker: Optional[str] = None,
    ) -> None:
        self.model_path = Path(
            model_path
            or settings.silero_model_path
            or Path(__file__).resolve().parent.parent
            / "russian_text_to_speech"
            / "model.pt"
        )
        self.speaker = (speaker or settings.silero_speaker).lower()
        if self.speaker not in SILERO_VOICES:
            logger.warning(
                f"Голос '{self.speaker}' неизвестен, использую 'eugene'. "
                f"Доступные: {', '.join(SILERO_VOICES)}"
            )
            self.speaker = "eugene"

        # Формально Silero умеет 48000/24000/8000 Гц; берём максимум и
        # ресемплим сами — так голос звучит лучше.
        self.native_sr = 48000
        self._model = None
        self._lock = threading.Lock()

    # --- загрузка ---------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.model_path.is_file():
                raise FileNotFoundError(
                    f"Не найден файл модели Silero: {self.model_path}\n"
                    "Он должен лежать в russian_text_to_speech/model.pt"
                )
            import torch

            t0 = time.time()
            # Замерено на Jetson AGX Orin: 4 потока — оптимум для Silero
            # (2 → 4.0 с, 4 → 3.45 с, 8 → 3.75 с на 3.6 с речи).
            # На GPU этот Silero медленнее (4.5 с): маленькая модель, много
            # мелких операций, доминирует накладные расходы ядер.
            torch.set_num_threads(int(os.getenv("VALERA_SILERO_THREADS", "4")))
            importer = torch.package.PackageImporter(str(self.model_path))
            self._model = importer.load_pickle("tts_models", "model")
            self._model.to("cpu")  # CPU освобождает GPU для языковой модели
            logger.info(
                f"Silero TTS загружен за {time.time() - t0:.1f} с "
                f"(голос: {self.speaker}, CPU)"
            )

    # --- подготовка текста ------------------------------------------------
    @staticmethod
    def _ordinal_year(n: int, case: str) -> str:
        """Порядковое числительное для года в нужном падеже.

        Silero НЕ читает многозначные числа — «в 1958 году» и «в году» звучат
        одинаково (год просто пропадает). Поэтому числа обязательно пишем
        словами, причём год — порядковым: «в тысяча девятьсот пятьдесят
        восьмом году».
        """
        from num2words import num2words

        word = num2words(n, to="ordinal", lang="ru")
        if case == "nom":
            return word
        # «восьмой» → «восьмом» (предложный) / «восьмого» (родительный)
        for end, (gen, prep) in (
            ("ой", ("ого", "ом")),
            ("ый", ("ого", "ом")),
            ("ий", ("его", "ем")),
        ):
            if word.endswith(end):
                return word[: -len(end)] + (prep if case == "prep" else gen)
        return word

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Числа → слова, латиница → кириллица.

        Нужно для Silero: многозначные цифры он молча пропускает.
        """
        text = (text or "").strip()
        if not text:
            return ""

        try:
            from num2words import num2words
        except ImportError:
            num2words = None

        if num2words is not None and not settings.speak_numbers_as_words:
            num2words = None  # числа словами отключены в настройках

        if num2words is not None:
            # 0) десятилетия: «в 1990-х годах» → «в тысяча девятьсот девяностых годах»
            def _decade(m: re.Match) -> str:
                n = int(m.group(1))
                word = num2words(n, to="ordinal", lang="ru")  # …девяностый
                for end, plural in (("ой", "ых"), ("ый", "ых"), ("ий", "их")):
                    if word.endswith(end):
                        word = word[: -len(end)] + plural
                        break
                return f"{word} {m.group(2)}"

            text = re.sub(
                r"\b(\d{3,4})\s*-?\s*(?:х|е|ые|ых)\s+(годах|годы|годов|годам)\b",
                _decade,
                text,
                flags=re.IGNORECASE,
            )

            # 1) годы: «в 1958 году» → «в тысяча девятьсот пятьдесят восьмом году»
            def _year(m: re.Match) -> str:
                n, tail = int(m.group(1)), (m.group(2) or "").lower()
                if tail in ("годы", "годах", "годам", "годов"):
                    # «в 1990-х годах» — десятилетия, порядковое во мн. ч. сложно,
                    # берём количественное: главное — чтобы год прозвучал
                    return f"{num2words(n, lang='ru')} {tail}"
                if tail in ("году",):
                    case = "prep"
                elif tail in ("года", "г.", "гг."):
                    case = "gen"
                else:
                    case = "nom"
                return f"{SileroTTS._ordinal_year(n, case)} {tail}".strip()

            text = re.sub(
                r"\b(\d{3,4})\s*-?\s*(году|годы|годов|годах|годам|года|год|гг\.|г\.)",
                _year,
                text,
                flags=re.IGNORECASE,
            )

            # 2) проценты: «15 %» → «пятнадцать процентов»
            def _percent(m: re.Match) -> str:
                n = int(m.group(1))
                n %= 100
                if 11 <= n <= 14:
                    form = "процентов"
                elif n % 10 == 1:
                    form = "процент"
                elif n % 10 in (2, 3, 4):
                    form = "процента"
                else:
                    form = "процентов"
                return f"{num2words(n, lang='ru')} {form}"

            text = re.sub(r"(\d+)\s*%", _percent, text)

            # 3) остальные числа — количественные
            text = re.sub(
                r"-?[0-9][0-9,._]*",
                lambda m: num2words(m.group().replace(",", "."), lang="ru"),
                text,
            )

        try:  # опционально: transliterate (Silero не читает латиницу)
            from transliterate import translit

            text = translit(text, "ru")
        except ImportError:
            pass

        return text

    @staticmethod
    def _split_text(text: str, limit: int = SILERO_MAX_CHARS) -> list[str]:
        """Режет длинную реплику по предложениям — Silero не любит длинный вход."""
        if len(text) <= limit:
            return [text]

        parts, current = [], ""
        for sentence in re.split(r"(?<=[.!?…])\s+", text):
            if len(current) + len(sentence) + 1 <= limit:
                current = f"{current} {sentence}".strip()
            else:
                if current:
                    parts.append(current)
                # предложение само длиннее лимита — режем по запятым/словам
                while len(sentence) > limit:
                    cut = sentence.rfind(" ", 0, limit)
                    cut = cut if cut > 0 else limit
                    parts.append(sentence[:cut].strip())
                    sentence = sentence[cut:].strip()
                current = sentence
        if current:
            parts.append(current)
        return [p for p in parts if p]

    # --- основной метод ---------------------------------------------------
    def synthesize(self, text: str) -> np.ndarray:
        """Синтезирует речь и возвращает float32-массив в settings.sample_rate."""
        text = self._normalize_text(text)
        if not text:
            return np.zeros(0, dtype=np.float32)

        self._ensure_loaded()

        import torch

        chunks: list[np.ndarray] = []
        with self._lock:  # модель не потокобезопасна
            for part in self._split_text(text):
                try:
                    audio = self._model.apply_tts(
                        text=part,
                        speaker=self.speaker,
                        sample_rate=self.native_sr,
                    )
                except ValueError as exc:
                    logger.warning(f"Silero не смог озвучить фрагмент: {exc}")
                    continue
                chunks.append(audio.detach().cpu().numpy().astype(np.float32))

        if not chunks:
            return np.zeros(0, dtype=np.float32)

        # Пауза между фрагментами, чтобы речь не была «слипшейся»
        silence = np.zeros(int(0.15 * self.native_sr), dtype=np.float32)
        waveform = chunks[0]
        for chunk in chunks[1:]:
            waveform = np.concatenate([waveform, silence, chunk])

        # Пауза В НАЧАЛЕ и в конце: звуковые устройства «съедают» первые
        # десятки миллисекунд при старте потока, из-за чего речь начиналась
        # не с первого звука. Свинцовые тишины это лечат.
        lead_in = np.zeros(int(0.25 * self.native_sr), dtype=np.float32)
        lead_out = np.zeros(int(0.20 * self.native_sr), dtype=np.float32)
        waveform = np.concatenate([lead_in, waveform, lead_out])

        return _resample(waveform, self.native_sr, settings.sample_rate)


def _resample(waveform: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Приводит частоту дискретизации к dst_sr."""
    if src_sr == dst_sr or waveform.size == 0:
        return waveform.astype(np.float32)
    try:
        from scipy.signal import resample_poly

        from math import gcd

        g = gcd(src_sr, dst_sr)
        return resample_poly(waveform, dst_sr // g, src_sr // g).astype(np.float32)
    except ImportError:  # pragma: no cover
        # грубый фолбэк: линейная интерполяция
        n = int(round(len(waveform) * dst_sr / src_sr))
        return np.interp(
            np.linspace(0, len(waveform) - 1, n),
            np.arange(len(waveform)),
            waveform,
        ).astype(np.float32)


# ---------------------------------------------------------------------------
# Единая точка доступа
# ---------------------------------------------------------------------------

_silero: Optional[SileroTTS] = None
_silero_lock = threading.Lock()


def get_silero() -> SileroTTS:
    """Возвращает единственный экземпляр SileroTTS (ленивая загрузка)."""
    global _silero
    if _silero is None:
        with _silero_lock:
            if _silero is None:
                _silero = SileroTTS()
    return _silero


def tts_backend(override: Optional[str] = None) -> str:
    """Текущий бэкенд синтеза: 'russian_tts' или 'model'.

    Args:
        override: значение из запроса (перебивает настройку из .env).
    """
    return (override or settings.tts_backend or "russian_tts").strip().lower()


def tts_uses_model(override: Optional[str] = None) -> bool:
    """True, если озвучка идёт встроенным Talker'ом модели."""
    return tts_backend(override) in ("model", "qwen", "builtin")


def synthesize(text: str) -> np.ndarray:
    """Озвучивает текст выбранным бэкендом (только для внешнего TTS)."""
    if tts_uses_model():
        raise RuntimeError(
            "Бэкенд 'model' озвучивает ответ прямо в generate_response(), "
            "вызывать synthesize() для него не нужно."
        )
    return get_silero().synthesize(text)


def describe_backend() -> str:
    """Человекочитаемое описание бэкенда — для /health и логов."""
    if tts_uses_model():
        return f"model (Talker Qwen, голос {settings.speaker_voice})"
    return f"russian_tts (Silero, голос {get_silero().speaker}, CPU)"
