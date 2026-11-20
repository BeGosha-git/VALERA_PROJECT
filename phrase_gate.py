"""
Смысловая проверка конца фразы, предложенной темы и контекста разговора.

VAD режет речь по паузе. Гейт проверяет распознанный текст и при обрыве
просит подождать продолжения до установленного предела. Он также выделяет
предложенную роботом тему и ведёт краткую сводку разговора.

VoiceAssistant использует настроенную локальную модель Ollama. При
необходимости для гейта можно указать собственную модель в конфигурации.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import deque

import requests

# Модель по умолчанию при использовании PhraseGate отдельно от VoiceAssistant.
DEFAULT_MODEL = os.getenv("GATE_MODEL", "llama3.2:3b")

# Сколько ждём продолжения незаконченной фразы, прежде чем обработать как есть.
DEFAULT_WAIT_SEC = float(os.getenv("GATE_WAIT_SEC", "5.0"))

# Висячие на конце слова — явный признак обрыва (ASR часто теряет «?»).
# Быстрая предпроверка без вызова модели: дешевле и надёжнее на очевидных
# «…и», «…про», «…потому что».
_DANGLING_RE = re.compile(
    r"(?i)(?:^|\s)(?:и|или|но|а|да\s+и|который|которая|которое|которые|"
    r"потому\s+что|так\s+что|чтобы|если|когда|пока|на|в|во|с|со|по|про|о|об|"
    r"у|к|ко|без|для|из|от|до|над|под|за|между|через|ну|вот|э-э|эм|мм|типа|"
    r"как-то|так\s+сказать|это\s+самое|то\s+есть|и\s+всё|и\s+такое)$"
)


def _post_chat(ollama_url: str, model: str, prompt: str, max_tokens: int,
               timeout: float) -> str:
    """
    Один служебный запрос к локальной модели через ``/api/chat``.

    Тело собираем ``json.dumps(..., ensure_ascii=False)`` — кириллица уходит
    в UTF-8 без раздувания в ``\\uXXXX``. ``think: false`` безвреден для
    llama3.2 (у неё нет режима размышлений) и обязателен для Gemma/Qwen.
    """
    options = {"num_predict": max_tokens, "temperature": 0.1}
    if model.startswith("llama3.2:"):
        # У отдельной модели гейта достаточно небольшого контекста.
        options["num_ctx"] = 1024
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Ты выполняешь служебную проверку диалога. Соблюдай заданный формат ответа."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "keep_alive": -1,  # держим в памяти между вызовами
        "options": options,
    }
    resp = requests.post(
        f"{ollama_url}/api/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    if resp.status_code != 200:
        return ""
    data = resp.json()
    return ((data.get("message") or {}).get("content") or "").strip()


class PhraseGate:
    """
    Смысловой гейт фразы + контекст разговора.

    Создаётся один раз на весь запуск ассистента; модель грузится лениво при
    первом вызове и остаётся в памяти (``keep_alive=-1``).
    """

    def __init__(self, ollama_url: str = "http://localhost:11434",
                 model: str = DEFAULT_MODEL,
                 wait_sec: float = DEFAULT_WAIT_SEC,
                 timeout: float = 10.0,
                 enabled: bool = True):
        self.ollama_url = ollama_url
        self.model = model
        self.wait_sec = wait_sec
        self.timeout = timeout
        self.enabled = enabled
        # Сжатая смысловая сводка всего, что сказано роботу за сеанс.
        self.digest: str = ""
        self._last_call = 0.0
        self._digest_lock = threading.Lock()
        self._digest_queue = deque()
        self._digest_generation = 0
        self._digest_running = False

    # ------------------------------------------------------------------
    # 1. Законченность фразы
    # ------------------------------------------------------------------
    def is_complete(self, text: str) -> dict:
        """
        Возвращает ``{"complete": bool, "reason": str}``.

        ``reason`` — короткое пояснение для лога (или причина сбоя, если
        модель не ответила: тогда считаем фразу законченной и не блокируем
        робота — лучше ответить, чем молча ждать).
        """
        if not self.enabled:
            return {"complete": True, "reason": "гейт выключен"}
        text = (text or "").strip()
        if not text:
            return {"complete": True, "reason": "пусто"}

        # Быстрая детерминированная предпроверка: висячий союз/предлог на
        # конце — почти наверняка обрыв. Не тратим вызов модели.
        if _DANGLING_RE.search(text):
            return {"complete": False, "reason": "висячее слово на конце"}

        prompt = (
            "Ты определяешь, закончил ли человек говорить по смыслу.\n"
            "Отвечай ТОЛЬКО одним словом: да или нет (без кавычек, без точки, "
            "без пояснений).\n\n"
            "Ответь «нет», только если фраза ЯВНО оборвана: незаконченная "
            "мысль, обрыв на полуслове, перечисление без конца.\n"
            "Ответь «да» во всех остальных случаях, ВКЛЮЧАЯ короткие вопросы "
            "без знака «?» в конце — «Кто ректор», «Сколько общежитий», "
            "«Привет», «Какие есть направления», «Расскажи про институты». "
            "Отсутствие знака вопроса — НЕ признак обрыва.\n"
            "Если сомневаешься — отвечай «да» (лучше ответить, чем ждать).\n\n"
            f"Фраза: «{text}»"
        )
        try:
            answer = _post_chat(self.ollama_url, self.model, prompt,
                                max_tokens=8, timeout=self.timeout)
            self._last_call = time.time()
        except requests.RequestException as exc:
            return {"complete": True, "reason": f"ошибка гейта: {exc}"}

        # Парсим первое слово ответа, сняв кавычки и знаки препинания.
        low = (answer or "").strip().strip("«»\"'“”` ")
        low = re.sub(r"[^а-яёa-z]", " ", low.lower()).split()
        first = low[0] if low else ""
        if first in ("нет", "no"):
            return {"complete": False, "reason": "оборвана на полуслове"}
        if first in ("да", "yes"):
            return {"complete": True, "reason": "завершена"}
        # Непонятный ответ — не блокируем.
        return {"complete": True, "reason": f"неоднозначно: {answer[:40]!r}"}

    def detect_offered_topic(self, answer: str) -> str:
        """Выделяет тему, которую Валера предложил в предыдущем ответе.

        Модель ищет предложение продолжить разговор или перейти
        к следующему разделу экскурсии. Метод возвращает название темы,
        если оно дословно есть в ответе Валеры. При отсутствии предложения,
        выключенном гейте или ошибке запроса возвращает пустую строку.
        """
        if not self.enabled or not answer.strip():
            return ""

        prompt = (
            "Определи, предложил ли Валера тему для следующей реплики. "
            "Если он предложил рассказать о чём-то или объявил следующий "
            "раздел экскурсии, но ещё ничего о нём не сообщил, "
            "скопируй полное название темы из текста. "
            "Ответь одной строкой в формате «ТЕМА: название». "
            "Если предложения нет или тема непонятна, ответь «НЕТ».\n\n"
            f"Ответ Валеры: «{answer}»"
        )

        try:
            result = _post_chat(
                self.ollama_url,
                self.model,
                prompt,
                max_tokens=32,
                timeout=min(self.timeout, 3.0),
            )
        except requests.RequestException:
            return ""

        match = re.fullmatch(r"ТЕМА:[ \t]*([^\r\n]{1,100})", result.strip(), re.I)
        if not match:
            return ""

        topic = match.group(1).strip(' «»".')
        if not topic or _DANGLING_RE.search(topic) or topic.casefold() not in answer.casefold():
            return ""
        return topic

    # ------------------------------------------------------------------
    # 2. Смысловой учёт сказанного
    # ------------------------------------------------------------------
    def enqueue_digest(self, user_text: str, answer: str) -> None:
        """Обновляет сводку в фоне, в порядке поступления реплик."""
        if not self.enabled:
            return
        with self._digest_lock:
            self._digest_queue.append((self._digest_generation, user_text, answer))
            if not self._digest_running:
                self._digest_running = True
                threading.Thread(target=self._drain_digest_queue, daemon=True).start()

    def _drain_digest_queue(self) -> None:
        while True:
            with self._digest_lock:
                if not self._digest_queue:
                    self._digest_running = False
                    return
                generation, question, answer = self._digest_queue.popleft()
            try:
                self.update_digest(question, answer, generation=generation)
            except (AttributeError, TypeError, ValueError) as exc:
                print(f"⚠️ Гейт: неверный ответ при обновлении сводки ({type(exc).__name__}).", flush=True)

    def update_digest(self, user_text: str, answer: str, *, generation: int | None = None) -> str:
        """
        Обновляет смысловую сводку разговора и возвращает её.

        Сводка короткая (до 60 слов) и хранит ТЕМУ, ключевые факты и интересы
        гостя — всё, что основная модель должна помнить для связного диалога.
        """
        if not self.enabled:
            return self.digest

        with self._digest_lock:
            if generation is None:
                generation = self._digest_generation
            if generation != self._digest_generation:
                return self.digest
            prev = self.digest or "(пока пусто — это первая реплика)"
        prompt = (
            "Ты ведёшь краткую смысловую сводку разговора с гостем "
            "(робот-экскурсовод РТУ МИРЭА).\n"
            "Запиши ОДНИМ абзацем до 60 слов: тема, ключевые факты, что "
            "интересует гостя, имена и цифры. ТОЛЬКО по-русски, русскими "
            "буквами, без английских слов. Без приветствий и без списков.\n\n"
            f"Прошлая сводка:\n{prev}\n\n"
            f"Гость спросил: {user_text}\n"
            f"Валера ответил: {answer}\n\n"
            "Новая сводка:"
        )
        try:
            digest = _post_chat(self.ollama_url, self.model, prompt,
                                max_tokens=90, timeout=self.timeout)
            self._last_call = time.time()
        except requests.RequestException as exc:
            print(f"⚠️ Гейт: не удалось обновить сводку ({exc})", flush=True)
            return self.digest

        with self._digest_lock:
            if digest and generation == self._digest_generation:
                self.digest = digest.strip()
            return self.digest

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Сброс сводки (например, при новом чате)."""
        with self._digest_lock:
            self._digest_generation += 1
            self._digest_queue.clear()
            self.digest = ""
