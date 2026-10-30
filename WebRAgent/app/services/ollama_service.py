import os
import json
import logging
from datetime import datetime
import requests
from app.services.llm_service import LLMService
from app.services.model_service import ModelService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Лимит размера ТЕЛА запроса к Ollama
# ---------------------------------------------------------------------------
# В этом окружении VPN перехватывает соединения и ОБРЫВАЕТ крупные HTTP-тела
# (проверено: 2-5 КБ проходит, ~7 КБ и больше — виснет). При обрыве Ollama ждёт
# остаток тела до собственного таймаута (~107 с), всё это время слот занят,
# и ВСЕ следующие запросы встают в очередь и падают по таймауту — поиск в базе
# знаний «висел» именно из-за этого (10.4 КБ → 500 на /api/generate).
#
# Второй важный момент: `requests` при передаче `json=` сериализует с
# ensure_ascii=True, то есть КАЖДАЯ русская буква превращается в `\u0444` —
# 6 байт вместо 2. Тело раздувается В ТРИ РАЗА. Поэтому отправляем свой JSON
# с ensure_ascii=False.
# Лимит поднят 30.10.2026: новый VPN большие ответы не режет (проверено на 3 МБ).
_MAX_BODY_BYTES = int(os.getenv("OLLAMA_MAX_BODY_BYTES", "65536"))

class OllamaService(LLMService):
    """Service for interacting with Ollama API"""
    
    def __init__(self):
        """Initialize the Ollama service"""
        # Load configuration from the model service
        self.model_service = ModelService()
        self.config = self.model_service.config
        
        # Get Ollama host from environment variable
        self.ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.base_url = f"{self.ollama_host}/api"
        
        # Get model from JSON config instead of environment variable
        if self.config["active"]["models"]["ollama_llm"]:
            self.model = self.config["active"]["models"]["ollama_llm"]
        else:
            # Fallback to a default model
            self.model = 'llama2'

    @staticmethod
    def _now_context() -> str:
        """Возвращает текущую дату/время для подстановки в промпт."""
        from zoneinfo import ZoneInfo
        try:
            now = datetime.now(ZoneInfo("Europe/Moscow"))
        except Exception:
            now = datetime.now()
        weekdays = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
        months = ["января", "февраля", "марта", "апреля", "мая", "июня",
                  "июля", "августа", "сентября", "октября", "ноября", "декабря"]
        wd = weekdays[now.weekday()]
        return (f"Сегодня {wd}, {now.day} {months[now.month - 1]} {now.year} года. "
                f"Сейчас {now.hour:02d}:{now.minute:02d} по московскому времени. "
                f"Используй эту дату, когда спрашивают о сегодняшнем дне, числе или времени.")
    
    @staticmethod
    def _fit_body(payload):
        """
        Уменьшает тело запроса до безопасного размера (см. _MAX_BODY_BYTES).

        Умеет ужимать оба формата: ``{"prompt": "..."}`` и
        ``{"messages": [{"role": ..., "content": ...}]}``. Коротко обрезаемый
        хвост промпта теряет контекст, но это лучше, чем полностью зависший
        запрос: база знаний отдаёт самые нужные фрагменты в НАЧАЛЕ контекста.
        """
        def encode(p):
            return json.dumps(p, ensure_ascii=False).encode("utf-8")

        data = encode(payload)
        if len(data) <= _MAX_BODY_BYTES:
            return data

        # Находим самое длинное текстовое поле и обрезаем его.
        target = None
        if isinstance(payload.get("prompt"), str):
            target = ("prompt", None)
        elif isinstance(payload.get("messages"), list):
            best, best_len = None, -1
            for i, m in enumerate(payload["messages"]):
                if isinstance(m, dict) and isinstance(m.get("content"), str):
                    if len(m["content"]) > best_len:
                        best, best_len = i, len(m["content"])
            if best is not None:
                target = ("messages", best)

        if target is None:
            logger.warning(
                "[OLLAMA] Тело %d байт > лимита %d, но текстового поля нет — "
                "отправляю как есть", len(data), _MAX_BODY_BYTES,
            )
            return data

        key, idx = target
        text = payload[key] if idx is None else payload["messages"][idx]["content"]
        # Сколько байт остаётся на текст: вычитаем всё остальное в теле.
        if idx is None:
            probe = {**payload, key: ""}
        else:
            msgs = [dict(m) for m in payload["messages"]]
            msgs[idx] = {**msgs[idx], "content": ""}
            probe = {**payload, "messages": msgs}
        reserve = len(encode(probe))
        allow = max(400, _MAX_BODY_BYTES - reserve)

        # Режем по границе символов (чтобы не сломать UTF-8) и ВЫРЕЗАЕМ
        # СЕРЕДИНУ, а не хвост: в конце промпта стоит сам вопрос и метка
        # «Assistant:» — если отрезать хвост, модель останется без вопроса.
        # Начало (системная инструкция) и конец (вопрос) сохраняем, а теряем
        # только ненужные куски контекста из середины.
        marker = "\n\n...[часть контекста опущена]...\n\n"
        head_ratio = 0.6
        cut = text
        while len(cut.encode("utf-8")) > allow and len(cut) > 200:
            keep = int(len(cut) * 0.85)
            h = int(keep * head_ratio)
            t = keep - h
            cut = cut[:h] + marker + (cut[-t:] if t > 0 else "")
            # Режем по границе предложения, чтобы текст остался читаемым.
            for sep in (". ", "! ", "? ", "\n"):
                pos = cut.find(sep, max(0, h - 120))
                if 0 < pos < h:
                    cut = cut[: pos + 1] + marker + cut[h + len(marker):]
                    break

        logger.warning(
            "[OLLAMA] Промпт обрезан: %d → %d символов (лимит тела %d байт)",
            len(text), len(cut), _MAX_BODY_BYTES,
        )

        if idx is None:
            new_payload = {**payload, key: cut}
        else:
            msgs = [dict(m) for m in payload["messages"]]
            msgs[idx] = {**msgs[idx], "content": cut}
            new_payload = {**payload, "messages": msgs}
        return encode(new_payload)

    def _post_chat(self, messages, max_tokens, stream=False):
        """
        Отправляет запрос в Ollama через ``/api/chat`` и возвращает текст ответа.

        ПОЧЕМУ НЕ ``/api/generate``. Этот эндпоинт возвращает в ответе ещё и
        массив ``context`` — ВСЕ токены промпта числами. Замерено на реальном
        запросе: ответ на вопрос из 95 символов весил 12 011 байт, из которых
        13 548 байт (до сжатия) — именно ``context``. При контексте из базы
        знаний массив переваливал за 16 КБ, VPN обрывал ответ, клиент получал
        ноль байт и падал по таймауту — именно это выглядело как «RAG не
        работает / висит».

        ``/api/chat`` того же ответа занимает 441 байт и поля ``context`` не
        содержит вообще, поэтому ответ всегда проходит целиком.

        Плюс отправляем тело своим JSON с ``ensure_ascii=False``: стандартный
        ``requests(json=...)`` экранирует кириллицу в ``\\uXXXX``, раздувая тело
        в три раза (6 байт на букву вместо 2).
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": -1,  # модель всегда прогрета
            "options": {"num_predict": max_tokens},
        }
        if self._can_think():
            payload["think"] = False   # без режима размышлений — быстрый ответ
        if stream:
            payload["stream"] = True
            return self._iter_stream(self._fit_body(payload))
        data = self._fit_body(payload)
        resp = requests.post(
            f"{self.base_url}/chat",
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            timeout=120,
        )
        if resp.status_code >= 400:
            logger.error(
                "[OLLAMA] /api/chat → %s (тело %d байт): %s",
                resp.status_code, len(data), resp.text[:200],
            )
        resp.raise_for_status()
        return (resp.json().get("message") or {}).get("content", "") or \
            'No response received'

    def _can_think(self):
        """Всегда True: think=false безвреден для моделей без размышлений, а Gemma/Qwen без него отдают пустой ответ."""
        return True

    def _iter_stream(self, data):
        """Читает NDJSON-поток /api/chat и отдаёт кусочки текста."""
        with requests.post(
            f"{self.base_url}/chat", data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            stream=True, timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                piece = (obj.get("message") or {}).get("content") or ""
                if piece:
                    yield piece
                if obj.get("done"):
                    break

    def _stream_chat_completion(self, messages, max_tokens):
        """Потоковая версия _generate_chat_completion (тот же промпт, stream=True)."""
        self._want_stream = True
        try:
            yield from self._generate_chat_completion(messages, max_tokens)
        finally:
            self._want_stream = False

    def _generate_completion(self, system_message, user_message, max_tokens):
        """
        Generate a completion using Ollama API
        
        Args:
            system_message (str): The system message
            user_message (str): The user message
            max_tokens (int): The maximum number of tokens to generate
            
        Returns:
            str: The generated response text
        """
        # Добавляем текущую дату, чтобы модель могла отвечать на вопросы о «сегодня»
        system_message = f"{system_message}\n\n{self._now_context()}"
        return self._post_chat(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens,
        )

    def _generate_chat_completion(self, messages, max_tokens):
        """
        Generate a chat completion using Ollama API
        
        Args:
            messages (list): The formatted message list
            max_tokens (int): The maximum number of tokens to generate
            
        Returns:
            str: The generated response text
        """
        # Extract system message if present
        system_content = "You are a helpful assistant."
        for message in messages:
            if message['role'] == 'system':
                system_content = message['content']
                break

        # Добавляем текущую дату, чтобы модель могла отвечать на вопросы о «сегодня»
        system_content = f"{system_content}\n\n{self._now_context()}\n\n"
        
        # Add a strong instruction so the model prioritizes the FRESHEST messages.
        # This fixes "stale context wins": the newest user question and the last
        # assistant reply should dominate, older messages are just background.
        system_content = (
            f"{system_content}\n\n"
            "Conversation dynamics: The messages at the END of the conversation are "
            "the most important. The very last user message is your current question "
            "and must be answered first. The most recent assistant replies and the "
            "last few turns carry more weight than old ones. Use older messages only "
            "as background context; do not let them override the latest topic."
        )
        
        # Format the conversation as a text prompt since many Ollama models
        # handle this format better than structured chat messages.
        # Separate older history from the latest exchange so the model can
        # clearly see what is fresh.
        non_system = [m for m in messages if m['role'] != 'system']
        
        # Keep at most the last N messages as "recent context" (excluding current question)
        RECENT_LIMIT = 6
        history = non_system[:-1] if len(non_system) > 1 else []
        last = non_system[-1] if non_system else None
        
        conversation_prompt = f"System: {system_content}\n\n"
        
        # Older messages (background only)
        if len(history) > RECENT_LIMIT:
            conversation_prompt += "[Earlier conversation omitted — only the recent messages matter]\n\n"
            history = history[-RECENT_LIMIT:]
        
        if history:
            conversation_prompt += "[Conversation history (background):]\n"
            for msg in history:
                role_name = "User" if msg['role'] == 'user' else "Assistant"
                conversation_prompt += f"{role_name}: {msg['content']}\n\n"
            conversation_prompt += "[End of history]\n\n"
        
        # The current question — emphasized as the priority
        if last is not None:
            role_name = "User" if last['role'] == 'user' else "Assistant"
            conversation_prompt += f"[CURRENT {role_name.upper()} — answer this now]:\n{last['content']}\n\n"
        else:
            conversation_prompt += "[CURRENT USER]:\n\n"
        
        conversation_prompt += "Assistant:"

        return self._post_chat(
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": conversation_prompt},
            ],
            max_tokens,
            stream=getattr(self, "_want_stream", False),
        )