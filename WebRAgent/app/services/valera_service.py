"""Провайдер локальной модели Qwen2.5-Omni (проект VALERA_PROJECT).

Заменяет Ollama: вместо внешнего `ollama serve` используется наш собственный
FastAPI-сервер (`python main.py`), который держит модель Qwen2.5-Omni-7B
в памяти на GPU и отдаёт чистую генерацию через `/api/v1/llm/generate`.

Важно: WebRAgent делает RAG сам, поэтому вызывается именно «сырой» эндпоинт —
без персоны «Валера», без авто-RAG и без истории диалога.
"""

import os

import requests

from app.services.llm_service import LLMService
from app.services.model_service import ModelService

#: Базовый URL нашего API (см. config.py: VALERA_API_HOST/VALERA_API_PORT)
DEFAULT_API_BASE = "http://localhost:8765/api/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-Omni-7B"


class ValeraService(LLMService):
    """LLM-провайдер поверх локального сервера VALERA_PROJECT."""

    def __init__(self):
        """Читает адрес API и имя модели из окружения/конфига."""
        self.model_service = ModelService()
        self.config = self.model_service.config

        self.api_base = os.getenv("VALERA_API_BASE", DEFAULT_API_BASE).rstrip("/")

        # Таймаут щедрый: на Jetson инференс небыстрый (десятки секунд).
        self.timeout = float(os.getenv("VALERA_TIMEOUT", "600"))

        models = self.config.get("active", {}).get("models", {})
        self.model = (
            models.get("valera_llm")
            or os.getenv("VALERA_MODEL_NAME_OR_PATH")
            or DEFAULT_MODEL
        )

    def is_available(self):
        """Проверяет, что наш сервер поднят и модель загружена."""
        try:
            resp = requests.get(f"{self.api_base}/health", timeout=5)
            resp.raise_for_status()
            return bool(resp.json().get("model_loaded"))
        except Exception:
            return False

    def _call(self, messages, max_tokens):
        """Отправляет сообщения на /llm/generate и возвращает текст ответа."""
        resp = requests.post(
            f"{self.api_base}/llm/generate",
            json={"messages": messages, "max_new_tokens": max_tokens},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("text", "No response received")

    def _generate_completion(self, system_message, user_message, max_tokens):
        """Одиночный запрос «system + user»."""
        return self._call(
            [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens,
        )

    def _generate_chat_completion(self, messages, max_tokens):
        """Многоходовой чат — Qwen2.5-Omni нативно понимает chat-формат."""
        clean = [
            {"role": m["role"], "content": m.get("content") or ""}
            for m in messages
            if m.get("role") in ("system", "user", "assistant")
        ]
        if not any(m["role"] == "user" for m in clean):
            return "No user message provided"
        return self._call(clean, max_tokens)
