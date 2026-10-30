"""
Сервис DeepSeek — облачная LLM для ответов по базе знаний РТУ МИРЭА.

Зачем отдельный провайдер. Локальная 3B-модель (Ollama) быстрая, но на
фактических вопросах («кто ректор», «сколько бюджетных мест») часто путает
факты и искажает имена. DeepSeek даёт заметно более точные ответы по
контексту базы знаний, при этом стоит копейки и работает без GPU.

Выбор провайдера — на сайте: /admin/models → «DeepSeek» → выбрать модель.
Переключение применяется сразу, перезапуск не нужен (LLMFactory создаёт
сервис на каждый запрос и читает активную конфигурацию).

API DeepSeek полностью совместим с OpenAI SDK, поэтому используется
официальный клиент `openai` с подменой `base_url`. Тело запроса при этом
собирает сам SDK — кириллица уходит в UTF-8, без раздувания в `\\uXXXX`.
"""

import os

from openai import OpenAI

from app.services.llm_service import LLMService
from app.services.model_service import ModelService

# Официальный адрес API DeepSeek (OpenAI-совместимый интерфейс).
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# Модели по умолчанию и их назначение:
#   deepseek-chat     — быстрая универсальная модель, лучше для живого диалога;
#   deepseek-reasoner — «думающая» модель, точнее на сложных вопросах,
#                       но отвечает медленнее (не подходит для голоса).
DEFAULT_MODEL = "deepseek-chat"

# ---------------------------------------------------------------------------
# «ГРАДУСЫ» (temperature) — живость речи
# ---------------------------------------------------------------------------
# 0.3, которые стояли раньше, делали DeepSeek сухим и предсказуемым: ответы
# звучали как справка («Ректор — Станислав Алексеевич Кудж, доктор наук…»).
# Для робота-экскурсовода нужна живая разговорная речь, поэтому температуру
# поднимаем — но РАЗНУЮ для разных задач:
#
#   PERSONA — разговор о себе, приветствия, болталка: можно свободнее;
#   FACT    — ответ по базе знаний: выше 0.7 модель начинает путать цифры
#             и имена, поэтому держим умеренно;
#   DEFAULT — прочее.
#
# Все значения переопределяются переменными окружения, чтобы можно было
# подкрутить без правки кода.
TEMPERATURE_PERSONA = float(os.getenv("DEEPSEEK_TEMPERATURE_PERSONA", "0.95"))
TEMPERATURE_FACT = float(os.getenv("DEEPSEEK_TEMPERATURE_FACT", "0.65"))
TEMPERATURE_DEFAULT = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.80"))

# top_p и штрафы — против «зацикливания» и повторов одной и той же фразы,
# из-за которых речь звучит механически.
TOP_P = float(os.getenv("DEEPSEEK_TOP_P", "0.95"))
PRESENCE_PENALTY = float(os.getenv("DEEPSEEK_PRESENCE_PENALTY", "0.4"))
FREQUENCY_PENALTY = float(os.getenv("DEEPSEEK_FREQUENCY_PENALTY", "0.3"))



class DeepSeekService(LLMService):
    """Ответы через DeepSeek API."""

    def __init__(self):
        # Ключ берём ТОЛЬКО из окружения (файл .env), в коде его нет.
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPSEEK_API_KEY не задан. Добавьте ключ в WebRAgent/.env "
                "(строка DEEPSEEK_API_KEY=...) и перезапустите контейнер."
            )

        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)

        # Активная модель — из общей конфигурации моделей (её меняет сайт).
        self.model_service = ModelService()
        self.config = self.model_service.config
        active = (self.config.get("active", {}).get("models", {}) or {})
        self.model = active.get("deepseek_llm") or DEFAULT_MODEL

    def _temperature_for(self, system_message: str, messages=None) -> float:
        """
        Подбирает температуру под тип запроса.

        Определяем по СОДЕРЖАНИЮ системного промта (он у нас один на каждый
        режим и содержит узнаваемые маркеры). Так не нужно менять сигнатуры
        методов у всех провайдеров.
        """
        text = (system_message or "").lower()
        if messages:
            for msg in messages:
                if msg.get("role") == "system":
                    text = (msg.get("content") or "").lower()
                    break

        # persona — РАЗГОВОР О СЕБЕ. Самая свободная речь: приветствия,
        # «что умеешь», ответы на реплики. Фактов тут нет, ошибиться не в чем,
        # зато живая интонация важна больше всего.
        if "unitree" in text or "робот-экскурсовод" in text:
            if "не выдумывай" not in text and "согласно источникам" not in text:
                return TEMPERATURE_PERSONA

        # Ответ ПО БАЗЕ ЗНАНИЙ: имена, цифры, адреса. Здесь нужна точность,
        # поэтому температура ниже.
        if "не выдумывай" in text or "ничего не выдумывай" in text:
            return TEMPERATURE_FACT

        return TEMPERATURE_DEFAULT

    def _generate_completion(self, system_message, user_message, max_tokens):
        """
        Один ответ на пару «системная инструкция + вопрос».

        ``max_tokens`` ограничивается разумным минимумом: для голосового
        робота ответ — 1–2 предложения, и большой лимит только провоцирует
        модель «лить воду» и тратить время.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max(max_tokens, 64),
            temperature=self._temperature_for(system_message),
            top_p=TOP_P,
            presence_penalty=PRESENCE_PENALTY,
            frequency_penalty=FREQUENCY_PENALTY,
        )
        return (response.choices[0].message.content or "").strip()

    def _stream_chat_completion(self, messages, max_tokens):
        """Потоковая генерация (OpenAI-совместимый stream=True)."""
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max(max_tokens, 64),
            temperature=self._temperature_for("", messages),
            top_p=TOP_P,
            presence_penalty=PRESENCE_PENALTY,
            frequency_penalty=FREQUENCY_PENALTY,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            piece = getattr(chunk.choices[0].delta, "content", None)
            if piece:
                yield piece

    def _generate_chat_completion(self, messages, max_tokens):
        """
        Ответ с учётом истории диалога.

        Сообщения приходят уже подготовленными (с подмешанным контекстом
        базы знаний), поэтому передаём их как есть — DeepSeek понимает
        стандартный формат ролей system/user/assistant.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max(max_tokens, 64),
            temperature=self._temperature_for("", messages),
            top_p=TOP_P,
            presence_penalty=PRESENCE_PENALTY,
            frequency_penalty=FREQUENCY_PENALTY,
        )
        return (response.choices[0].message.content or "").strip()
