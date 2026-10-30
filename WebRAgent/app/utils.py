"""
Общие утилиты приложения.

Здесь живёт ``slim_result`` — уменьшение JSON-ответа RAG перед отправкой
клиенту. Это не оптимизация «для красоты», а необходимость конкретного
окружения: VPN перехватывает соединения и ОБРЫВАЕТ HTTP-ответы крупнее
~16 КБ, поэтому клиент получает ноль байт и падает по таймауту.

Симптом до правки: запрос к ``/chat/<id>/query`` отдавал 16 809 байт, из них
большую часть занимали ПОЛНЫЕ тексты найденных чанков (``contexts``) и
продублированный ответ (``response`` + ``formatted_response``). Ответ обрывался,
и голосовой ассистент сообщал «Ошибка соединения».
"""

import json

# Сколько символов контекста оставлять в ответе.
#
# Голосовому ассистенту нужен только САМ ОТВЕТ и ЧИСЛО найденных фрагментов
# (по нему он решает, отвечать или признать, что данных нет). Полные тексты
# чанков он не использует, поэтому их можно безопасно обрезать до превью — это
# и удерживает ответ в пределах лимита.
# Лимит поднят 30.10.2026: новый VPN большие ответы не режет (проверено на 3 МБ).
CONTEXT_PREVIEW_CHARS = 2000


def slim_result(result, context_chars: int = CONTEXT_PREVIEW_CHARS):
    """
    Уменьшает результат поиска, сохраняя всё нужное клиенту.

    Что делаем:
      * обрезаем ``contexts[].content`` до ``context_chars`` (превью);
      * НЕ дублируем ответ в ``formatted_response`` — поле нужно только
        серверным шаблонам, а JSON-клиенту оно лишь удваивает размер.

    Возвращает новый словарь; исходный не меняется.
    """
    if not isinstance(result, dict):
        return result

    slim = dict(result)

    contexts = slim.get("contexts")
    if isinstance(contexts, list):
        trimmed = []
        for ctx in contexts:
            if not isinstance(ctx, dict):
                trimmed.append(ctx)
                continue
            item = dict(ctx)
            content = item.get("content")
            if isinstance(content, str) and len(content) > context_chars:
                item["content"] = content[:context_chars] + "…"
                item["content_truncated"] = True
            trimmed.append(item)
        slim["contexts"] = trimmed

    # Ответ уже есть в `response`, дубликат только раздувает тело.
    slim.pop("formatted_response", None)

    return slim


def estimate_size(obj) -> int:
    """
    Оценивает размер JSON-ответа в байтах (в UTF-8, без экранирования).

    Важно считать именно так: ``len(json.dumps(obj))`` даёт раздутую оценку,
    потому что по умолчанию кириллица экранируется в ``\\uXXXX`` (6 байт на
    букву вместо 2).
    """
    try:
        return len(json.dumps(obj, ensure_ascii=False).encode("utf-8"))
    except (TypeError, ValueError):
        return 0
