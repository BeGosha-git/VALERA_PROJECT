"""Фильтры и нормализация текста (перенесено из ветки PC/main).

Зачем: распознавание речи (ASR) почти всегда слышит «МИРЭА» как «мир»,
«мире», «мирэ», «миру» и т.п. Модель на такой вопрос отвечает про мир вообще,
а не про университет. Нормализация подменяет такие слова на «МИРЭА» и
позволяет дальше искать по нужной коллекции документов.

Две группы слов:

* ``MIREA_STRICT_ALIASES`` — только искажённая аббревиатура («мирэ», «мирэа»).
  Включаются всегда, обычную речь не портят.
* ``MIREA_WORD_ALIASES`` — обычные падежи слова «мир» (мир, мира, миру…).
  Это агрессивный режим из оригинала (робот-экскурсовод МИРЭА), потому что
  «расскажи про мир» и «расскажи про МИРЭА» ASR различает плохо. Включается
  только при ``VALERA_MIREA_AGGRESSIVE=true``, иначе фраза «во всём мире»
  превратится в «во всём МИРЭА».
"""

from __future__ import annotations

import os
import re

#: Искажённое ASR-написание аббревиатуры — безопасно заменять всегда
MIREA_STRICT_ALIASES = [
    "мирэ", "мирэа", "мирэа.", "мирэи", "мирэу", "мирэе", "мирэю", "мирэей",
    "мирэаим", "мрэа", "миреа", "миреа.", "миреаим",
]

#: Обычные падежи слова «мир» — включаются флагом VALERA_MIREA_AGGRESSIVE
MIREA_WORD_ALIASES = [
    "мир", "мира", "миру", "мире", "миры", "миров", "мирами", "мирах",
]

#: Алиасы, которые реально используются (заполняется configure_mirea_filter)
MIREA_ALIASES = list(MIREA_STRICT_ALIASES)

#: Ключевые слова, при которых ищем по коллекции документов, а не в интернете
COLLECTION_KEYWORDS = [
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


def configure_mirea_filter(aggressive: bool = False) -> None:
    """Пересобирает список алиасов (обычный или агрессивный режим)."""
    global MIREA_ALIASES
    MIREA_ALIASES = list(MIREA_STRICT_ALIASES)
    if aggressive:
        MIREA_ALIASES = MIREA_WORD_ALIASES + MIREA_ALIASES
    # Убираем дубликаты, сохраняя порядок (длинные слова — раньше коротких,
    # иначе «мирэа» может быть перебито правилом для «мирэ»)
    MIREA_ALIASES = sorted({a for a in MIREA_ALIASES if a}, key=len, reverse=True)
    _rebuild_patterns()


def _rebuild_patterns() -> None:
    global _MIREA_RE, _KEYWORDS_RE
    _MIREA_RE = _compile(MIREA_ALIASES)
    _KEYWORDS_RE = _compile(MIREA_ALIASES + COLLECTION_KEYWORDS)


def _compile(words: list[str]) -> re.Pattern | None:
    if not words:
        return None
    # \b в Python плохо работает с кириллицей, поэтому границы задаём явно
    alternation = "|".join(re.escape(w) for w in words)
    return re.compile(
        rf"(?<![0-9A-Za-zА-Яа-яЁё])({alternation})(?![0-9A-Za-zА-Яа-яЁё])",
        re.IGNORECASE,
    )


def normalize_mirea(text: str) -> str:
    """Заменяет слова, похожие на МИРЭА, на «МИРЭА» (по границам слов)."""
    if not text:
        return text
    if _MIREA_RE is None:
        return text
    return _MIREA_RE.sub("МИРЭА", text)


def is_mirea_related(text: str) -> bool:
    """Есть ли в тексте упоминание МИРЭА или ключевых слов коллекции."""
    if not text:
        return False
    return _KEYWORDS_RE is not None and _KEYWORDS_RE.search(text) is not None


# ---------------------------------------------------------------------------
# Обрезка вежливых «хвостов» в ответах модели
# ---------------------------------------------------------------------------
# Модель регулярно добавляет «Если у вас есть ещё вопросы, задавайте» — на
# Jetson это лишние ~7 токенов генерации и ~2 секунды синтеза речи. Персона
# с просьбой не делать так помогает не всегда, поэтому подчищаем ответ сами.
#: «Служебное» предложение — целиком, а не часть фразы
_COURTESY_SENTENCE = re.compile(
    r"""^\s*(
        если \s+ у \s+ (?:вас|тебя)
      | если \s+ (?:нужна|понадобится|будет) \s+ помощь
      | если \s+ (?:хочешь|хотите|что-то|что-нибудь)
      | if \s+ you \s+ have \s+ (?:any \s+ )?(?:other|more|additional)
      | if \s+ you \s+ (?:have|need) \s+ (?:any \s+ )?questions?
      | if \s+ you \s+ need \s+ (?:any \s+ )?(?:help|assistance)
      | feel \s+ free \s+ to \s+ ask
      | let \s+ me \s+ know
      | (?: i \s+ )? hope \s+ (?: this | that ) \s+ helps?
      | задавайте
      | задавай
      | спрашивайте
      | спрашивай
      | обращайтесь
      | обращайся
      | пишите
      | пиши
      | звоните
      | звони
      | заходи
      | если \s+ что \s+ (?:нужно|надо)
      | чем \s+ (?:ещё|еще) \s+ могу \s+ помочь
      | (?:всегда|буду) \s+ рад
      | рад \s+ был \s+ помочь
      | надеюсь[,!]?
      | удачи
      | всего \s+ доброго
      | хорошего \s+ дня
    )\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

#: Минимальная длина «полезной» части — чтобы не обрезать ответ целиком
MIN_KEEP_CHARS = 10


def is_courtesy_sentence(sentence: str) -> bool:
    """True, если предложение целиком служебное («спрашивай, если что»)."""
    return bool(sentence) and _COURTESY_SENTENCE.match(sentence.strip()) is not None


# ---------------------------------------------------------------------------
# Нарезка текста на куски для озвучки
# ---------------------------------------------------------------------------
#: Граница куска: запятая/тире/двоеточие/точка с запятой (с пробелом) или
#: конец предложения (с пробелом либо в самом конце строки).
_SPEECH_BOUNDARY = re.compile(r"[,\u2014;:]\s+|[.!?\u2026](?=\s|$)")
#: Знак конца предложения
_SENTENCE_TERMINATOR = re.compile(r"[.!?\u2026]\s*$")
#: Слова вместе с разделителем — чтобы не резать слово пополам
_COMPLETE_WORD = re.compile(r"\S+\s+")
#: Деление текста на предложения (с сохранением знака в конце)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\u2026])\s+")


def split_speech_chunks(
    text: str,
    min_words: int = 3,
    max_words: int = 25,
    keep_words: int = 2,
) -> tuple[list[str], str]:
    """Режет текст на куски для озвучки **по ближайшей запятой или точке**.

    Args:
        text: накопленный текст ответа.
        min_words: не отдавать кусок короче — ждать следующей запятой
            (одно-двухсловные куски звучат рвано).
        max_words: предохранитель — если пунктуации долго нет, резать принудительно.
        keep_words: сколько слов оставить в остатке при принудительном резе.

    Returns:
        ``(chunks, rest)`` — готовые куски и остаток буфера.
    """
    chunks: list[str] = []
    buf = text

    while True:
        match = _SPEECH_BOUNDARY.search(buf)
        if not match:
            break

        candidate = buf[: match.end()].strip()
        rest = buf[match.end():]

        if not candidate:
            buf = rest
            continue

        # Служебные фразы («если есть вопросы, спрашивай») не озвучиваем:
        # пока предложение не закончилось — просто ждём, а законченное — выкидываем
        if is_courtesy_sentence(candidate):
            if _SENTENCE_TERMINATOR.search(candidate):
                buf = rest
                continue
            break

        # Кусок слишком короткий и это не конец предложения — берём до
        # следующей границы (иначе на «Я Валера, гид…» буфер застрянет)
        end_of_sentence = bool(_SENTENCE_TERMINATOR.search(candidate))
        if not end_of_sentence and len(_COMPLETE_WORD.findall(candidate)) < min_words:
            nxt = _SPEECH_BOUNDARY.search(buf, match.end())
            if not nxt:
                break  # ждём ещё текста
            wider = buf[: nxt.end()].strip()
            if is_courtesy_sentence(wider):
                if _SENTENCE_TERMINATOR.search(wider):
                    buf = buf[nxt.end():]
                    continue
                break
            candidate, rest = wider, buf[nxt.end():]

        chunks.append(candidate)
        buf = rest

    # Предохранитель: пунктуации нет слишком долго
    words = _COMPLETE_WORD.findall(buf)
    if len(words) > max_words:
        take = len(words) - keep_words
        head = "".join(words[:take])
        chunks.append(head.strip())
        buf = buf[len(head):].lstrip()

    return chunks, buf


def split_courtesy_tail(text: str) -> str:
    """Хвост ответа для озвучки: без служебных предложений в начале.

    В отличие от :func:`strip_courtesy` не бросает короткий остаток: если после
    выкидывания служебной фразы ничего не осталось — вернёт пустую строку, а
    иначе — только полезный текст.
    """
    text = (text or "").strip()
    if not text:
        return ""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    if not parts:
        return text
    # Выкидываем служебные предложения, но только если останется что сказать
    kept = [p for p in parts if not is_courtesy_sentence(p)]
    if not kept:
        return ""
    return " ".join(kept).strip()


_LATIN_WORD = re.compile(r"[A-Za-z]{3,}")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")


def looks_english(text: str) -> bool:
    """True, если текст похож на английский: есть латинские слова и нет кириллицы.

    Используется как страховка к персоне при VALERA_RUSSIAN_ONLY=true:
    промпт просят отвечать по-русски, но модель иногда всё равно уезжает
    в английский (особенно 3B).
    """
    if not text or not text.strip():
        return False
    return bool(_LATIN_WORD.search(text)) and not _CYRILLIC.search(text)


def strip_courtesy(text: str, min_keep: int = MIN_KEEP_CHARS) -> str:
    """Убирает служебные «хвосты» вроде «если у вас есть ещё вопросы, задавайте».

    Режется **по предложениям** (целиком), поэтому текст не рвётся посередине и
    пунктуация полезной части сохраняется. Если после обрезки остаётся меньше
    ``min_keep`` символов — возвращается исходный текст.
    """
    if not text:
        return text

    sentences = re.split(r"(?<=[.!?…])\s+", text.strip())
    kept = list(sentences)
    while len(kept) > 1 and _COURTESY_SENTENCE.match(kept[-1]):
        kept.pop()

    if len(kept) == len(sentences):
        return text

    result = " ".join(kept).strip()
    return result if len(result) >= min_keep else text


# Инициализация при импорте (обычный, неагрессивный режим)
_MIREA_RE: re.Pattern | None = None
_KEYWORDS_RE: re.Pattern | None = None
_rebuild_patterns()

# Агрессивный режим берём из окружения, чтобы модуль не зависел от config.py
configure_mirea_filter(
    os.getenv("VALERA_MIREA_AGGRESSIVE", "").strip().lower() in ("1", "true", "yes", "on")
)
