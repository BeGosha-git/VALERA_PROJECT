import os
import re
import uuid
import mimetypes
import logging
import tempfile
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Docling imports for advanced document processing
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling import document_converter

# Keep original imports for fallback handling
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import markdown
from werkzeug.datastructures import FileStorage

from app.models.document import Document
from app.models.collection import Collection
from app.services.qdrant_service import QdrantService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Гибридное ранжирование (вектор + лексика)
# ---------------------------------------------------------------------------
# Коллекция «МИРЭА» использует all-MiniLM-L6-v2 — АНГЛОЯЗЫЧНУЮ модель. На
# русских вопросах она даёт посредственные вектора (score 0.38-0.52) и часто
# НЕ находит нужный чанк, хотя он есть в базе: например, на «Какие институты?»
# выдавала старые факультеты вместо новых ИИТ/ИИИ/ИКБ/ИРИП из README.
#
# Поэтому после векторного поиска переранжируем кандидатов с учётом
# ЛЕКСИЧЕСКОГО совпадения (пересечение слов вопроса и текста). Это заметно
# поднимает точные термины и названия, которые векторная модель «не видит».

# Слова, которые не несут смысла и только мешают совпадению.
_STOPWORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за",
    "бы", "по", "только", "ее", "мне", "было", "вот", "от", "меня", "еще",
    "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну", "вдруг", "ли",
    "если", "уже", "или", "ни", "быть", "был", "него", "до", "вас", "нибудь",
    "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя",
    "их", "чем", "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже",
    "себе", "под", "будет", "ж", "тогда", "кто", "этот", "того", "потому",
    "этого", "какой", "совсем", "ним", "здесь", "этом", "один", "почти",
    "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем", "всех",
    "никогда", "можно", "при", "наконец", "два", "об", "другой", "хоть",
    "после", "над", "больше", "тот", "через", "эти", "нас", "про", "всего",
    "них", "какая", "много", "разве", "три", "эту", "моя", "впрочем", "хорошо",
    "свою", "этой", "перед", "иногда", "лучше", "чуть", "том", "нельзя",
    "такой", "им", "более", "всегда", "конечно", "всю", "между",
    # вопросительные и служебные «около-темные» слова
    "какие", "какой", "каких", "сколько", "расскажи", "скажи", "подскажи",
    "можно", "нужно", "хочу", "хотел", "бы", "про", "об", "это",
    # «пустые» глаголы-связки — есть в половине предложений базы
    "являет", "называ", "входит", "входят", "включ", "состав", "имеет",
    "предст", "указан", "указан", "основн", "главн", "общий", "общая",
    # указательные и «вопросные» слова: «Что такое СНО?» ищет «сно», а не «что такое»
    "такое", "такая", "такие", "такого", "такому", "таком", "эдак",
    "значит", "означа", "отлич", "отлича", "зовут", "называют",
    "интерес", "интересн", "рассказа", "подробн", "кратко", "пожалуй",
    # «находится / расположен» — есть почти в каждой строке про адреса
    "находи", "найдит", "располож", "расположен", "находит",
}

# Синонимы для слов базы знаний МИРЭА.
#
# Слова обрезаются до 6 букв (см. _tokens), поэтому ключи — обрезанные формы.
# Нужно, потому что в базе одни формулировки («Профсоюзная организация»,
# «Студенческий союз»), а спросить могут другими («профком», «студсовет»).
# Без расширения такие вопросы не находят нужный раздел вообще.
_QUERY_EXPANSIONS = {
    # профсоюз / профком / профбюро / профорганизация
    "профко": ["профсо", "профбю", "профор", "профсоюзна", "объедин"],
    "профсо": ["профко", "профбю", "профор", "объедин"],
    "профор": ["профсо", "профко", "профбю"],
    "профбю": ["профсо", "профко"],
    # Студсоюз / студсовет / студенческий совет
    "студсо": ["sumire", "студенческ", "союз", "самоуправ"],
    "sumire": ["студсо", "союз"],
    "студен": ["студсо", "самоуправ"],
    "самоуп": ["студсо", "студсов", "самоуправ"],
    "студсов": ["студсо", "самоуправ", "общежит"],
    # старосты, волонтёры, отряды, СНО, ССК
    "старос": ["старос"],
    "волонт": ["волонт", "добровол"],
    "отряд": ["отряд", "мсо"],
    "сно": ["студенческ", "научное", "общество"],
    "сск": ["спорти", "клуб", "альянс"],
    "киберк": ["кибер", "спорти"],
    "медиац": ["медиа", "медиац"],
    "студоф": ["студоф", "офис"],
    # ЛАТИНСКАЯ ТРАНСЛИТЕРАЦИЯ ИМЁН ФАЙЛОВ.
    #
    # Таблицы базы залиты из CSV, и имя файла транслитерировано:
    # «9-1_obschezhitiya_shest_korpusov_i_das_2_t1.csv». Русского слова
    # «общежитие» в таком чанке НЕТ вообще — поэтому вопрос «где находятся
    # общежития?» не находил таблицу с АДРЕСАМИ общежитий: поиску было не за
    # что зацепиться. Расширяем слова запроса латинскими вариантами из имён
    # файлов (ключи — после обрезки до 6 букв, см. _tokens).
    "общежи": ["obsche", "obshch", "hostel"],
    "стипен": ["stipend"],
    "бюджет": ["biudzh", "budget"],
    "приём": ["priem"],
    "прием": ["priem"],
    "поступ": ["postupl", "priem"],
    "инстит": ["instit"],
    "стоимо": ["stoimost", "price"],
    "кампус": ["kampus"],
    "филиал": ["filial"],
    "количе": ["kolichestvo"],
    # ректорат и руководство
    "ректор": ["ректор", "кудж"],
    "директ": ["директ", "руковод"],
}


def _tokens(text: str) -> list:
    """
    Разбивает текст на значимые слова (без стоп-слов, в нижнем регистре).

    Русские слова дополнительно «обрезаем» до основы (первые 6 букв), чтобы
    ловить формы: «институты» / «института» / «институт» → «инсти».

    ВАЖНО: стоп-слово проверяется и целиком, и по ОСНОВЕ. Иначе фильтр почти
    бесполезен: «находится» не совпадёт со стоп-словом «находи», потому что до
    обрезки слово длиннее. Именно из-за этого служебные слова («находится»,
    «представляет») забивали ранжирование — на вопрос «где находится Студсоюз?»
    побеждал случайный чанк, где просто есть слово «находится».
    """
    if not text:
        return []
    words = re.findall(r"[а-яёa-z0-9]+", text.lower())
    out = []
    for w in words:
        if len(w) < 3:
            continue
        stem = w[:6]
        if w in _STOPWORDS or stem in _STOPWORDS:
            continue
        out.append(stem)
    return out


def _counter(text: str) -> "Counter":
    """
    Частоты токенов текста (токен → сколько раз встретился).

    Нужно для BM25: важно не просто «есть ли слово», а КАК ЧАСТО. Чанк, где
    «общежитие» встречается 20 раз, почти наверняка про общежития, а чанк с
    одним упоминанием — нет.
    """
    from collections import Counter

    return Counter(_tokens(text))


def _counter_q(text: str) -> "Counter":
    """Частоты токенов ВОПРОСА (с сохранением вопросительных слов)."""
    from collections import Counter

    return Counter(_tokens_for_questions(text))


def _lexical_score(query_terms, content: str, idf: dict = None) -> float:
    """
    Взвешенная доля слов запроса, найденных в тексте (0..1).

    ``query_terms`` — список НАБОРОВ словоформ: для каждого слова вопроса это
    само слово плюс его синонимы (см. ``_QUERY_EXPANSIONS``). Слово считается
    найденным, если в тексте есть ХОТЯ БЫ ОДНА форма из набора.

    Если передан ``idf`` (обратная частота слова по кандидатам), редкие слова
    получают больший вес. Это важно: в базе МИРЭА почти в каждом чанке есть
    «институт» и «МИРЭА», поэтому по ним совпадение ничего не различает, а
    редкое «ректор» или «стипендия» — различает.
    """
    if not query_terms:
        return 0.0
    content_tokens = set(_tokens(content))
    if not content_tokens:
        return 0.0

    total_w = 0.0
    hit_w = 0.0
    for alts in query_terms:
        w = 1.0
        if idf:
            w = max((idf.get(a, 1.0) for a in alts), default=1.0)
        total_w += w
        if content_tokens & alts:
            hit_w += w
    return (hit_w / total_w) if total_w > 0 else 0.0


def _faq_question(content: str) -> str:
    """
    Вытаскивает вопрос из фрагмента формата «В: <вопрос> О: <ответ>».

    Нужно для точного бонуса: раньше ЛЮБОЙ FAQ-фрагмент получал плоскую
    надбавку, и на вопрос «Кто ректор МИРЭА?» побеждал фрагмент «Почему часть
    данных о МИРЭА неофициальная?» — он тоже из FAQ, но к ректору не относится,
    просто содержит слово «МИРЭА». Теперь надбавка пропорциональна совпадению
    ВОПРОСА фрагмента с вопросом гостя.
    """
    if not content:
        return ""
    import re as _re

    # Формат: «### В: … О: …» или «В: … О: …»
    m = _re.search(r"(?:^|\n)\s*(?:#{1,6}\s*)?В:\s*(.+?)\s*(?:\n|О:)", content)
    return m.group(1)[:200] if m else ""


def _faq_questions(content: str):
    """
    Вытаскивает ВСЕ вопросы из фрагмента формата «В: <вопрос> О: <ответ>».

    ВАЖНО: один чанк почти всегда содержит НЕСКОЛЬКО FAQ-записей (чанкование
    идёт по размеру текста, а не по записям). Раньше брался только ПЕРВЫЙ
    вопрос, поэтому чанк с нужным ответом («Сколько бюджетных мест на
    бакалавриат…») оценивался по чужому вопросу и проигрывал случайному
    разделу. Теперь собираем все вопросы чанка и при ранжировании берём
    ЛУЧШЕЕ совпадение.
    """
    if not content:
        return []
    import re as _re

    found = []
    for m in _re.finditer(
        r"(?:^|\n)\s*(?:#{1,6}\s*)?В:\s*([^\n]+)", content
    ):
        q = m.group(1).strip()
        if q:
            found.append(q[:200])
        if len(found) >= 25:
            break
    return found


# Порог, ниже которого FAQ-запись считается нерелевантной и НЕ подставляется
# в контекст (отсекает случайные совпадения по одному общему слову).
# Порог намеренно невысокий: короткий вопрос гостя («Какие есть институты?»)
# физически не может покрыть весь текст длинного вопроса FAQ, но если в нём
# есть КЛЮЧЕВОЙ термин («институты») — это уже попадание.
FAQ_MIN_QLEX = 0.45
# Сколько лучших FAQ-записей подставлять в контекст. Две, а не одна: короткий
# вопрос гостя законно подходит к нескольким записям сразу («Сколько бюджетных
# мест?» — бакалавриат 3 088 и магистратура 1 085), и обе цифры полезны.
FAQ_TOP_N = 2

# Признаки «пустого» ответа внутри FAQ-записи.
#
# В базе есть записи вида «Кто был ректором МИРЭА до Куджа? — В базе нет
# таких сведений». По словам они идеально совпадают с вопросом «Кто ректор
# МИРЭА?», поэтому раньше такая запись вставала ПЕРВОЙ — и модель начинала
# ответ с «в базе нет сведений», хотя рядом лежала правильная запись.
# Такие записи получают штраф и уходят ниже.
_NEG_ANSWER_MARKERS = (
    "в базе нет", "нет таких сведений", "нет данных", "нет сведений",
    "не назван", "не подтвержд", "неизвестно", "не располагаем",
    "информации нет", "не удалось найти", "сведений нет",
)


def _is_negative_answer(text: str) -> bool:
    """True, если FAQ-запись отвечает «данных нет» — её нельзя ставить первой."""
    low = (text or "").lower()
    return any(m in low for m in _NEG_ANSWER_MARKERS)

# Слова, которые ВАЖНЫ при сопоставлении ВОПРОСА с ВОПРОСОМ, хотя бесполезны
# при сравнении с обычным текстом.
#
# Зачем отдельный набор: «где», «они», «сколько» выброшены из _STOPWORDS,
# потому что в тексте разделов они ничего не значат. Но при подборе FAQ-записи
# они — главный различающий признак: «Сколько ОБЩЕЖИТИЙ и ГДЕ ОНИ?» должно
# совпасть с «Сколько общежитий … и где они находятся?», а НЕ с «Сколько СТОИТ
# общежитие?». Без этих слов оба вопроса имели одинаковый вес, и выбирался
# неверный.
_QUESTION_WORDS = {
    "кто", "что", "где", "когда", "сколько", "как", "какой", "какая",
    "какие", "каких", "каком", "почему", "зачем", "куда", "откуда", "чей",
    "чья", "они", "есть", "ли", "можно", "нужно", "сколько-нибудь",
    "находится", "находятся", "сколько", "есть",
    # ОБРЕЗАННЫЕ формы (в _tokens слово режется до 6 букв). Без них проверка
    # `stem in _QUESTION_WORDS` не срабатывала: «сколько» → «скольк», а такого
    # ключа не было, поэтому слово уходило в _STOPWORDS и ВЫБРАСЫВАЛОСЬ.
    # Из-за этого вопросы «Сколько …?» теряли главное слово и матчились не с
    # той записью FAQ («Сколько бюджетных мест?» → отвечала запись про
    # экономику, а не про бакалавриат).
    "скольк", "находи", "найдут", "каков", "какова", "каковы",
}

# Мелкие искажения имён, которые лёгкая модель уверенно выдаёт вслух.
# Только для имён, которые реально звучат в ответе (руководство вуза).
_NAME_FIXES = {
    "санислав": "Станислав",
    "станислова": "Станислава",
    "алексевеч": "Алексеевич",
    "алексеевичь": "Алексеевич",
    "кужд": "Кудж",
    "сигово": "Сигова",
    "сигав": "Сигов",
}


def _tokens_for_questions(text: str) -> list:
    """
    Токены для сопоставления ВОПРОСА с вопросом: вопросительные слова СОХРАНЯЮТСЯ.

    Отличается от ``_tokens`` только тем, что не выкидывает слова из
    ``_QUESTION_WORDS``. Это даёт правильный выбор FAQ-записи.
    """
    if not text:
        return []
    words = re.findall(r"[а-яёa-z0-9]+", text.lower())
    out = []
    for w in words:
        if len(w) < 3:
            continue
        stem = w[:6]
        if stem in _QUESTION_WORDS:
            out.append(stem)
            continue
        if w in _STOPWORDS or stem in _STOPWORDS:
            continue
        out.append(stem)
    return out


def fix_known_names(text: str) -> str:
    """Исправляет частые искажения имён руководства вуза."""
    if not text:
        return text
    out = text
    for bad, good in _NAME_FIXES.items():
        out = re.sub(rf"\b{bad}", good, out, flags=re.IGNORECASE)
    return out


def _faq_entries(content: str):
    """
    Разбивает текст на FAQ-записи: [(вопрос, текст_записи «В: … О: …»)].

    Зачем построчно, а не чанками: чанкование идёт по размеру текста, поэтому
    в ОДНОМ чанке лежат 5–15 разных вопросов. Раньше чанк оценивался по одному
    (первому) вопросу или по всем сразу — и в контекст уходил блок, где нужный
    ответ зарыт среди чужих. Теперь мы можем ВЫРЕЗАТЬ ровно ту запись, чей
    вопрос совпал с вопросом гостя, и поставить её первой в контекст.
    """
    if not content:
        return []
    import re as _re

    marks = []
    for m in _re.finditer(r"(?:^|\n)\s*(?:#{1,6}\s*)?В:\s*([^\n]+)", content):
        # ВАЖНО: в этой базе вопрос и ответ часто лежат НА ОДНОЙ строке:
        #   «### В: Сколько бюджетных мест…? О: В 2026 году … 3 088 мест».
        # Раньше вопросом считалась ВСЯ строка целиком, вместе с ответом. Из-за
        # этого при сопоставлении «вопрос FAQ» содержал все слова ответа, и
        # любая запись с подходящими словами ГДЕ УГОДНО в ответе получала
        # полное покрытие. На «Сколько бюджетных мест?» побеждала запись про
        # магистратуру (цифра «1 085 бюджетных мест» в ответе), а «Какие есть
        # институты?» уходило к «Какие кафедры есть в ИИИ?» (там в ответе есть
        # слово «Институт»).
        # Теперь вопрос обрезается по маркеру ответа «О:».
        q = _re.split(r"\s+О\s*:", m.group(1), maxsplit=1)[0].strip()
        marks.append((m.start(), q))
        if len(marks) >= 30:
            break
    entries = []
    for i, (pos, q) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(content)
        entry = content[pos:end].strip()
        if q and entry:
            entries.append((q, entry))
    return entries


def _extract_heading(content: str) -> str:
    """
    Достаёт содержательный заголовок чанка.

    ВАЖНО: нельзя слепо брать первую строку. У таблиц первая строка — это ШАПКА
    СТОЛБЦОВ («| № | Адрес (2ГИС) | Тип, этажность | Мест | Плата в месяц
    (бюджет / платно) |»), и она случайно содержит слова запроса («мест»,
    «бюджет»). Из-за этого скоринг по заголовку давал 1.0 нерелевантным чанкам,
    все оценки сравнивались, и в ответ уходил мусор.

    Поэтому шапки таблиц как заголовок игнорируем: берём первую строку, которая
    не начинается с «|» и не является ссылкой/служебной пометкой.
    """
    for line in (content or "").split("\n")[:5]:
        line = line.strip()
        if not line or line.startswith("|"):
            continue
        # Строка-разделитель таблицы (|---|---|) и цитаты-ссылки не подходят.
        if set(line) <= set("-|: "):
            continue
        return line[:200]
    return ""


def _bm25_terms(query_terms, tf, n_toks: int, avg_len: float, idf: dict,
                k1: float = 1.5, b: float = 0.6) -> float:
    """
    BM25-подобный скор по набору терминов запроса (нормированный в 0..1).

    Зачем BM25, а не доля совпадений: раньше любой чанк, где встретились все
    слова вопроса, получал ровно 1.0. Таких чанков в базе МИРЭА десятки — все
    сравнивались, и победитель выбирался случайно (выигрывал «институт ИТУ»
    вместо сводки о бюджетных местах).

    BM25 учитывает ЧАСТОТУ слова в чанке и ДЛИНУ чанка: короткий текст, плотно
    посвящённый теме («9.1. Общежития: шесть корпусов и ДАС-2»), обгоняет
    длинный текст, где слово упомянуто мельком в одном абзаце.
    """
    if not query_terms or not tf:
        return 0.0
    norm = 0.0
    total = 0.0
    for alts in query_terms:
        w = max((idf.get(a, 1.0) for a in alts), default=1.0)
        norm += w
        freq = max((tf.get(a, 0) for a in alts), default=0)
        if freq:
            denom = freq + k1 * (1 - b + b * (n_toks / max(1.0, avg_len)))
            total += w * (freq * (k1 + 1.0)) / denom
    if not norm:
        return 0.0
    # Делим ещё на (k1+1), чтобы результат был <= 1 и был сопоставим с весами.
    return total / (norm * (k1 + 1.0))


def _lexical_score_terms(query_terms, content_tokens: set, idf: dict = None) -> float:
    """
    Вариант ``_lexical_score`` для УЖЕ готового набора токенов текста.

    Нужен для снимка коллекции: там токены каждого чанка посчитаны заранее, и
    повторно токенизировать сотни чанков на каждый вопрос не надо.
    """
    if not query_terms or not content_tokens:
        return 0.0
    total_w = 0.0
    hit_w = 0.0
    for alts in query_terms:
        w = max((idf.get(a, 1.0) for a in alts), default=1.0) if idf else 1.0
        total_w += w
        if content_tokens & alts:
            hit_w += w
    return (hit_w / total_w) if total_w > 0 else 0.0


def _key_term(query_terms, idf: dict, threshold: float = None):
    """
    Возвращает набор словоформ САМОГО РЕДКОГО слова вопроса — «ключевого».

    Зачем: в базе МИРЭА почти в каждом чанке есть «МИРЭА» и «институт». На
    вопросе «Кто ректор МИРЭА?» доля совпадений у всех чанков почти одинаковая
    (lex = 1.0 у десятков чанков), и победитель выбирается почти случайно.
    А вот «ректор» встречается редко, и чанк с ним — почти наверняка ответ.

    Берём слово с МАКСИМАЛЬНЫМ IDF из запроса. Порог (если задан) лишь отсекает
    случай, когда в вопросе одни общеупотребительные слова.

    ВАЖНО: сравнение с процентилем IDF по всей коллекции НЕ подходит — в базе
    полно уникальных слов (номера источников, редкие термины), и 75-й процентиль
    оказывается очень высоким (~6.8), так что «ректор» (3.5) до него не дотягивал
    и правило не срабатывало вообще.
    """
    if not query_terms or not idf:
        return set()
    best, best_w = set(), 0.0
    for alts in query_terms:
        w = max((idf.get(a, 1.0) for a in alts), default=1.0)
        if w > best_w:
            best, best_w = alts, w
    if threshold is not None and best_w < threshold:
        return set()
    return best


def _idf_median_high(idf: dict, q: float = 0.10) -> float:
    """
    Порог «редкости» — 10-й процентиль IDF по коллекции.

    Низкая квантиль (не 75-я!) выбрана специально: нужен фильтр только против
    совсем общеупотребительных слов, а не «отбор самых уникальных». Так редкие
    предметные слова вопроса («ректор», «стипендия») точно проходят порог.
    """
    if not idf:
        return 0.0
    vals = sorted(idf.values())
    idx = min(len(vals) - 1, int(len(vals) * q))
    return vals[idx]


def _query_terms(query_tokens) -> list:
    """
    Превращает слова запроса в наборы словоформ (слово + синонимы).

    Пример: «профком» (основа «профко») → {"профко", "профсо", "профор", …},
    поэтому находится раздел про «Профсоюзную организацию».
    """
    terms = []
    for t in query_tokens:
        alts = {t}
        for extra in _QUERY_EXPANSIONS.get(t, ()):
            alts.add(extra)
        terms.append(alts)
    return terms


def _compute_idf(token_lists) -> dict:
    """
    Считает IDF по набору документов: 1 + log(N / df).

    Чем в большем числе чанков встречается слово, тем меньше его вес.
    """
    import math

    n = max(1, len(token_lists))
    df: dict = {}
    for toks in token_lists:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    return {t: 1.0 + math.log(n / c) for t, c in df.items()}


# ---------------------------------------------------------------------------
# Снимок коллекции в памяти (сильно ускоряет поиск)
# ---------------------------------------------------------------------------
# Раньше КАЖДЫЙ поиск заново: выгружал все чанки с векторами из Qdrant (сотни
# КБ), заново токенизировал их для IDF и считал близость по одному вектору.
# На один вопрос это десятки-сотни миллисекунд, а в диалоге копится.
#
# База вуза меняется только при заливке, поэтому держим снимок в памяти
# процесса: вектора как одну numpy-матрицу + заранее посчитанные токены и IDF.
# Тогда поиск = эмбеддинг вопроса + одно умножение матриц.
_SNAPSHOT_CACHE: Dict[str, Dict[str, Any]] = {}
_SNAPSHOT_LOCK = __import__("threading").Lock()
_SNAPSHOT_TTL_SEC = 600.0          # пересобирать не чаще, чем раз в 10 минут
_SNAPSHOT_MAX = 4                  # сколько коллекций держим одновременно


def invalidate_snapshot(collection_id: str = None) -> None:
    """
    Сбрасывает кэш снимка (после заливки/удаления документов).

    Без этого поиск отдавал бы старые данные до истечения TTL.
    """
    with _SNAPSHOT_LOCK:
        if collection_id is None:
            _SNAPSHOT_CACHE.clear()
        else:
            _SNAPSHOT_CACHE.pop(str(collection_id), None)


def _build_snapshot(qdrant_service: "QdrantService", collection_id: str):
    """
    Выгружает коллекцию целиком и готовит её к быстрому поиску.

    Возвращает dict с элементами (payload + токены + вектор) и numpy-матрицей
    векторов, либо ``None``, если коллекция недоступна/пуста.
    """
    import time as _time

    points = qdrant_service.scroll_all(
        collection_id, max_points=5000, with_vector=True
    )
    if not points:
        return None

    items = []
    vectors = []
    for p in points:
        payload = p.get("payload") or {}
        content = payload.get("content", "") or ""
        title = payload.get("document_title", "") or ""
        item = {
            "id": p.get("id"),
            "payload": payload,
            "content": content,
            "title": title,
            "heading": _extract_heading(content),
            # Токены считаем ОДИН раз — они нужны и для IDF, и для скоринга.
            # Counter (token → частота) — BM25 нужна именно частота.
            "toks": _counter(content),
        }
        item["head_toks"] = _counter(item["heading"])
        item["title_toks"] = _counter(title)
        item["n_toks"] = sum(item["toks"].values()) or 1
        # FAQ-записи чанка (вопрос + текст ответа) — нужны, чтобы вырезать
        # и подставить в контекст РОВНО ту запись, чей вопрос совпал.
        item["entries"] = [(_counter_q(q), txt) for q, txt in _faq_entries(content)]
        items.append(item)
        vectors.append(p.get("vector"))

    avg_len = (sum(it["n_toks"] for it in items) / len(items)) if items else 1.0

    matrix = None
    try:
        import numpy as np

        if all(v is not None for v in vectors):
            m = np.asarray(vectors, dtype="float32")
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = m / norms
    except Exception as e:  # noqa: BLE001
        print(f"Снимок: не удалось собрать матрицу векторов: {e}")

    # IDF считаем ОДИН раз по всей коллекции — это основа лексического скоринга.
    idf = _compute_idf([it["toks"] for it in items])

    # ОТДЕЛЬНЫЙ IDF ПО ВОПРОСАМ FAQ.
    # IDF по контенту для выбора ключевого слова вопроса не годится: в самих
    # документах слово «какие» почти не встречается, поэтому его IDF огромный,
    # и ключевым термином запроса «Какие есть ИНСТИТУТЫ?» становилось «какие»,
    # а не «институты». В пространстве же FAQ-вопросов «какие»/«есть» частые
    # (низкий IDF), а «институты» — редкое (высокий IDF), как и должно быть.
    faq_token_lists = [qc for it in items for qc, _ in it["entries"]]
    faq_idf = _compute_idf(faq_token_lists) if faq_token_lists else idf

    return {
        "items": items,
        "matrix": matrix,
        "idf": idf,
        "faq_idf": faq_idf,
        "avg_len": avg_len,
        "ts": _time.time(),
    }


_SNAPSHOT_REFRESHING: set = set()   # коллекции, снимок которых уже пересобирается


def _store_snapshot(key: str, snap) -> None:
    with _SNAPSHOT_LOCK:
        _SNAPSHOT_CACHE[key] = snap
        # Ограничиваем число коллекций в памяти (самые старые — вон).
        while len(_SNAPSHOT_CACHE) > _SNAPSHOT_MAX:
            oldest = min(_SNAPSHOT_CACHE, key=lambda k: _SNAPSHOT_CACHE[k]["ts"])
            _SNAPSHOT_CACHE.pop(oldest, None)


def _refresh_snapshot_bg(qdrant_service: "QdrantService", collection_id: str) -> None:
    """Пересобирает снимок в фоне; пока он собирается, гости получают старый."""
    key = str(collection_id)
    with _SNAPSHOT_LOCK:
        if key in _SNAPSHOT_REFRESHING:
            return
        _SNAPSHOT_REFRESHING.add(key)

    def _work():
        try:
            snap = _build_snapshot(qdrant_service, collection_id)
            if snap:
                _store_snapshot(key, snap)
        except Exception as exc:  # noqa: BLE001
            print(f"Фоновое обновление снимка не удалось: {exc}")
        finally:
            with _SNAPSHOT_LOCK:
                _SNAPSHOT_REFRESHING.discard(key)

    __import__("threading").Thread(target=_work, daemon=True).start()


def _get_snapshot(qdrant_service: "QdrantService", collection_id: str):
    """
    Возвращает снимок коллекции из кэша.

    Stale-while-revalidate: если снимок старше TTL, гостю СРАЗУ отдаётся старый,
    а новый собирается в фоне. Раньше вопрос, попавший на истёкший TTL, ждал
    полную пересборку (выгрузка всех векторов из Qdrant) — это несколько секунд
    тишины раз в 10 минут. Синхронно собираем только когда снимка нет вовсе.
    """
    import time as _time

    key = str(collection_id)
    with _SNAPSHOT_LOCK:
        snap = _SNAPSHOT_CACHE.get(key)
    if snap:
        if (_time.time() - snap["ts"]) >= _SNAPSHOT_TTL_SEC:
            _refresh_snapshot_bg(qdrant_service, collection_id)
        return snap
    # Снимка нет — собираем ВНЕ блокировки (сетевой запрос, лок держать нельзя).
    snap = _build_snapshot(qdrant_service, collection_id)
    if snap:
        _store_snapshot(key, snap)
    return snap


_RERANK_DOWN_UNTIL = 0.0


def _maybe_rerank(query, scored, limit):
    """
    Переранжирование лучших кандидатов cross-encoder'ом (сервис rerank_service на GPU).

    ``scored`` — [(оценка, item)] по убыванию гибридной оценки. Берём первых N, просим сервис оценить
    пары «вопрос — фрагмент» и смешиваем: итог = alpha * оценка_реранкера + (1 - alpha) * гибридная
    (нормированная). Сервис недоступен — молча возвращаем исходный порядок (30 с не стучимся).
    Отключить: RAG_RERANK=0. Подбор alpha: RERANK_ALPHA (по умолчанию 0.7), см. eval_retrieval.py.
    """
    global _RERANK_DOWN_UNTIL
    import time as _t

    if os.getenv("RAG_RERANK", "1") == "0" or _t.time() < _RERANK_DOWN_UNTIL or len(scored) < 2:
        return scored
    n = min(len(scored), max(limit * 3, int(os.getenv("RERANK_TOP", "10"))))
    head, tail = scored[:n], scored[n:]
    try:
        import requests as _rq

        url = os.getenv("RERANK_URL", "http://172.17.0.1:8152")
        r = _rq.post(f"{url}/rerank", json={"query": query, "passages": [it["content"] for _, it in head]},
                     timeout=float(os.getenv("RERANK_TIMEOUT", "4")))
        rr = r.json()["scores"]
    except Exception:  # noqa: BLE001
        _RERANK_DOWN_UNTIL = _t.time() + 30
        return scored
    alpha = float(os.getenv("RERANK_ALPHA", "0.7"))
    hs = [s for s, _ in head]
    lo, hi = min(hs), max(hs)
    span = (hi - lo) or 1.0
    mixed = [(alpha * rr[i] + (1 - alpha) * (hs[i] - lo) / span, head[i][1]) for i in range(len(head))]
    mixed.sort(key=lambda x: x[0], reverse=True)
    # возвращаем в шкале исходных оценок (дальше код использует score только для вывода)
    return [(m, it) for m, it in mixed] + tail

class DocumentService:
    """Service for processing and managing documents"""
    
    def __init__(self):
        """Initialize document service"""
        self.upload_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                       'data', 'uploads')
        os.makedirs(self.upload_dir, exist_ok=True)
        self.qdrant_service = QdrantService()
        
        # Initialize docling document converter with allowed formats
        self.converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.DOCX,
                InputFormat.PPTX,
                InputFormat.XLSX,
                InputFormat.HTML,
                InputFormat.MD,
                InputFormat.ASCIIDOC,
                InputFormat.CSV,
                InputFormat.XML_JATS,
                InputFormat.IMAGE,
            ]
        )
        
        # Extension to format mapping for supported types
        self.extension_map = {
            # PDF files
            '.pdf': InputFormat.PDF,
            
            # Office documents
            '.doc': InputFormat.DOCX,
            '.docx': InputFormat.DOCX,
            '.ppt': InputFormat.PPTX, 
            '.pptx': InputFormat.PPTX,
            '.xls': InputFormat.XLSX,
            '.xlsx': InputFormat.XLSX,
            
            # Web/markup formats
            '.html': InputFormat.HTML,
            '.htm': InputFormat.HTML,
            '.md': InputFormat.MD,
            '.adoc': InputFormat.ASCIIDOC,
            
            # Data formats
            '.csv': InputFormat.CSV,
            '.xml': InputFormat.XML_JATS,
            
            # Image files (with OCR)
            '.jpg': InputFormat.IMAGE,
            '.jpeg': InputFormat.IMAGE,
            '.png': InputFormat.IMAGE,
            '.tif': InputFormat.IMAGE,
            '.tiff': InputFormat.IMAGE,
            
            # Archive formats - special handling
            '.zip': 'ZIP_ARCHIVE',  # Not an InputFormat, handled specially
        }
        
    def save_uploaded_file(self, file):
        """
        Save uploaded file to disk
        
        Args:
            file: The uploaded file object
            
        Returns:
            str: Path to the saved file
        """
        filename = str(uuid.uuid4()) + "_" + file.filename
        file_path = os.path.join(self.upload_dir, filename)
        file.save(file_path)
        return file_path
        
    def process_zip_file(self, zip_file_path):
        """
        Extract files from a ZIP archive and return paths to extracted files
        
        Args:
            zip_file_path: Path to the ZIP file
            
        Returns:
            List[Tuple[str, FileStorage]]: List of (extracted file path, file object) tuples
        """
        extracted_files = []
        extract_dir = os.path.join(self.upload_dir, str(uuid.uuid4()) + "_extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # Get list of files in the zip
                file_list = [f for f in zip_ref.namelist() if not f.endswith('/')]
                
                # Extract each file
                for file_name in file_list:
                    # Extract the file
                    extracted_path = os.path.join(extract_dir, os.path.basename(file_name))
                    
                    # Use a unique filename to avoid collisions
                    unique_filename = str(uuid.uuid4()) + "_" + os.path.basename(file_name)
                    extracted_path = os.path.join(extract_dir, unique_filename)
                    
                    # Extract the file
                    with zip_ref.open(file_name) as source, open(extracted_path, 'wb') as target:
                        shutil.copyfileobj(source, target)
                    
                    # Create a FileStorage-like object for compatibility with the rest of the code
                    file_storage = FileStorage(
                        stream=open(extracted_path, 'rb'),
                        filename=os.path.basename(file_name),
                        name=os.path.basename(file_name)
                    )
                    
                    extracted_files.append((extracted_path, file_storage))
                    
            return extracted_files
            
        except Exception as e:
            logger.error(f"Error extracting ZIP file: {str(e)}")
            # Clean up the extraction directory if there was an error
            shutil.rmtree(extract_dir, ignore_errors=True)
            return []
    
    def extract_text(self, file_path, use_docling=False):
        """
        Extract text from various file formats, with option to use docling or fallback methods
        
        Args:
            file_path (str): Path to the file
            use_docling (bool): Whether to use Docling (True) or fallback methods (False)
            
        Returns:
            str: Extracted text content
        """
        # Determine file type based on extension
        file_extension = os.path.splitext(file_path)[1].lower()
        mime_type, _ = mimetypes.guess_type(file_path)
        
        # Try using docling for processing if enabled
        if use_docling and file_extension in self.extension_map:
            logger.info(f"Using Docling to process {file_extension} file")
            
            # Create a temporary working directory for Docling
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Create a copy of the file in the temp directory to avoid permission issues
                    temp_file = os.path.join(temp_dir, os.path.basename(file_path))
                    shutil.copy2(file_path, temp_file)
                    
                    # Convert file path to Path object
                    path_obj = Path(temp_file)
                    
                    # Use docling to process document
                    result = self.converter.convert(source=path_obj)
                    
                    # Extract text - based on the example, we need to access document
                    if result and hasattr(result, 'document'):
                        # Get text using markdown export (ensures consistent formatting)
                        text = result.document.export_to_markdown()
                        
                        if text and text.strip():
                            logger.info(f"Successfully extracted text with Docling (text length: {len(text)})")
                            return text
                        
                        # Try getting text from raw content if markdown export is empty
                        if not text.strip() and hasattr(result.document, 'content'):
                            text = result.document.content
                            if text.strip():
                                logger.info(f"Successfully extracted raw content with Docling")
                                return text
                            
                        logger.warning(f"Docling returned empty text for {file_path}")
                    else:
                        logger.warning(f"Docling conversion failed to return document for {file_path}")
                        logger.debug(f"Result properties: {dir(result)}")
                except Exception as e:
                    logger.error(f"Docling conversion failed for {file_path}: {str(e)}")
        
        # If docling is disabled, fails, or not applicable, use fallback methods
        logger.info(f"Using fallback extraction for {file_path}")
        
        # Specialized handling based on file type
        
        # Handle Office documents
        if file_extension in ['.docx', '.doc']:
            try:
                import docx
                doc = docx.Document(file_path)
                text = "\n\n".join([paragraph.text for paragraph in doc.paragraphs])
                return text
            except Exception as e:
                logger.error(f"Error extracting Word document: {str(e)}")
                
        elif file_extension in ['.pptx', '.ppt']:
            try:
                from pptx import Presentation
                presentation = Presentation(file_path)
                text = ""
                for slide in presentation.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            text += shape.text + "\n"
                    text += "\n---\n"
                return text
            except Exception as e:
                logger.error(f"Error extracting PowerPoint: {str(e)}")
                
        elif file_extension in ['.xlsx', '.xls']:
            try:
                import pandas as pd
                sheets = pd.read_excel(file_path, sheet_name=None)
                text = ""
                for sheet_name, df in sheets.items():
                    text += f"Sheet: {sheet_name}\n"
                    text += df.to_string(index=False) + "\n\n"
                return text
            except Exception as e:
                logger.error(f"Error extracting Excel: {str(e)}")
        
        # Handle PDF files
        if file_extension == '.pdf' or mime_type == 'application/pdf':
            try:
                # Extract text from PDF
                reader = PdfReader(file_path)
                text = ""
                for i, page in enumerate(reader.pages):
                    text += f"--- Page {i+1} ---\n"
                    text += page.extract_text() + "\n\n"
                return text
            except Exception as e:
                logger.error(f"Error extracting PDF: {str(e)}")
            
        # Handle HTML files
        elif file_extension in ['.html', '.htm'] or mime_type == 'text/html':
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                soup = BeautifulSoup(content, 'html.parser')
                return soup.get_text()
            except Exception as e:
                logger.error(f"Error extracting HTML: {str(e)}")
            
        # Handle Markdown files
        elif file_extension == '.md' or mime_type == 'text/markdown':
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                html = markdown.markdown(content)
                soup = BeautifulSoup(html, 'html.parser')
                return soup.get_text()
            except Exception as e:
                logger.error(f"Error extracting Markdown: {str(e)}")
            
        # Handle text files
        elif file_extension in ['.txt', '.text'] or (mime_type and mime_type.startswith('text/')):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error extracting text file: {str(e)}")
                
        # Attempt to read as raw text as last resort
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read file as text: {str(e)}")
            return f"Could not extract text from {os.path.basename(file_path)} (format: {file_extension})"
    
    def chunk_text(self, text, chunk_size=1000, overlap=200, strategy="sentence"):
        """
        Split text into overlapping chunks using the specified strategy
        
        Args:
            text (str): Text to chunk
            chunk_size (int, optional): Size of each chunk in characters
            overlap (int, optional): Overlap between chunks in characters
            strategy (str, optional): Chunking strategy ('sentence', 'paragraph', 'fixed')
            
        Returns:
            list: List of text chunks
        """
        if not text:
            return []
        
        # Clean text (normalize whitespace but preserve paragraph breaks)
        if strategy == "paragraph":
            # Preserve paragraph breaks
            paragraphs = re.split(r'\n\s*\n', text)
            # Clean each paragraph
            paragraphs = [re.sub(r'\s+', ' ', p).strip() for p in paragraphs]
            # Filter out empty paragraphs
            paragraphs = [p for p in paragraphs if p]
        else:
            # For other strategies, normalize all whitespace
            text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = []
        
        # Choose chunking strategy
        if strategy == "paragraph":
            chunks = self._chunk_by_paragraph(paragraphs, chunk_size, overlap)
        elif strategy == "sentence":
            chunks = self._chunk_by_sentence(text, chunk_size, overlap)
        elif strategy == "fixed":
            chunks = self._chunk_fixed_size(text, chunk_size, overlap)
        else:
            # Default to sentence-based chunking
            chunks = self._chunk_by_sentence(text, chunk_size, overlap)
        
        return chunks
    
    def _chunk_by_sentence(self, text, chunk_size, overlap):
        """Helper method for sentence-based chunking"""
        # Split text into sentences using improved regex
        # This matches sentence endings with punctuation followed by space or end of string
        sentences = re.split(r'(?<=[.!?])\s+|(?<=[.!?])$', text)
        sentences = [s for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence doesn't exceed chunk size, add it
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                # Save current chunk if it's not empty
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap if possible
                if overlap > 0 and current_chunk:
                    # Calculate how many characters to include in overlap
                    # Try to end at word boundaries
                    overlap_chars = min(overlap, len(current_chunk))
                    overlap_text = current_chunk[-overlap_chars:].lstrip()
                    
                    # If overlap doesn't start at a word boundary, adjust it
                    if overlap_chars > 0 and current_chunk[-overlap_chars] != ' ' and len(current_chunk) > overlap_chars:
                        # Find the nearest word boundary
                        space_pos = current_chunk[:-overlap_chars].rfind(' ')
                        if space_pos >= 0:
                            overlap_text = current_chunk[space_pos+1:]
                    
                    current_chunk = overlap_text + " " + sentence + " "
                else:
                    current_chunk = sentence + " "
        
        # Add the last chunk if it's not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def _chunk_by_paragraph(self, paragraphs, chunk_size, overlap):
        """Helper method for paragraph-based chunking"""
        chunks = []
        current_chunk = ""
        current_paragraphs = []
        
        for paragraph in paragraphs:
            # If adding this paragraph doesn't exceed chunk size, add it
            if len(current_chunk) + len(paragraph) + 2 <= chunk_size:  # +2 for \n\n
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                    current_paragraphs.append(paragraph)
                else:
                    current_chunk = paragraph
                    current_paragraphs.append(paragraph)
            else:
                # Save current chunk if it's not empty
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Start new chunk with overlap if possible
                if overlap > 0 and current_paragraphs:
                    # Include previous paragraphs that fit within overlap
                    overlap_text = ""
                    overlap_paragraphs = []
                    
                    # Start from the end and add paragraphs until we exceed overlap
                    for p in reversed(current_paragraphs):
                        if len(p) + 2 + len(overlap_text) <= overlap:  # +2 for \n\n
                            if overlap_text:
                                overlap_text = p + "\n\n" + overlap_text
                            else:
                                overlap_text = p
                            overlap_paragraphs.insert(0, p)
                        else:
                            break
                    
                    current_chunk = overlap_text
                    if overlap_text:
                        current_chunk += "\n\n"
                    current_chunk += paragraph
                    current_paragraphs = overlap_paragraphs + [paragraph]
                else:
                    current_chunk = paragraph
                    current_paragraphs = [paragraph]
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks
    
    def _chunk_fixed_size(self, text, chunk_size, overlap):
        """Helper method for fixed-size chunking"""
        chunks = []
        
        # Simple character-based chunking with fixed size
        for i in range(0, len(text), chunk_size - overlap):
            if i > 0:
                # Start 'overlap' characters before to create overlap
                i = max(0, i - overlap)
            
            # Get chunk of appropriate size
            end = min(i + chunk_size, len(text))
            chunk = text[i:end]
            
            # Try to break at word boundaries if possible
            if end < len(text) and text[end] != ' ' and ' ' in chunk:
                # Find last space in chunk
                last_space = chunk.rfind(' ')
                if last_space > 0:
                    chunk = chunk[:last_space]
                    i = i + last_space
            
            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def extract_enhanced_metadata(self, text, file_path):
        """
        Extract enhanced metadata from the document text
        
        Args:
            text (str): Document text
            file_path (str): Path to the document file
            
        Returns:
            dict: Enhanced metadata
        """
        metadata = {}
        
        # Basic document stats
        metadata['char_count'] = len(text)
        metadata['word_count'] = len(text.split())
        metadata['line_count'] = text.count('\n') + 1
        
        # Try to detect language (simple heuristic)
        # For a production system, use a proper language detector library
        common_english_words = {'the', 'and', 'of', 'to', 'in', 'a', 'is', 'that', 'for', 'it'}
        common_spanish_words = {'el', 'la', 'de', 'y', 'en', 'un', 'una', 'es', 'que', 'para'}
        common_french_words = {'le', 'la', 'les', 'de', 'et', 'un', 'une', 'est', 'que', 'pour'}
        
        words = set(word.lower() for word in re.findall(r'\b\w+\b', text.lower()))
        english_matches = len(words.intersection(common_english_words))
        spanish_matches = len(words.intersection(common_spanish_words))
        french_matches = len(words.intersection(common_french_words))
        
        if english_matches > spanish_matches and english_matches > french_matches:
            metadata['language'] = 'english'
        elif spanish_matches > english_matches and spanish_matches > french_matches:
            metadata['language'] = 'spanish'
        elif french_matches > english_matches and french_matches > spanish_matches:
            metadata['language'] = 'french'
        else:
            metadata['language'] = 'unknown'
        
        # Try to extract title
        lines = text.split('\n')
        if lines and lines[0].strip():
            # Assume first non-empty line might be a title if it's reasonably short
            first_line = lines[0].strip()
            if len(first_line) <= 200 and not first_line.endswith(('.', '?', '!')):
                metadata['potential_title'] = first_line
        
        # Extract potential headers
        headers = []
        for line in lines:
            line = line.strip()
            # Look for short lines that might be headers (not ending with sentence-ending punctuation)
            if line and len(line) <= 100 and not line.endswith(('.', '?', '!')):
                if re.match(r'^[A-Z0-9]', line):  # Starts with uppercase or number
                    headers.append(line)
        
        if headers:
            metadata['potential_headers'] = headers[:10]  # Limit to avoid oversized metadata
        
        # Attempt to detect document type based on content patterns
        if re.search(r'(copyright|all rights reserved|©)', text.lower()):
            metadata['contains_copyright'] = True
        
        if re.search(r'(confidential|internal use only|not for distribution)', text.lower()):
            metadata['confidential'] = True
        
        # Extract dates if present (simple regex for common formats)
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b', text)
        if dates:
            metadata['dates_found'] = dates[:5]  # Limit to first 5 dates
            
        return metadata
        
    def process_document(self, collection_id, file, title, use_docling=False, chunk_size=1000, 
                         chunk_overlap=200, chunk_strategy="sentence", extract_metadata="basic", file_path=None):
        """
        Process an uploaded document - extract text, chunk it, and store in Qdrant
        
        Args:
            collection_id (str): ID of the collection
            file: Uploaded file object
            title (str): Document title
            use_docling (bool): Whether to use Docling for document processing
            chunk_size (int): Size of each chunk in characters
            chunk_overlap (int): Overlap between chunks in characters
            chunk_strategy (str): Chunking strategy to use
            extract_metadata (str): Metadata extraction level (basic or enhanced)
            file_path (str, optional): If provided, use this path instead of saving the file
            
        Returns:
            Document: Created document
        """
        # Save file if not already saved (like from zip extraction)
        if file_path is None:
            file_path = self.save_uploaded_file(file)
        
        # Extract text using the specified processor
        text = self.extract_text(file_path, use_docling=use_docling)
        
        # Prepare base metadata
        metadata = {
            'filename': file.filename,
            'size': os.path.getsize(file_path),
            'chunk_count': 0,
            'processed_with': 'docling' if use_docling else 'fallback',
            'chunking': {
                'strategy': chunk_strategy,
                'size': chunk_size,
                'overlap': chunk_overlap
            }
        }
        
        # Add enhanced metadata if requested
        if extract_metadata == "enhanced":
            enhanced_metadata = self.extract_enhanced_metadata(text, file_path)
            metadata.update({'enhanced': enhanced_metadata})
        
        # Create document
        document = Document(
            collection_id=collection_id,
            title=title,
            content=text[:1000] + "..." if len(text) > 1000 else text,  # Store preview
            file_path=file_path,
            metadata=metadata
        )
        
        # Save document
        document.save()
        
        # Process document for Qdrant - chunk text and store
        chunks = self.chunk_text(text, chunk_size=chunk_size, 
                               overlap=chunk_overlap, strategy=chunk_strategy)
        
        # Get collection to determine embedding model to use
        collection = Collection.get(collection_id)
        
        # Ensure collection exists in Qdrant
        if not self.qdrant_service.collection_exists(collection_id):
            # Collection must exist and have embedding information
            if not collection:
                raise ValueError(f"Collection {collection_id} not found")
                
            if not collection.embedding_provider or not collection.embedding_model:
                raise ValueError(f"Collection {collection_id} does not have embedding model information")
                
            success = self.qdrant_service.create_collection(
                collection_name=collection_id, 
                vector_size=collection.embedding_dimensions,
                provider=collection.embedding_provider,
                model=collection.embedding_model
            )
            
            if not success:
                raise ValueError(f"Failed to create vector collection for {collection_id}")
        
        # Insert chunks into Qdrant
        for i, chunk in enumerate(chunks):
            # We'll use a simple index as reference and let the Qdrant service handle UUID generation
            chunk_metadata = {
                'document_id': document.id,
                'document_title': title,
                'chunk_index': i,
                'original_chunk_id': f"{document.id}_{i}",
                'content': chunk,
                'file_path': document.file_path,
                'chunk_strategy': chunk_strategy,
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_overlap
            }
            
            # Add page number approximation
            if "--- Page " in chunk:
                # Try to extract actual page number from text
                page_match = re.search(r"--- Page (\d+) ---", chunk)
                if page_match:
                    chunk_metadata['page_number'] = int(page_match.group(1))
                else:
                    chunk_metadata['page_number'] = i + 1  # Simple approximation
            else:
                chunk_metadata['page_number'] = i + 1  # Simple approximation
            
            # We'll use a simple reference ID and let Qdrant service generate a proper UUID
            ref_id = f"{document.id}_{i}"
            
            # Use collection's embedding model if defined
            if collection and collection.embedding_provider and collection.embedding_model:
                self.qdrant_service.insert_document(
                    collection_name=collection_id,
                    document_id=ref_id,
                    text=chunk,
                    metadata=chunk_metadata,
                    provider=collection.embedding_provider,
                    model=collection.embedding_model
                )
            else:
                self.qdrant_service.insert_document(
                    collection_name=collection_id,
                    document_id=ref_id,
                    text=chunk,
                    metadata=chunk_metadata
                )
        
        # Update document metadata
        document.metadata['chunk_count'] = len(chunks)
        document.save()

        # Коллекция изменилась — сбрасываем снимок, иначе поиск до 10 минут
        # отдавал бы старые данные (без только что залитых файлов).
        invalidate_snapshot(collection_id)

        return document
    
    def search_documents(self, collection_id, query, limit=5):
        """
        Search for documents in a collection
        
        Args:
            collection_id (str): ID of the collection
            query (str): Search query
            limit (int, optional): Maximum number of results
            
        Returns:
            dict: Search results with text chunks and document info
        """
        if not self.qdrant_service.collection_exists(collection_id):
            return {
                'query': query,
                'results': [],
                'message': 'Collection does not exist'
            }

        # Get collection to determine embedding model to use
        collection = Collection.get(collection_id)

        # Берём готовый снимок коллекции из памяти: вектора, токены и IDF уже
        # посчитаны. Иначе на КАЖДЫЙ вопрос заново качались бы все чанки с
        # векторами (сотни КБ) и пересчитывался IDF — из-за этого поиск «долго
        # идёт». Снимок пересобирается раз в 10 минут или после заливки.
        snapshot = _get_snapshot(self.qdrant_service, collection_id)
        if not snapshot or not snapshot.get("items"):
            return {
                'query': query,
                'results': [],
                'message': 'Collection is empty or unavailable'
            }

        # Эмбеддинг вопроса — один раз (самая дорогая часть поиска).
        query_vector = None
        try:
            if collection and collection.embedding_provider and collection.embedding_model:
                query_vector = self.qdrant_service.get_embedding(
                    query,
                    provider=collection.embedding_provider,
                    model=collection.embedding_model,
                )
            else:
                query_vector = self.qdrant_service.get_embedding(query)
        except Exception as e:
            print(f"Не удалось получить эмбеддинг запроса: {e}")

        items = snapshot["items"]
        matrix = snapshot.get("matrix")
        idf = snapshot["idf"]
        # Векторная близость одним умножением матриц (646 × 384 → микросекунды).
        # Косинус = скалярное произведение нормализованных векторов.
        vec_scores = None
        if query_vector is not None and matrix is not None:
            try:
                import numpy as np

                qv = np.asarray(query_vector, dtype="float32")
                qn = float(np.linalg.norm(qv))
                if qn > 0:
                    vec_scores = matrix @ (qv / qn)
            except Exception as e:  # noqa: BLE001
                print(f"Векторный скоринг не удался: {e}")

        query_tokens = _tokens(query)
        query_terms = _query_terms(query_tokens)
        # Отдельный набор терминов ДЛЯ СОПОСТАВЛЕНИЯ С ВОПРОСАМИ FAQ: здесь
        # сохранены вопросительные слова («где», «они», «сколько»).
        q_terms = _query_terms(_tokens_for_questions(query))
        # Вопрос ли это (есть вопросительное слово или «?») — от этого зависит
        # бонус фрагментам формата «В: … О: …».
        is_question = query.rstrip().endswith("?") or bool(re.match(
            r"(?i)^\s*(кто|что|где|когда|сколько|как|какой|какая|какие|каких|"
            r"почему|зачем|куда|откуда|чей|чья|есть\s+ли|можно\s+ли)",
            query.strip(),
        ))
        # Отдельного бонуса за «самое редкое слово» здесь БОЛЬШЕ НЕТ: в этой базе
        # тематические слова («ректор», «стипендия») встречаются в СОТНЯХ чанков,
        # и бонус за них только шумел — победителями становились случайные
        # FAQ-фрагменты, где слово просто упомянуто.

        # Вес вектора: у английской MiniLM он низкий (на русском она ненадёжна),
        # у многоязычных моделей (multilingual-*, bge-m3, e5) вектору можно доверять больше.
        emb_name = (getattr(collection, "embedding_model", "") or "").lower()
        # Подобрано по eval_retrieval.py на FAQ (139 вопросов): вес 0,2→0,4→0,7 дал top-1 44,6→44,6→48,2 %, MRR 0,477→0,478→0,522.
        vec_weight = 0.70 if any(k in emb_name for k in ("multilingual", "bge-m3", "e5")) else 0.20
        if os.getenv("RAG_VEC_WEIGHT"):          # подбор веса по тестовому набору (eval_retrieval.py)
            vec_weight = float(os.getenv("RAG_VEC_WEIGHT"))

        scored = []
        avg_len = snapshot.get("avg_len") or 1.0
        # Ключевой (самый редкий) термин вопроса — см. _key_term. IDF берём ПО
        # ВОПРОСАМ FAQ (faq_idf), а не по контенту: иначе служебные "какие"/"есть"
        # оказываются "редкими" и перебивают смысловые слова.
        key_terms = _key_term(q_terms, snapshot.get("faq_idf") or idf)
        # Лучшая совпавшая FAQ-запись по всей коллекции — (bm25, текст).
        faq_best = (0.0, "")
        # Все FAQ-записи, прошедшие порог, — чтобы подставить в контекст не одну,
        # а несколько лучших (см. блок инжекта ниже).
        faq_hits = []
        for idx, item in enumerate(items):
            vec = float(vec_scores[idx]) if vec_scores is not None else 0.0

            # BM25 по СОДЕРЖИМОМУ: учитывает частоту слова и длину чанка.
            lex = _bm25_terms(query_terms, item["toks"], item["n_toks"],
                              avg_len, idf)
            head_lex = _bm25_terms(query_terms, item["head_toks"],
                                   max(1, sum(item["head_toks"].values())),
                                   avg_len, idf)
            title_lex = _bm25_terms(query_terms, item["title_toks"],
                                    max(1, sum(item["title_toks"].values())),
                                    avg_len, idf)

            # Веса подобраны по реальным вопросам к базе МИРЭА:
            #  • заголовок раздела («3.1. Ректор», «9.2. Стипендии») — сильный
            #    признак темы чанка;
            #  • BM25 по тексту — основа (плотность темы);
            #  • document_title почти всегда общий («faq», имя файла);
            #  • вектор — лёгкая поправка (на русском ненадёжен).
            score = (vec_weight * vec
                     + 0.90 * lex
                     + 0.10 * title_lex
                     + 0.60 * head_lex)

            # Обрывки-«приложения» — списки источников [И12, И34] и дампы таблиц
            # («| колонка | колонка |»). Они набиты служебными токенами, поэтому
            # по лексике часто обгоняют настоящий раздел. Ответом они быть не
            # могут — слегка штрафуем, чтобы уступали содержательным чанкам.
            if item["heading"].startswith(("Источники", "|", "Приложение")):
                score -= 0.15

            # Кураторские фрагменты (chunks/) — короткие, с заголовками разделов
            # («# 9.2. Стипендии…»), лучший источник для ответа. Раньше бонус был
            # 0.05 — слишком мал, чтобы перебить мусорные таблицы.
            if item["title"].startswith("chunks"):
                score += 0.20

            # Фрагмент в формате «В: <вопрос> О: <ответ>» — курируемая
            # энциклопедия, созданная РОВНО под такие вопросы.
            # Считаем совпадение по каждому вопросу чанка и запоминаем лучшую
            # запись — её потом подставим первой в контекст.
            if is_question and item.get("entries"):
                for qc, etext in item["entries"]:
                    q_lex = _bm25_terms(q_terms, qc,
                                        max(1, sum(qc.values())), avg_len, idf)
                    # --- 1. Покрытие ЗАПРОСА: доля слов гостя, найденных в
                    # вопросе FAQ. Ловит «мы ответили про то же, о чём спросили».
                    matched = sum(1 for alts in q_terms
                                  if any(qc.get(a) for a in alts))
                    coverage = matched / max(1, len(q_terms))

                    # --- 2. СХОДСТВО МНОЖЕСТВ (Jaccard) между словами запроса и
                    # словами вопроса FAQ.
                    #
                    # Почему не просто «доля слов FAQ, найденных в запросе»:
                    # эта доля — 1/длина, и она давала ничьи. На «Кто ректор
                    # МИРЭА?» одинаково хорошо подходили и «Кто сейчас ректор
                    # МИРЭА?», и «Кто был ректором МИРЭА до Куджа?» — вторая
                    # побеждала по случайному совпадению «кудж» и отвечала
                    # «в базе нет сведений», хотя верный ответ был в первой.
                    #
                    # Jaccard = |пересечение| / |объединение| наказывает ЛИШНИЕ
                    # слова. У «Кто сейчас ректор МИРЭА?» объединение меньше,
                    # поэтому она выигрывает у длинной формулировки.
                    faq_words = [t for t in qc if t not in _STOPWORDS]
                    hit_faq = sum(1 for t in faq_words
                                  if any(t in alts for alts in q_terms))
                    union = len(q_terms) + (len(faq_words) - hit_faq)
                    jaccard = matched / max(1, union)
                    cov_faq = jaccard

                    # --- 3. КЛЮЧЕВОЙ ТЕРМИН. Самый редкий (высокий IDF) термин
                    # запроса есть в вопросе FAQ — почти всегда попадание в цель.
                    # Без этого сигнала «Какие есть ИНСТИТУТЫ?» выбирало «Какие
                    # КАФЕДРЫ есть в ИКБ?»: там совпадали «какие» и «есть», а
                    # редкое и главное «институт» игнорировалось.
                    key_hit = 1.0 if (key_terms
                                      and any(qc.get(a) for a in key_terms)) else 0.0

                    q_lex = (0.20 * q_lex
                             + 0.30 * coverage
                             + 0.25 * cov_faq
                             + 0.35 * key_hit)
                    if coverage >= 1.0:
                        q_lex += 0.15
                    # Запись, которая отвечает «данных нет», не должна идти
                    # первой — иначе модель начинает ответ с отказа, хотя
                    # правильный факт лежит в соседней записи.
                    if _is_negative_answer(etext):
                        q_lex -= 0.30
                    score += 0.80 * max(0.0, q_lex)
                    if q_lex >= FAQ_MIN_QLEX:
                        faq_hits.append((q_lex, etext))
                    if q_lex > faq_best[0]:
                        faq_best = (q_lex, etext)

            scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        scored = _maybe_rerank(query, scored, limit)
        results = scored[:limit]

        # Format search results
        formatted_results = []
        for score, item in results:
            payload = item["payload"]
            formatted_results.append({
                'score': score,
                'document_id': payload.get('document_id'),
                'document_title': payload.get('document_title'),
                'content': item["content"],
                'file_path': payload.get('file_path', ''),
                'page_number': payload.get('page_number', 1)
            })

        # Если нашлись FAQ-записи, чей вопрос почти дословно совпал с вопросом
        # гостя, — ставим их ПЕРВЫМИ и отдельными результатами. В базе МИРЭА
        # FAQ — самый точный источник («Сколько бюджетных мест на бакалавриат
        # было в 2026 году?» → конкретные цифры), но он лежит в больших чанках
        # вместе с чужими вопросами, поэтому модель его не использовала и
        # отвечала общими словами или путала соседние темы.
        #
        # Берём НЕ ОДНУ, а до FAQ_TOP_N записей. Причина: короткий вопрос гостя
        # («Сколько бюджетных мест?») законно подходит сразу к нескольким
        # записям — про бакалавриат (3 088) и про магистратуру (1 085). Если
        # подставить только победителя, остальные цифры теряются и ответ
        # выходит однобоким. Две записи покрывают оба случая.
        faq_hits.sort(key=lambda x: x[0], reverse=True)
        if is_question and faq_hits:
            picked, seen = [], set()
            for q_lex, etext in faq_hits:
                # Дедуп: одна и та же запись встречается в нескольких чанках
                # (и в chunks/, и в полном KB), поэтому сравниваем по тексту.
                sig = re.sub(r"\s+", " ", etext)[:180]
                if sig in seen:
                    continue
                seen.add(sig)
                picked.append((q_lex, etext))
                if len(picked) >= FAQ_TOP_N:
                    break
            print(f"⭐ FAQ-записей подставлено: {len(picked)} "
                  f"(лучшая={picked[0][0]:.2f})")
            for rank, (q_lex, etext) in enumerate(reversed(picked)):
                formatted_results.insert(0, {
                    'score': 10.0 + q_lex,   # заведомо первые места
                    'document_id': f'faq-best-{rank}',
                    'document_title': 'faq',
                    'content': etext,
                    'file_path': '',
                    'page_number': 1,
                })
            formatted_results = formatted_results[:limit]

        return {
            'query': query,
            'results': formatted_results
        }