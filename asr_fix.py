"""
Исправление типичных ошибок распознавания речи в доменных словах.

Два слоя:
1. ручной словарь ослышек (пополняется по логу вопросов: valera_memory.jsonl);
2. нечёткое сопоставление: слово, которого нет среди «известных» (база знаний + прошлые ответы),
   но которое очень похоже на доменный термин, заменяется этим термином.

Осторожность важнее охвата: нечёткая замена только для слов от 7 букв и с порогом сходства 0.84,
а обычные слова (встречавшиеся в базе или в ответах) никогда не трогаются.
"""

from __future__ import annotations

import difflib
import glob
import json
import os
import re
from pathlib import Path
from functools import lru_cache
from typing import Dict, Optional, Set

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ручные замены: ослышка → правильно. Пополнять по логу.
MANUAL: Dict[str, str] = {
    # IT
    "гейхаб": "гитхаб", "гидхаб": "гитхаб", "гитхап": "гитхаб", "гидхап": "гитхаб", "гейтхаб": "гитхаб",
    "шидхап": "гитхаб", "хитхаб": "гитхаб", "гитхуб": "гитхаб",
    "алама": "оллама", "олама": "оллама", "оламу": "олламу", "аламу": "олламу",
    "докир": "докер", "дакер": "докер",
    # роботы
    "прошика": "прошивка", "прошики": "прошивки", "прошике": "прошивке", "прошикой": "прошивкой",
    "бобот": "робот", "бабот": "робот", "ребот": "робот", "ропот": "робот",
    "бобота": "робота", "бабота": "робота", "боботу": "роботу", "бабота": "робота", "боботы": "роботы",
    "робототехнеке": "робототехнике", "робатотехника": "робототехника",
    # вуз
    "стутсоюз": "студсоюз", "субсоюз": "студсоюз", "студсоюс": "студсоюз", "стуцсоюз": "студсоюз",
    "мирам": "МИРЭА", "мираж": "МИРЭА", "сумира": "МИРЭА", "мирея": "МИРЭА", "мирэя": "МИРЭА",
    "мираэ": "МИРЭА", "мерэа": "МИРЭА", "мирае": "МИРЭА",
    "магестратура": "магистратура", "магистротура": "магистратура", "бокалавриат": "бакалавриат",
    "аспирантюра": "аспирантура",
}

# Доменные термы для нечёткого сопоставления (в нижнем регистре, начальная форма или частая форма).
DOMAIN_TERMS = [
    "студсоюз", "профсоюз", "профком", "магистратура", "магистратуру", "бакалавриат", "аспирантура",
    "специалитет", "стипендия", "стипендию", "общежитие", "приёмная", "комиссия", "абитуриент",
    "ректор", "институт", "кафедра", "мегалаборатория", "лаборатория", "робототехника", "робототехнику",
    "манипулятор", "гуманоид", "антропоморфный", "мехатроника", "полигон", "хакатон", "стартап",
    "киберспорт", "кибербезопасность", "информационных", "технологий", "радиоэлектроника",
    "гитхаб", "докер", "оллама", "питон", "нейросеть", "программирование", "алгоритм",
    "экскурсия", "волонтёрский", "студенческий", "военный", "учебный", "колледж", "филиал",
]

_known: Optional[Set[str]] = None


def _known_words() -> Set[str]:
    """Слова, которые считаем «нормальными»: база знаний + ответы Валеры из памяти."""
    global _known
    if _known is not None:
        return _known
    words: Set[str] = set()
    for f in glob.glob(os.path.join(PROJECT_DIR, "knowledge_base", "rtu_mirea_llm_kb", "chunks", "*.md")):
        try:
            words.update(re.findall(r"[а-яё]{3,}", open(f, encoding="utf-8").read().lower()))
        except OSError:
            pass
    try:
        with open(os.path.join(PROJECT_DIR, "valera_memory.jsonl"), encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("{"):
                    try:
                        words.update(re.findall(r"[а-яё]{3,}", json.loads(line).get("a", "").lower()))
                    except ValueError:
                        pass
    except OSError:
        pass
    words.update(DOMAIN_TERMS)
    words.update(MANUAL.values())
    _known = words
    return words


@lru_cache(maxsize=1)
def _spoken_abbreviations():
    """Буквенные произношения сокращений из первой колонки глоссария МИРЭА."""
    glossary = Path(PROJECT_DIR, "knowledge_base/rtu_mirea_llm_kb/chunks/03_glossariy_sokrascheniy.md")
    try:
        rows = glossary.read_text(encoding="utf-8")
    except OSError:
        return {}
    letters = dict(zip("абвгдеёжзийклмнопрстуфхцчшщыэюя",
                       "а бэ вэ гэ дэ е ё жэ зэ и й ка эль эм эн о пэ эр эс тэ у эф ха цэ че ша ща ы э ю я".split()))
    aliases = {}
    for cell in re.findall(r"^\| ([^|]+?) \|", rows, re.M):
        for term in cell.split(" / "):
            if not re.fullmatch(r"[А-ЯЁ]{3,6}", term) or len(set(term)) == 1:
                continue
            sounds = [letters[ch] for ch in term.lower() if ch in letters]
            if len(sounds) == len(term):
                aliases[" ".join(sounds)] = term
                aliases["-".join(sounds)] = term
                aliases["".join(sounds)] = term
    return aliases


@lru_cache(maxsize=1)
def _abbreviation_pattern():
    return re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, sorted(_spoken_abbreviations(), key=len, reverse=True))) + r")(?!\w)", re.I)


_MANUAL_RE = re.compile(r"(?i)(?<![а-яёa-z])(" + "|".join(sorted(map(re.escape, MANUAL), key=len, reverse=True)) + r")(?![а-яёa-z])")
_WORD_RE = re.compile(r"[А-Яа-яЁё]{7,}")


def _match_case(src: str, dst: str) -> str:
    return dst[:1].upper() + dst[1:] if src[:1].isupper() and dst.islower() else dst


def fix_asr(text: str) -> str:
    """Применяет ручной словарь, затем осторожную нечёткую коррекцию доменных слов."""
    if not text:
        return text
    out = _MANUAL_RE.sub(lambda m: _match_case(m.group(1), MANUAL[m.group(1).lower()]), text)
    # «Между викендом, робототехникой и IT» — ослышка в сравнении областей.
    out = re.sub(r"(?i)(\bмежду\s+)викендом(?=[,\s]+(?:и\s+)?(?:робототехник|программир|фронтенд|IT\b|ИТ\b))",
                 r"\1бэкендом", out)
    out = re.sub(r"\b(?:goed|go|гоу)[ -]?(один|два|[12])\b",
                 lambda m: "Go1" if m[1].lower() in {"один", "1"} else "Go2", out, flags=re.I)
    aliases = _spoken_abbreviations()
    if aliases:
        out = _abbreviation_pattern().sub(lambda m: aliases[m[0].lower()], out)
    known = _known_words()

    def _fuzzy(m: "re.Match") -> str:
        w = m.group(0)
        low = w.lower()
        if low in known:
            return w
        best = difflib.get_close_matches(low, DOMAIN_TERMS, n=1, cutoff=0.84)
        return _match_case(w, best[0]) if best else w

    return _WORD_RE.sub(_fuzzy, out)
