"""
Долговременная память Валеры: что спрашивали и что он отвечал — между чатами.

Зачем. Серверный чат WebRAgent помнит только текущий разговор, а после минуты
тишины (авто-новый чат) контекст обнуляется. Гость, вернувшийся через пять минут
(«а что ты мне говорил про общежитие?»), получал «не помню». Память решает это:

* каждый обмен «вопрос — ответ» дописывается в ``valera_memory.jsonl``;
* ``recall()`` достаёт из памяти нужные обмены по запросу гостя («что я спрашивал»,
  «о чём говорили», «повтори», «что ты говорил про стипендию»);
* ``followup_base()`` помогает уточняющим вопросам («а сколько стоит?»), опираясь
  на предыдущий обмен.

Файл локальный и в git не попадает (там вопросы гостей) — см. .gitignore.
Только стандартная библиотека.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from valera_dialogue import (
    INSTITUTES, TOUR, GuestFact, Outline, admission_scope, asks_about_fact, asks_example, institute_names, institute_overview, lab_robots,
    asks_next, guest_statement, named_program, names_new_subject, needs_program, ordered_request, search_topic,
)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH = os.getenv("VALERA_MEMORY_PATH", os.path.join(PROJECT_DIR, "valera_memory.jsonl"))
QUESTIONS_LOG = os.path.join(PROJECT_DIR, "questions.log")
MAX_ROWS = int(os.getenv("VALERA_MEMORY_ROWS", "2000"))
# Уточнение считается продолжением, если с прошлого обмена прошло не больше стольки секунд.
FOLLOWUP_WINDOW_SEC = float(os.getenv("VALERA_FOLLOWUP_WINDOW", "600"))

# ---------------------------------------------------------------------------
# Распознавание намерений
# ---------------------------------------------------------------------------
# «Что было раньше / о чём говорили / что я спрашивал / что ты говорил».
RECALL_RE = re.compile(
    r"(?i)(?:"
    r"что\s+(?:я\s+)?(?:тебя\s+|у\s+тебя\s+)?(?:спрашивал[аи]?|говорил[аи]?|просил[аи]?)"
    r"|о\s+чём\s+(?:мы\s+|я\s+)?(?:говорили|разговаривали|шла\s+речь|была\s+речь|спрашивал[аи]?)"
    r"|что\s+ты\s+(?:мне\s+|нам\s+)?(?:говорил|рассказывал|отвечал|сказал|советовал)"
    r"|в\s+прошлый\s+раз|прошл\w+\s+(?:раз|разговор|чат|беседа|вопрос)"
    r"|\bнапомни\b|\bпомнишь\b|\bчто\s+(?:у\s+нас\s+)?было\b"
    r"|\bчто\s+(?:мы|я|ты|вы)\s+(?:\w+\s+){0,2}(?:говорили|обсуждали|спрашивал\w*|говорил\w*|просил\w*|"
    r"рассказывал\w*|отвечал\w*|сказал\w*)"
    r"|\bпро\s+что\s+(?:мы|я)\s+(?:\w+\s+){0,2}(?:говорили|спрашивал\w*|обсуждали)"
    r"|\bо\s+ч[её]м\s+(?:мы|я)\s+(?:\w+\s+){0,2}(?:говорили|спрашивал\w*|обсуждали)"
    r"|\bна\s+чём\s+(?:мы\s+)?остановились|\bмо[йё]\s+(?:предыдущий|последний)\s+вопрос"
    r")"
)
# «Повтори / ещё раз» — вернуть последний ответ.
REPEAT_RE = re.compile(
    r"(?i)(?:(?:ну|а|пожалуйста)[,\s]+)*(?:"
    r"(?:можешь\s+повторить|можете\s+повторить|повтори(?:те)?)"
    r"(?:[,\s]+(?:это|ещ[её]\s+раз|(?:свой\s+)?(?:последний\s+|предыдущий\s+)?ответ|"
    r"(?:то,?\s+)?что\s+(?:ты|вы)\s+сказал(?:и)?))?"
    r"|скажи(?:те)?\s+ещ[её]\s+раз|ещ[её]\s+раз|что\s+ты\s+сказал)"
    r"(?:[,\s]+пожалуйста)?[.!?…]*"
)
# «Подробнее / дальше / продолжай» — продолжить прошлый ответ, не повторяясь.
CONTINUE_RE = re.compile(
    r"(?i)^\s*(?:(?:ну|а|и)[,\s]+)?(?:расскажи(?:те)?\s+)?"
    r"(?:подробнее|подробней|дальше|продолжай\w*|продолжи\w*|"
    r"рассказывай(?:те)?|больше|ещё|еще|что\s+ещё|что\s+еще|"
    r"а\s+ещё|а\s+еще|поясни|уточни|что\s+именно|объясни)\b"
)
# Начало реплики-уточнения, не имеющей собственной темы.
ELLIPSIS_START_RE = re.compile(
    r"(?i)^\s*(?:ну\s+|то\s+есть\s+|а\s+|и\s+)(?:если|как|где|когда|сколько|почему|зачем|какой|какая|какие|"
    r"что|кто|куда|можно|есть|нужно|надо|там|он|она|они|это|тогда)\b"
)
# Междометия и поддакивания — не вопросы и не продолжение темы.
FILLER_RE = re.compile(r"(?i)^\s*(?:ну\s+)?(?:да|нет|ага|угу|ладно|ок|окей|хорошо|понятно|ясно|ну|так|ясно\s+понятно)[\s.!?,]*$")
# Явно новая тема: «расскажи про X», «что такое X», «как работает X».
NEWTOPIC_RE = re.compile(
    r"(?i)^\s*(?:ну\s+|а\s+)?(?:(?:расскажи\w*|покажи\w*|объясни\w*|опиши\w*)\s+(?:мне\s+)?(?:про|о|об)\b"
    r"|что\s+такое\b|кто\s+так(?:ой|ая)\b|что\s+за\b|как\s+работает\b)"
)
PRONOUN_RE = re.compile(r"(?i)\b(?:это|этот|эта|эти|этого|этом|там|туда|оттуда|он|она|они|его|её|ее|их|тот|то)\b")
ARCHIVE_RE = re.compile(r"(?i)\b(?:вчера|в\s+прошлый\s+раз|прошл\w+\s+(?:чат\w*|разговор\w*|бесед\w*))\b")
_FOLLOWUP_PREFIX_RE = re.compile(r"(?i)^[\s,]*(?:(?:а|и|ну)\s+)+")
_BARE_FOLLOWUP_RE = re.compile(r"(?i)где|куда|когда|как|почему|зачем|кто|какой|какая|какие")
_NUMERIC_FOLLOWUP_RE = re.compile(
    r"(?i)сколько(?:\s+(?:стоит|стоят|баллов|мест|лет|времени))?"
    r"(?:\s+(?:для|на|в|у)\s+.+)?"
)
_FOLLOWUP_DETAIL_RE = re.compile(
    r"(?i)(?:для|на|в|у|по|без|с)\s+"
    r"(?!(?:.+\s)?(?:есть|стоит|сколько|нужно|можно|кто|когда|почему|работает|холодно|жарко|тепло)\b)"
    r"\w+(?:\s+\w+){0,4}"
)
_FACETS = (
    (re.compile(r"(?i)\b(?:стоимост\w*|стоит|стоят|цен[ауые]|оплат\w*)\b"), "стоимость"),
    (re.compile(r"(?i)\b(?:(?:проходн\w*\s+)?балл\w*|егэ)\b"), "баллы"),
    (re.compile(r"(?i)\b(?:мест[ао]?|местах|местам)\b"), "места"),
)


@dataclass(frozen=True)
class FollowUp:
    query: str
    topic: str
    mirea: bool
    choices: tuple[str, ...] = ()
    outline: Optional[Outline] = None
    awaiting_program: bool = False
    reply: str = ""

    @property
    def clarification(self) -> str:
        if self.awaiting_program:
            return "Для какой образовательной программы Вас интересуют баллы или стоимость?"
        if self.reply:
            return self.reply
        return "Вы про " + " или ".join(self.choices) + "?" if self.choices else ""


def is_offer_confirmation(text: str) -> bool:
    """Проверяет, может ли реплика быть согласием на предложение."""
    phrase = text.strip(" ,.!?…")
    return bool(
        re.fullmatch(
            r"(?i)(?:(?:валера|лера)[,\s]+)?(?:(?:ну|так|тогда|да)[,\s]+)*"
            r"(?:да|давай(?:те)?|хорошо|о['’]?кей)",
            phrase,
        )
    )


def accepts_offer(text: str, offered_topic: str) -> bool:
    """Распознаёт согласие, если у Валеры есть предложенная тема."""
    return bool(offered_topic and is_offer_confirmation(text))


def needs_topic(text: str, has_own_topic: bool = False) -> bool:
    """Определяет, нужна ли реплике тема из предыдущего разговора."""
    text = text.strip(" ,.!?…")
    detail = _FOLLOWUP_PREFIX_RE.sub("", text)
    if not detail or GREET_RE.fullmatch(text) or THANKS_RE.fullmatch(text) or FILLER_RE.fullmatch(text):
        return False
    if NEWTOPIC_RE.match(text):
        return False
    return bool(
        CONTINUE_RE.fullmatch(text)
        or _BARE_FOLLOWUP_RE.fullmatch(detail)
        or _NUMERIC_FOLLOWUP_RE.fullmatch(detail)
        or (_FOLLOWUP_DETAIL_RE.fullmatch(detail) and (detail != text or not has_own_topic))
        or (PRONOUN_RE.search(detail) and (ELLIPSIS_START_RE.search(text) or not has_own_topic))
        or any(pattern.fullmatch(detail) for pattern, _ in _FACETS)
    )


_STOP = {
    "что", "как", "где", "когда", "кто", "это", "эти", "про", "для", "или", "так", "вот", "ещё", "еще",
    "мне", "тебя", "тебе", "меня", "ты", "вы", "мы", "они", "она", "оно", "был", "была", "было",
    "спрашивал", "спрашивала", "говорил", "говорила", "говорили", "рассказывал", "отвечал", "сказал",
    "раньше", "ранее", "вчера", "прошлый", "раз", "напомни", "помнишь", "ранее", "нам", "наш", "про",
    "разговор", "разговаривали", "речь", "было", "быть", "есть", "уже", "ли", "же", "бы", "не", "на",
    "по", "из", "от", "до", "со", "за", "при", "над", "под", "если", "чтобы", "чем", "тот", "той",
}


def _stems(text: str) -> List[str]:
    """Грубые основы слов (первые 5 букв) — чтобы «общежитие» находило «общежитии»."""
    out = []
    for w in re.findall(r"[а-яёa-z0-9]+", text.lower()):
        if len(w) < 3 or w in _STOP:
            continue
        out.append(w[:5])
    return out


def recall_kind(text: str) -> Optional[str]:
    """'recall' | 'repeat' | None — что просит гость про прошлое."""
    if not text:
        return None
    if REPEAT_RE.fullmatch(text.strip()):
        return "repeat"
    if RECALL_RE.search(text):
        return "recall"
    return None


def _ago(ts: float, now: Optional[float] = None) -> str:
    d = max(0.0, (now or time.time()) - ts)
    if d < 90:
        return "только что"
    if d < 3600:
        return f"{int(d // 60)} мин назад"
    if d < 86400:
        return f"{int(d // 3600)} ч назад"
    days = int(d // 86400)
    return "вчера" if days == 1 else f"{days} дн. назад"


class ConversationMemory:
    """Журнал обменов с поиском. Потокобезопасен; файл — JSON Lines."""

    def __init__(self, path: str = MEMORY_PATH):
        self.path = path
        self._lock = threading.Lock()
        self.rows: List[Dict] = []
        self._load()
        if not self.rows:
            self._import_questions_log()

    # -- хранение ---------------------------------------------------------
    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith("{"):
                        try:
                            self.rows.append(json.loads(line))
                        except ValueError:
                            continue
            self.rows = self.rows[-MAX_ROWS:]
        except OSError:
            self.rows = []

    def _import_questions_log(self) -> None:
        """Первый запуск: подтягиваем прошлые обмены из questions.log."""
        try:
            with open(QUESTIONS_LOG, encoding="utf-8") as fh:
                rows = []
                for line in fh:
                    if not line.startswith("{"):
                        continue
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    a = (r.get("answer") or "").strip()
                    q = (r.get("query_text") or "").strip()
                    if not q or not a or a.startswith("Error") or a.startswith("#"):
                        continue
                    try:
                        ts = time.mktime(time.strptime(r["timestamp"][:19], "%Y-%m-%dT%H:%M:%S"))
                    except (KeyError, ValueError):
                        ts = time.time()
                    rows.append({"ts": ts, "chat": r.get("chat_id"), "q": q, "a": a,
                                 "mirea": bool(r.get("mirea_related"))})
        except OSError:
            return
        if not rows:
            return
        self.rows = rows[-MAX_ROWS:]
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                for r in self.rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def add(self, question: str, answer: str, chat_id: Optional[str] = None,
            mirea: bool = False, eq: Optional[str] = None,
            topic: Optional[str] = None, use_as_topic: bool = True,
            outline: Optional[Outline] = None, fact: Optional[GuestFact] = None) -> None:
        if not question or not answer:
            return
        row = {"ts": time.time(), "chat": chat_id, "q": question.strip()[:400],
               "a": answer.strip()[:1500], "mirea": bool(mirea)}
        if eq and eq.strip() != question.strip():
            row["eq"] = eq.strip()[:300]   # «эффективный» вопрос с учётом контекста
        if topic and topic.strip() != question.strip():
            row["topic"] = topic.strip()[:240]
        if not use_as_topic:
            row["use_as_topic"] = False
        if outline:
            row["outline"] = asdict(outline)
        if fact:
            row["guest_fact"] = asdict(fact)
        with self._lock:
            self.rows.append(row)
            self.rows = self.rows[-MAX_ROWS:]
            try:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            except OSError:
                pass

    # -- продолжение разговора -------------------------------------------
    def last(self) -> Optional[Dict]:
        with self._lock:
            return self.rows[-1] if self.rows else None

    def recent(self, chat_id: Optional[str], limit: int = 10) -> List[Dict]:
        with self._lock:
            return [row for row in self.rows if row.get("chat") == chat_id][-limit:]

    def guest_fact_reply(self, text: str, chat: str) -> tuple[str, Optional[GuestFact]]:
        with self._lock:
            rows = [row for row in self.rows if row.get("chat") == chat]
        last = rows[-1] if rows else {}
        facts = []
        previous = None
        for row in rows:
            # Старые журналы хранят только q/a. Восстанавливаем поправки по словам гостя.
            recorded = row.get("guest_fact")
            previous = GuestFact(**recorded) if recorded else guest_statement(row["q"], previous)
            if previous:
                facts.append(previous)
        if time.time() - last.get("ts", 0) > FOLLOWUP_WINDOW_SEC:
            previous = None
        fact = guest_statement(text, previous)
        if fact:
            if previous and fact.subject == previous.subject and fact.relation == previous.relation:
                return f"Понял, {fact.value}.", fact
            return f"Запомнил, что {fact.statement}.", fact
        for stored in reversed(facts):
            if asks_about_fact(text, stored):
                return f"Вы сказали, что {stored.statement}.", stored
        return "", None

    def current_discussion(self, chat: str) -> Dict:
        rows = self.recent(chat)
        recent = next((r for r in reversed(rows) if r.get("use_as_topic", True) or r.get("guest_fact")), {})
        return recent if time.time() - recent.get("ts", 0) <= FOLLOWUP_WINDOW_SEC else {}

    def discussion_followup(self, text: str, chat: str, extract_items=None, *, mirea=False,
                            pending: Optional[FollowUp] = None) -> Optional[FollowUp]:
        """Выбирает следующий пункт или уточняет текущий до обычной маршрутизации."""
        recent = self.current_discussion(chat)
        saved = recent.get("outline")
        outline = Outline(**saved) if saved else None
        previous_topic = recent.get("topic") or recent.get("q", "")
        previous_answer = recent.get("a", "")
        overview = institute_overview(text)
        general_scope = re.fullmatch(
            r"(?:(?:ну|давай|расскажи(?:те)?)[,\s]+)*(?:для всего|про вс[её]|вс[её]|"
            r"(?:для|по) все[хм](?: (?:институт\w*|программ\w*|направлени\w*))?)"
            r"(?: подряд| в целом)?(?: расскажи(?:те)?)?", text.strip(" ,.!?…"), re.I)
        awaiting_program = pending is not None and pending.awaiting_program
        if general_scope and (awaiting_program or previous_topic == "Институты РТУ МИРЭА"):
            if awaiting_program:
                query = (f"{pending.query} Гостю нужен общий обзор поступления. Раскрой перечисленные вопросы, "
                         "приведи примеры программ и доступные в базе сведения о баллах и стоимости. "
                         "Для чисел укажи программу и год, если они известны.")
            else:
                query = recent.get("eq") or recent["q"]
            return FollowUp(query, "Институты РТУ МИРЭА", True, outline=Outline(tuple(INSTITUTES), mirea=True))
        if pending and not overview:
            return None
        if (re.search(r"(?:про\s+себя|о\s+себе)", text, re.I)
                and re.search(r"\bинститут", text, re.I) and re.search(r"лаборатори", text, re.I)):
            return FollowUp(f"{text} Ответь по порядку: представь Валеру, расскажи об институте "
                            "и объясни на примере, чем занимаются в лаборатории.",
                            "Валера, институт и мегалаборатория", True)
        if overview and not re.search(r"по (?:очереди|порядку)|поочер[её]дно|по шагам", text, re.I):
            query = (f"Обзор институтов РТУ МИРЭА: {', '.join(INSTITUTES)}. "
                     "Назови институты и примеры направлений. По вопросам о поступлении и стоимости "
                     "различай средние баллы и проходные. Приводи цифры с программой и годом из базы. "
                     f"Вопрос гостя: {text}")
            return FollowUp(query, "Институты РТУ МИРЭА", True, outline=Outline(tuple(INSTITUTES), mirea=True))
        offer = re.split(r"(?<=[.!?…])\s+", previous_answer.strip())[-1]
        # «По ней» может ссылаться на лабораторию в предыдущем предложении.
        last_two = " ".join(re.split(r"(?<=[.!?…])\s+", previous_answer.strip())[-2:])
        if ((is_offer_confirmation(text) or asks_next(text)) and offer.endswith("?")
                and re.search(r"экскурси", offer, re.I) and (not outline or tuple(outline.items) != TOUR)
                and re.search(r"лаборатори", last_two, re.I)):
            subject = "Экскурсия по мегалаборатории робототехники"
            return FollowUp(f"{subject}. Кратко представь маршрут, начиная с полигона «Город».",
                            subject, True, outline=Outline(TOUR, mirea=True, subject=subject))
        # В старых записях к фактическому анонсу маршрута мог прикрепиться план институтов.
        if (asks_next(text) and (not outline or tuple(outline.items) != TOUR)
                and re.search(r"экскурси\w*\s+по\s+(?:мега)?лаборатори", previous_answer, re.I)
                and re.search(r"полигон", previous_answer, re.I)):
            outline = Outline(TOUR, mirea=True, subject="Экскурсия по мегалаборатории робототехники")
        if offer.endswith("?") and not re.search(r"следующ|дальше|остальн|продолж", offer, re.I):
            named = institute_names(previous_answer)
            subject = (outline.current if outline else "") or (named[-1] if named else previous_topic)
            if is_offer_confirmation(text) and re.search(r"хотите|рассказать|интересует", offer, re.I) and " или " not in offer:
                if subject:
                    query = f"{search_topic(subject)}. {offer} Гость согласился. Раскрой предложенную тему."
                    if named and re.search(r"программ|направлени", offer, re.I):
                        query += " Назови программы института и поясни, что на них изучают."
                    return FollowUp(query, subject, bool(named or recent.get("mirea")), outline=outline)
            if " или " in offer and re.fullmatch(r"(?i)(?:давай(?:те)?\s+)?(?:про\s+)?[\w -]{2,60}[.!]?", text.strip()):
                words = set(_stems(text)) - {"давай", "расск"}
                choices = [part for part in re.split(r"\s+или\s+", offer, flags=re.I) if words.intersection(_stems(part))]
                if len(choices) == 1:
                    query = f"{search_topic(subject)}. {offer} Гость выбрал: {text}. Раскрой этот вариант с примером."
                    if named and lab_robots(text):
                        query = f"Подбор направления в РТУ МИРЭА. {query} Интересы: робототехника."
                    return FollowUp(query, subject, bool(named or recent.get("mirea")), outline=outline)

        # Слова «собаки» и «модели» относятся к оборудованию только в разговоре о роботах.
        if lab_robots(previous_topic + " " + previous_answer) and re.search(
            r"кажд\w* модел|\bсобак\w*\b|\bGo[12]\b", text, re.I
        ) and not re.search(r"пород|домашн|ветеринар", text, re.I):
            topic = "Роботы лаборатории РТУ МИРЭА"
            if re.search(r"собак|Go[12]", text, re.I):
                topic += ", четвероногие роботы Unitree"
            return FollowUp(f"{topic}. {text} Различай характеристики моделей и наличие в лаборатории. "
                            "Назови только подтверждённые сведения.", topic, True)

        if lab_robots(previous_topic + " " + previous_answer) and re.search(
            r"про что можно пообщаться|про вс[её],? что (?:есть|знаешь)", text, re.I
        ):
            topic = "Роботы и оборудование лаборатории РТУ МИРЭА"
            return FollowUp(f"{topic}. Перечисли темы для разговора с примерами из лаборатории.", topic, True)
        scope = admission_scope(text, previous_topic)
        if scope and not overview:
            program = named_program(text)
            if not program and not institute_names(text):
                program = previous_topic.partition("Программа: ")[2]
            if program:
                scope += f". Программа: {program}"
            return FollowUp(f"{scope}. {text}", scope, True, outline=outline,
                            awaiting_program=needs_program(text) and not program)
        if re.search(r"\b(?:колледж\w*|спо)\b", text, re.I):
            return None

        next_item = asks_next(text) or is_offer_confirmation(text)
        ordered = ordered_request(text)
        example = asks_example(text)
        detail = bool(needs_topic(text, mirea) or re.fullmatch(
            r"(?i)(?:о['’]?кей[,\s]+)?расскажи(?:те)?,? что (?:ты знаешь|вы знаете)[.?!]?", text))
        explicit_institutes = bool(re.search(r"\bинститут\w*", text, re.I))
        tour = bool(re.search(r"экскурси\w*.*лаборатори|по\s+лаборатории.*экскурси", text, re.I))
        institutes = tuple(INSTITUTES)
        if not outline and (next_item or ordered) and (
            len(institute_names(previous_answer)) >= 2 and
            ("институт" in previous_topic.lower() or re.search(r"следующ\w+ институт", previous_answer, re.I))
        ):
            outline = Outline(institutes, mirea=True)
            if outline.upcoming in institute_names(previous_answer):
                outline = outline.after_answer(outline.upcoming, previous_answer)
        start_institutes = explicit_institutes and (ordered or re.fullmatch(
            r"(?i)(?:расскаж\w* (?:про |об? )?)?институт(?:ы|ах|ов)[.!?]*", text.strip()))
        selected = tuple(institute_names(text))
        if outline and not ordered and len(selected) == 1 and selected[0] in outline.items:
            target = selected[0]
            return FollowUp(f"{search_topic(target)}. {text}", target, True, outline=outline)
        if outline and is_offer_confirmation(text) and previous_answer.rstrip().endswith("?"):
            offer = re.split(r"(?<=[.!?…])\s+", previous_answer.strip())[-1]
            if not re.search(r"следующ|дальше|остальн|продолж", offer, re.I):
                offered = institute_names(offer)
                if len(offered) == 1 and offered[0] in outline.items:
                    target = offered[0]
                elif outline.current and re.search(r"подробн|пример|об этом|о н[её]м", offer, re.I):
                    target = outline.current
                else:
                    return None  # новая тема или выбор разбираются обычным обработчиком предложений
                return FollowUp(f"{search_topic(target)}. {offer} Приведи конкретные детали.",
                                target, outline.mirea, outline=outline)
        if ordered and selected:
            outline = Outline(selected, mirea=True)
        elif start_institutes and (not outline or tuple(outline.items) != institutes):
            outline = Outline(institutes, mirea=True)
        elif tour:
            if not outline or tuple(outline.items) != TOUR:
                outline = Outline(TOUR, mirea=True)
        elif ordered and not start_institutes and (not outline or names_new_subject(text)):
            source = text if names_new_subject(text) else previous_answer
            items = extract_items(source, plan=source == text) if extract_items and source else ()
            if items:
                outline = Outline(tuple(items), mirea=mirea if source == text else bool(recent.get("mirea")),
                                  subject=text if source == text else previous_topic)
            elif source == text:
                return None  # явно задана новая тема, разбираем сам вопрос
            elif "институт" in previous_topic.lower() and source != text:
                outline = Outline(institutes, mirea=True)
            else:
                outline = None
        if outline and (ordered or next_item or start_institutes or tour):
            if next_item:
                outline = outline.next_part()
            target = outline.upcoming
            if not target:
                return FollowUp(text, previous_topic, outline.mirea, outline=outline,
                                reply="Мы дошли до конца списка. Какой пункт обсудим подробнее?")
            subject = f" Тема: {outline.subject}." if outline.subject else ""
            return FollowUp(f"{search_topic(target)}.{subject} Расскажи об этом пункте по существу с конкретным примером.",
                            target, outline.mirea, outline=outline)
        if outline and outline.current and (example or detail):
            target = outline.current
            request = "Приведи один конкретный пример." if example else text
            subject = f" Тема: {outline.subject}." if outline.subject else ""
            return FollowUp(f"{search_topic(target)}.{subject} {request}", target, outline.mirea, outline=outline)
        if (ordered or example) and previous_topic:
            request = "Приведи один конкретный пример" if example else "Начни с первого пункта и раскрой его"
            reference = f" Последнее пояснение: {previous_answer[:240]}" if example else ""
            return FollowUp(f"{request} по теме «{previous_topic}».{reference}",
                            previous_topic, bool(recent.get("mirea")))
        return None

    def followup_base(self, text: str, has_own_topic: bool = False, *,
                      current_chat: Optional[str] = None) -> Optional[Dict]:
        """
        Предыдущий обмен, если ``text`` — уточнение («а сколько стоит?», «подробнее»,
        «а там есть метро?», «а какие баллы примерно?»). Иначе None.

        ``has_own_topic`` — в самой реплике есть слово-тема (МИРЭА, общежитие…): тогда
        это скорее новый вопрос, и продолжением он считается только при явных признаках
        (союз «а…», местоимение, «подробнее»).
        """
        if not needs_topic(text, has_own_topic):
            return None
        cutoff = time.time() - FOLLOWUP_WINDOW_SEC
        with self._lock:
            return next((
                row for row in reversed(self.rows)
                if row.get("chat") == current_chat and row.get("use_as_topic", True)
                and row["ts"] >= cutoff
            ), None)

    def resolve_followup(
        self,
        text: str,
        current_chat: str,
        has_own_topic: bool,
        pending: Optional[FollowUp] = None,
        *,
        offered_topic: str = "",
    ) -> Optional[FollowUp]:
        """Подбирает поисковый запрос из текущей темы и новой детали."""
        if accepts_offer(text, offered_topic):
            recent = self.recent(current_chat, limit=1)
            if recent and time.time() - recent[0]["ts"] <= FOLLOWUP_WINDOW_SEC:
                return FollowUp(
                    query=f"Расскажи подробнее по теме «{offered_topic}».",
                    topic=offered_topic,
                    mirea=bool(recent[0].get("mirea")),
                )
        detail = _FOLLOWUP_PREFIX_RE.sub("", text.strip(" ,.!?…"))
        if re.search(r"\b(?:колледж\w*|спо)\b", detail, re.I):
            if re.fullmatch(r"(?:в |про )?(?:колледж\w*|спо)", detail, re.I):
                rows = self.recent(current_chat, limit=1)
                if rows and time.time() - rows[0]["ts"] <= FOLLOWUP_WINDOW_SEC:
                    previous_query = rows[0].get("eq") or rows[0]["q"]
                    facets = [label for pattern, label in _FACETS if pattern.search(previous_query)]
                    if facets:
                        return FollowUp("Колледж РТУ МИРЭА: " + " и ".join(facets), "колледж", True)
            return None
        if pending:
            if pending.awaiting_program and len(detail.split()) <= 12 and not re.search(
                r"[?!]|\b(?:кто|где|почему|расскажи\w*|что такое|я же сказал|все институты)\b", detail, re.I
            ):
                return FollowUp(f"{pending.query} Образовательная программа: {detail}.",
                                f"{pending.topic}. Программа: {detail}", True, outline=pending.outline)
            choice = re.sub(r"(?i)^(?:про|о|об)\s+", "", detail)
            for pattern, label in _FACETS:
                if label in pending.choices and pattern.fullmatch(choice):
                    return FollowUp(f"{label.capitalize()} по теме «{pending.topic}»", pending.topic, pending.mirea)

        previous = self.followup_base(text, has_own_topic, current_chat=current_chat)
        if previous is None:
            return None
        topic = previous.get("topic") or previous["q"].strip(" ,.!?…")

        effective = previous.get("eq") or previous["q"]
        mirea = bool(previous.get("mirea"))
        if detail.lower() == "сколько":
            focused = [label for pattern, label in _FACETS if pattern.search(effective)]
            choices = focused or [label for pattern, label in _FACETS if pattern.search(previous["a"])]
            if len(choices) > 1:
                return FollowUp(text, topic, mirea, tuple(choices))
            if choices:
                return FollowUp(f"{choices[0].capitalize()} по теме «{topic}»", topic, mirea)
        if _FOLLOWUP_DETAIL_RE.fullmatch(detail):
            base = effective.strip(" ,.!?…")
            if re.search(r"[.!?]\s", base) or len(base) + len(detail) > 240:
                base = topic
            query = f"{base} {detail}?"
        elif CONTINUE_RE.fullmatch(detail):
            query = topic
        else:
            query = f"{topic}. {detail[:1].upper()}{detail[1:]}?"
        return FollowUp(query, topic, mirea)

    # -- «что было раньше» -------------------------------------------------
    def recall(self, text: str, current_chat: Optional[str] = None, limit: int = 5) -> Tuple[str, List[Dict]]:
        """
        Подбирает обмены для ответа на «что я спрашивал / о чём говорили / повтори».
        Возвращает (вид, строки по порядку времени).
        """
        kind = recall_kind(text) or "recall"
        archive = bool(ARCHIVE_RE.search(text))
        with self._lock:
            rows = [r for r in self.rows if archive or r.get("chat") == current_chat]
        if re.search(r"(?i)\bвчера\b", text):
            yesterday = date.today() - timedelta(days=1)
            rows = [r for r in rows if datetime.fromtimestamp(r["ts"]).date() == yesterday]
        elif archive:
            rows = [r for r in rows if r.get("chat") != current_chat]
        if not rows:
            return kind, []
        if kind == "repeat":
            return kind, rows[-1:]
        if re.search(r"(?i)\b(?:только\s+что|сейчас|до\s+этого|перед\s+этим|минуту\s+назад)\b", text):
            now = time.time()
            recent = [r for r in rows if now - r["ts"] <= FOLLOWUP_WINDOW_SEC]
            if recent:
                return kind, recent[-limit:]
        topic = set(_stems(RECALL_RE.sub(" ", text)))
        if topic:
            scored = []
            for i, r in enumerate(rows):
                hay = set(_stems(r["q"] + " " + r["a"]))
                hit = len(topic & hay)
                if hit:
                    # свежее — важнее; обмен текущего чата — тоже
                    score = hit + i / max(1, len(rows)) * 0.5 + (0.3 if r.get("chat") == current_chat else 0)
                    scored.append((score, i, r))
            if scored:
                best = sorted(scored, key=lambda t: -t[0])[:limit]
                return kind, [r for _, _, r in sorted(best, key=lambda t: t[1])]
            return kind, []
        # общий вопрос «что было раньше» — просто последние обмены
        return kind, rows[-limit:]

    @staticmethod
    def format_context(rows: List[Dict], max_chars: int = 1800) -> str:
        """Текст для подсказки модели: «(5 мин назад) Гость: … / Валера: …»."""
        now = time.time()
        parts = []
        for r in rows:
            parts.append(f"({_ago(r['ts'], now)}) Гость: {r['q']} | Валера: {r['a']}")
        text = "\n".join(parts)
        return text[-max_chars:] if len(text) > max_chars else text


# ---------------------------------------------------------------------------
# Подстройка под собеседника (в официальной форме «Вы»)
# ---------------------------------------------------------------------------
GREET_RE = re.compile(r"(?i)\b(?:привет\w*|здравств\w*|добрый\s+(?:день|вечер)|доброе\s+утро|приветствую|добрый\s+день)\b")
THANKS_RE = re.compile(r"(?i)\b(?:спасибо|благодарю|благодарствую|благодарен|благодарна)\b")
TY_RE = re.compile(r"(?i)\b(?:ты|тебе|тебя|твой|твоя|твоё|твое|тобой)\b")
UPSET_RE = re.compile(r"(?i)\b(?:тупо\w*|тупой|дурак\w*|бесит\w*|достал\w*|не\s+работает|ничего\s+не\s+понима\w*|ерунда|чушь)\b")


def greeting_for_hour(hour: int) -> str:
    if 5 <= hour < 12:
        return "Доброе утро"
    if 12 <= hour < 18:
        return "Добрый день"
    if 18 <= hour < 23:
        return "Добрый вечер"
    return "Доброй ночи"


def style_hint(text: str, now: Optional[float] = None) -> str:
    """
    Короткая подсказка модели, как ответить именно этому человеку: подстроиться
    по длине, настроению и обращению, но остаться на «Вы». Пустая строка, если
    подстраиваться не к чему.
    """
    if not text:
        return ""
    now = now or time.time()
    parts = []
    if GREET_RE.search(text):
        parts.append(f"Гость поздоровался: ответьте приветствием «{greeting_for_hour(time.localtime(now).tm_hour)}» "
                     "и коротко предложите помощь.")
    if THANKS_RE.search(text):
        parts.append("Гость благодарит: ответьте коротко и тепло, например «Пожалуйста, рад помочь».")
    if UPSET_RE.search(text):
        parts.append("Гость раздражён: ответьте спокойно, коротко извинитесь и предложите конкретную помощь.")
    n = len(text.split())
    if re.search(r"очень подробно|большой текст|все нюансы", text, re.I):
        parts.append("Гость просит подробный рассказ. Раскрой каждую названную тему с конкретным примером.")
    elif asks_next(text):
        parts.append("Гость просит продолжить: раскрой следующий пункт текущей темы.")
    elif n <= 3 and not parts:
        parts.append("Гость говорит коротко — ответьте кратко, одной фразой.")
    elif n >= 18:
        parts.append("Гость говорит развёрнуто — можно ответить чуть подробнее, но не больше трёх предложений.")
    if TY_RE.search(text):
        parts.append("Гость обращается на «ты», но Вы отвечаете на «Вы», вежливо и без фамильярности.")
    return " ".join(parts)
