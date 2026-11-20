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
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

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
REPEAT_RE = re.compile(r"(?i)\b(?:повтори\w*|скажи\s+ещё\s+раз|ещё\s+раз|что\s+ты\s+сказал)\b")
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

    @property
    def clarification(self) -> str:
        return "Вы про " + " или ".join(self.choices) + "?" if self.choices else ""


def is_offer_confirmation(text: str) -> bool:
    """Проверяет, может ли реплика быть согласием на предложение."""
    phrase = text.strip(" ,.!?…")
    return bool(
        re.fullmatch(
            r"(?i)(?:(?:валера|лера)[,\s]+)?(?:ну[,\s]+)?"
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
    if REPEAT_RE.search(text):
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
            topic: Optional[str] = None, use_as_topic: bool = True) -> None:
        if not question or not answer:
            return
        row = {"ts": time.time(), "chat": chat_id, "q": question.strip()[:400],
               "a": answer.strip()[:600], "mirea": bool(mirea)}
        if eq and eq.strip() != question.strip():
            row["eq"] = eq.strip()[:300]   # «эффективный» вопрос с учётом контекста
        if topic and topic.strip() != question.strip():
            row["topic"] = topic.strip()[:240]
        if not use_as_topic:
            row["use_as_topic"] = False
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
        if pending:
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
    if n <= 3 and not parts:
        parts.append("Гость говорит коротко — ответьте кратко, одной фразой.")
    elif n >= 18:
        parts.append("Гость говорит развёрнуто — можно ответить чуть подробнее, но не больше трёх предложений.")
    if TY_RE.search(text):
        parts.append("Гость обращается на «ты», но Вы отвечаете на «Вы», вежливо и без фамильярности.")
    return " ".join(parts)
