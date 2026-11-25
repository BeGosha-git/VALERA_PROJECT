"""Порядок рассказа и сведения гостя в рамках одного диалога."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from xtts_service.speech_text import INSTITUTES

INSTITUTE_RE = re.compile(r"\b(?:" + "|".join(INSTITUTES) + r")\b", re.I)
TOUR = (
    "Полигон «Город»", "Полигон «Луна»", "Реконфигурируемый полигон",
    "Дипломы и стартапы", "Проекты и хакатоны", "KUKA youBot",
    "Роботы Unitree", "Манипуляторы KUKA iiwa и ABB GoFa", "Завершение экскурсии",
)
_ORDER_RE = re.compile(r"\b(?:по\s+(?:порядку|очереди|шагам)|пошагово|поочер[её]дно|вс[её]\s+подряд|про\s+(?:вс[её]|всех)|"
                       r"о\s+каждом|у\s+каждого|остальн\w+\s+институт\w*)\b", re.I)
_NEXT_RE = re.compile(r"(?:(?:ну|так|тогда|и|а|давай(?:те)?|рассказывай)[,\s]+)*"
                      r"(?:дальше|продолжай(?:те)?|продолжи(?:те)?|к следующему)(?:\s+рассказывай)?", re.I)
_EXAMPLE_RE = re.compile(r"(?:(?:ну|а)[,\s]+)?(?:например|приведи(?:те)? пример)", re.I)
_ADMISSION_RE = re.compile(r"\b(?:балл\w*|стоимост\w*|цен[ауые]|сколько стоит|поступ\w*)\b", re.I)
_LEVEL_RE = re.compile(r"\b(?:бакалавр\w*|специалитет\w*|магистратур\w*|аспирантур\w*|колледж\w*|спо)\b", re.I)
# Название без кавычек принимаем только после явного обозначения и с заглавной буквы.
_PROGRAM_RE = re.compile(r'\b\d{2}\.\d{2}\.\d{2}\b|[«"]([^»"]+)[»"]|'
                         r'(?i:\b(?:направлени[июяе]|профил[ьею]|программ[уеы]))\s+'
                         r'([А-ЯЁ][а-яё-]+(?:\s+[а-яё-]+){0,4})')
_SELECTION_RE = re.compile(
    r"куда.{0,35}поступ|шанс\w*\s+поступ|(?:хочу|хотел\w*|чтобы)\s+поступить\b|"
    r"как поступить (?:на|в)\b|"
    r"(?:выбр\w*|подобра\w*|посовет\w*|какие)\s+(?:себе\s+|есть\s+)?направлени|"
    r"направлени\w*.{0,25}(?:выбрать|подойд)|день открытых дверей.{0,180}направлени", re.I)
_INTERESTS = (
    (r"\b(?:робототехн|робот(?:ы|ов|ами|ах)\b|мехатрон)", "робототехника"),
    (r"\b(?:дрон|бпла\b|беспилот|мультиротор)", "беспилотные системы"),
    (r"\b(?:б[эе]к[эе]нд|backend)", "бэкенд"),
    (r"\b(?:программир|информатик|хими|радиоэлектрон|кибербезопасност)\w*", None),
)
_SELECTION_TOPIC = "Подбор направления"
_PREFERENCE_RE = re.compile(r"связан|увлекаюсь|интересуюсь|мне (?:нрав|ближе|интерес)|хотел[аи]? бы|"
                            r"что-то между|что изучают|чем.{0,24}занима", re.I)


def institute_overview(text: str) -> bool:
    """Просьба об общем перечне, в том числе с вопросом про баллы."""
    return bool(re.search(r"\b(?:все(?: виды)?|всех|каждый|какие (?:есть|бывают))\s+институт|"
                          r"институт\w*,?\s+какие есть", text, re.I))


def lab_robots(text: str) -> bool:
    return bool(re.search(r"\b(?:робот\w*|четыр[её]хног\w*|юнитри|unitree|go[12])\b", text, re.I))


def selection_topic(text: str, previous: str, answer: str) -> str:
    """Тема подбора, которую можно передать поиску и сохранить до следующей реплики."""
    if institute_overview(text):
        return ""
    active = previous.startswith(_SELECTION_TOPIC)
    interests = []
    for pattern, label in _INTERESTS:
        match = re.search(pattern, text, re.I)
        if match:
            interests.append(label or match[0].lower())
    if not interests and re.search(r"(?:этой|вашей|вашу|эту|нашей) лаборатори", text, re.I):
        if lab_robots(previous + " " + answer):
            interests.append("робототехника")
    if not interests and active:
        preference = re.search(r"(?:связан\w* с|интересуюсь|увлекаюсь|мне ближе)\s+([\w -]{2,65})", text, re.I)
        if preference:
            interests.append(preference[1].strip())
    short_choice = len(text.split()) <= 3 and re.fullmatch(r"[\w -]+[.!]?", text.strip())
    refinement = interests and (_PREFERENCE_RE.search(text) or short_choice)
    details = re.search(r"проходн\w* балл|сколько баллов|поступить|стоимость обучени|предмет\w* (?:сда|нуж)|"
                        r"экзамен|направлени|профил|там изучают", text, re.I)
    price = re.fullmatch(r"(?:а )?(?:сколько стоит|какая стоимость|стоимость|цена)[?.! ]*", text.strip(), re.I)
    if not (_SELECTION_RE.search(text) or active and (refinement or details or price)):
        return ""
    level = _LEVEL_RE.search(text) or (_LEVEL_RE.search(previous) if active else None)
    if level and re.match(r"колледж|спо", level[0], re.I):
        return ""
    base = f"{_SELECTION_TOPIC} в РТУ МИРЭА, {level[0] if level else 'высшее образование'}"
    # Поля берём только из текущей темы беседы. Новые интересы заменяют предыдущий выбор.
    fields = dict(part.split(": ", 1) for part in previous.split(". ") if ": " in part) if active else {}
    institutes = institute_names(text)
    if institutes:
        if fields.get("Институт") != ", ".join(institutes):
            fields.pop("Программа", None)
        fields["Институт"] = ", ".join(institutes)
    if interests:
        if fields.get("Интересы") != ", ".join(interests):
            fields.pop("Программа", None)
        fields["Интересы"] = ", ".join(interests)
    score = re.search(r"\b(\d{2,3})\s+балл", text, re.I)
    if score and not re.search(r"проходн\w*\s*$", text[:score.start()], re.I):
        personal_score = re.search(r"\b(?:у меня|мои(?:ми|х)?|я (?:набрал[аи]?|получил[аи]?))\b", text, re.I)
        if personal_score:
            fields["Баллы гостя"] = score[1]
    program = named_program(text)
    if not program and active and not interests and not institutes and "Программа" not in fields:
        # В ответе может быть несколько профилей одного направления. Сохраняем
        # только явно названное направление, иначе используем интересы гостя.
        directions = set(re.findall(r'(?i:направлени[иеюя])\s+[«"]([^»"]+)[»"]', answer))
        if len(directions) == 1:
            program = directions.pop()
    if program:
        fields["Программа"] = program
    return ". ".join([base, *(f"{key}: {value}" for key, value in fields.items())])


def institute_names(text: str) -> list[str]:
    return list(dict.fromkeys(m.group().upper() for m in INSTITUTE_RE.finditer(text)))


def search_topic(item: str) -> str:
    return f"{item} — {INSTITUTES[item]}" if item in INSTITUTES else item


def ordered_request(text: str) -> bool:
    return bool(_ORDER_RE.search(text))


def names_new_subject(text: str) -> bool:
    rest = _ORDER_RE.sub("", text.casefold())
    return bool(set(re.findall(r"\w+", rest)) - {
        "а", "ну", "так", "да", "давай", "давайте", "мне", "мы", "ты", "вы", "это", "этом", "то",
        "расскажи", "расскажите", "рассказывай", "опиши", "объясни", "разбери", "тогда", "обо",
        "подробно", "полностью", "пожалуйста", "про", "о", "об", "все", "всё", "продолжай",
    })


def asks_next(text: str) -> bool:
    phrase = re.sub(r"^\s*(?:о том|то),?\s+(?:о ч[её]м|что)\s+(?:ты|вы)\s+предлож\w+[?,.!\s]*",
                    "", text, flags=re.I)
    return bool(_NEXT_RE.fullmatch(phrase.strip(" ,.!?…")))


def asks_example(text: str) -> bool:
    return bool(_EXAMPLE_RE.fullmatch(text.strip(" ,.!?…")))


def admission_scope(text: str, previous_topic: str) -> str:
    """Наследуем предмет приёма, сохраняя явно названный уровень образования."""
    if not _ADMISSION_RE.search(text):
        return ""
    if not re.search(r"балл|обучени|поступ|программ|направлени|институт|магистратур|бакалавр", text, re.I):
        if not re.fullmatch(r"(?:а\s+)?(?:сколько стоит|стоимость|цена)[?.!\s]*", text.strip(), re.I):
            return ""
    level = _LEVEL_RE.search(text)
    if level and re.match(r"колледж|спо", level.group(), re.I):
        return ""
    names = institute_names(text) or institute_names(previous_topic)
    if not names and not re.search(r"институт", text + " " + previous_topic, re.I):
        return ""
    level = level or _LEVEL_RE.search(previous_topic)
    education = level.group() if level else "высшее образование"
    institute = ", ".join(search_topic(name) for name in names) or "институты РТУ МИРЭА"
    return f"{institute}, {education}"


def needs_program(text: str) -> bool:
    asks_numbers = re.search(r"балл|стоимост|цен[ауые]|сколько стоит", text, re.I)
    named = _PROGRAM_RE.search(text)
    return bool(asks_numbers and not named)


def named_program(text: str) -> str:
    match = _PROGRAM_RE.search(text)
    return (match.group(1) or match.group(2) or match.group()) if match else ""


@dataclass(frozen=True)
class Outline:
    items: tuple[str, ...]
    covered: tuple[str, ...] = ()
    current: str = ""
    mirea: bool = False
    subject: str = ""
    unavailable: str = ""

    @property
    def upcoming(self) -> str:
        return next((item for item in self.items if item not in self.covered), "")

    def after_answer(self, target: str, answer: str) -> Outline:
        """Сдвигаем позицию после содержательной части, включая два реально описанных пункта."""
        statements = [
            s for s in re.split(r"(?<=[.!?…])\s+", answer)
            if not s.endswith("?") and len(s.split()) >= 3 and not re.search(
                r"^\s*(?:(?:я|мы)\s+)?(?:могу рассказать|расскажу|хотите|давайте|перейд[её]м|сначала опишу|"
                r"продолжу рассказ|переходим|уточните)|\b(?:не знаю|нет сведений|нет данных|"
                r"не удалось|по этому (?:списку|пункту)|не наш[её]л)\b", s, re.I)
        ]
        if target not in self.items:
            return self
        if not statements and re.search(r"по этому пункту|нет сведений|не знаю|нет данных", answer, re.I):
            return replace(self, unavailable=target)
        described = []
        for sentence in statements:
            mentions = [item for item in self.items if item.casefold() in sentence.casefold()]
            if len(mentions) == 1 and len(sentence.split()) > len(mentions[0].split()) + 1:
                described.append(mentions[0])
            elif not mentions:
                # В ответе на вопрос об одном пункте возможна фраза «Здесь изучают…».
                described.append(target)
        if not described:
            return self
        covered = tuple(dict.fromkeys((*self.covered, *described)))
        return replace(self, covered=covered, current=described[-1], unavailable="")

    def next_part(self) -> Outline:
        if self.unavailable and self.unavailable == self.upcoming:
            return replace(self, covered=(*self.covered, self.unavailable), unavailable="")
        return self


@dataclass(frozen=True)
class GuestFact:
    subject: str
    relation: str
    value: str

    @property
    def statement(self) -> str:
        return " ".join(filter(None, (self.subject, self.relation, self.value)))


_REMEMBER = re.compile(r"^(?:(?:ну|пожалуйста|валера)[,\s]+)*запомни(?:те)?[,\s]+(?:что\s+)?", re.I)
_CORRECTION = re.compile(r"^(?:нет|поправка|исправляю|точнее|на самом деле|я имел[аи]? в виду)[,\s]+", re.I)
_RELATION = re.compile(r"\b(находится|расположен[аоы]?|называется|стоит|начинается)\s+", re.I)


def guest_statement(text: str, previous: GuestFact | None) -> GuestFact | None:
    phrase = text.strip(" ,.!?…")
    remember = _REMEMBER.match(phrase)
    correction = _CORRECTION.match(phrase)
    replacement = re.fullmatch(r"(?:нет[,\s]+)?не\s+(.+?),?\s+а\s+(.+)", phrase, re.I)
    if previous and replacement and "?" not in text:
        old, new = replacement.groups()
        if previous.value.casefold().endswith(old.casefold()):
            return replace(previous, value=previous.value[:-len(old)] + new)
    if not remember and not (previous and correction):
        return None
    body = phrase[(remember or correction).end():].strip(" ,")
    if not body or "?" in text:
        return None
    relation = _RELATION.search(body) or re.search(r"\s+([—–-])\s+", body)
    if relation:
        subject = body[:relation.start()].strip(" ,")
        if not subject and previous and correction:
            subject = previous.subject
        value = body[relation.end():].strip()
        if subject and value:
            return GuestFact(subject, relation.group(1).lower(), value)
    if remember:
        return GuestFact("", "", body)
    if previous and previous.relation in {"—", "–", "-", "называется"} and len(body.split()) <= 5:
        return replace(previous, value=body)
    if previous and previous.relation.startswith(("наход", "располож")) and re.match(r"(?:в|на)\s+", body, re.I):
        return replace(previous, value=body)
    if previous and re.fullmatch(r"(?:в\s+)?[А-Яа-яA-Za-z]{0,4}[- ]?\d+[А-Яа-яA-Za-z]?", body):
        value = body if not previous.value.startswith("в ") or body.startswith("в ") else "в " + body
        return replace(previous, value=value)
    return None


def asks_about_fact(text: str, fact: GuestFact) -> bool:
    if not fact.subject:
        return False
    words = re.findall(r"[а-яёa-z0-9]+", text.casefold())
    subject = {w[:6] for w in re.findall(r"[а-яёa-z0-9]+", fact.subject.casefold())
               if len(w) > 2 and w not in {"мой", "моя", "моё", "мое", "наш", "наша", "это"}}
    if not subject.intersection(w[:6] for w in words):
        return False
    if fact.relation.startswith(("наход", "располож")):
        return "где" in words or "адрес" in words
    if fact.relation in {"—", "–", "-"}:
        return bool({"какой", "какая", "какое", "какие", "кто", "что"} & set(words))
    return fact.relation in words and ("?" in text or bool({"как", "когда", "сколько"} & set(words)))
