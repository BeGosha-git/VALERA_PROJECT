"""Чтение чисел в тексте ответа перед показом и озвучкой."""
import re
from datetime import date
from decimal import Decimal

from num2words import num2words

_MONTHS = "января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split()
_AMOUNT = r"(?<!\d)(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[.,]\d+)?"


def _ordinal(value, **forms):
    return num2words(int(value), lang="ru", to="ordinal", **forms)


def spoken_numbers(text):
    if not re.search(r"\d", text):
        return text

    names = {"go1": "Го один", "go2": "Го два", "h1": "Аш-один", "g1": "Джи один", "3d": "три дэ", "5g": "пять джи", "v2": "вэ два"}
    text = re.sub(r"\b(?:go1|go2|h1|g1|3d|5g|v2)\b", lambda m: names[m[0].lower()], text, flags=re.I)

    def calendar(match):
        day, month, year = match.group("day", "month", "year")
        month_num = _MONTHS.index(month.lower()) + 1 if month.lower() in _MONTHS else int(month)
        try:
            date(int(year or 2000), month_num, int(day))
        except ValueError:
            return match.group()
        # «с первого мая», «на первое мая», «дата — первое мая».
        prefix = match.string[:match.start()]
        case = "genitive" if re.search(r"\b(?:с|со|от|до|после)\s+$", prefix, re.I) else "nominative"
        result = _ordinal(day, gender="neuter", case=case) + " " + _MONTHS[month_num-1]
        return result + (" " + _ordinal(year, case="genitive") + " года" if year else "")

    suffix = r"(?:\s+(?:года\b|г\.))?"
    text = re.sub(r"\b(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})\b", calendar, text)
    text = re.sub(r"\b(?P<day>\d{1,2})(?P<sep>[./-])(?P<month>\d{1,2})(?P=sep)(?P<year>\d{4})\b" + suffix, calendar, text)
    text = re.sub(r"\b(?P<day>\d{1,2})\s+(?P<month>" + "|".join(_MONTHS) +
                  r")\b(?:\s+(?P<year>\d{4})\b" + suffix + ")?", calendar, text, flags=re.I)
    text = re.sub(r"\b(в|с|до|от)\s+(\d{4})\s+г\.", lambda m: m[1] + " " + m[2] + " " +
                  {"в": "году", "с": "года", "до": "года", "от": "года"}[m[1].lower()], text, flags=re.I)
    cases = {"год": "nominative", "года": "genitive", "году": "prepositional", "годом": "instrumental"}
    text = re.sub(r"\b(\d{4})\s+(год(?:ом|а|у)?)\b",
                  lambda m: _ordinal(m[1], case=cases[m[2].lower()]) + " " + m[2], text, flags=re.I)

    def dotted(match):
        pieces = match["number"].split(".")
        spoken = []
        for part in pieces:
            if part.startswith("0") and len(part) > 1:
                spoken.append(" ".join(num2words(int(d), lang="ru") for d in part))
            else:
                spoken.append(num2words(int(part), lang="ru"))
        return (match.groupdict().get("prefix") or "") + " точка ".join(spoken)

    text = re.sub(r"(?P<prefix>\bверси[яиюей]+\s+)[vв]?(?P<number>\d+(?:\.\d+)+)", dotted, text, flags=re.I)
    text = re.sub(r"(?<!\d)(?P<number>\d+(?:\.\d+){2,})(?!\d)", dotted, text)

    def amount(match):
        value = Decimal("".join(match[0].split()).replace(",", "."))
        case = "genitive" if re.search(r"\b(?:от|до|около|свыше)\s+$", match.string[:match.start()], re.I) else "nominative"
        return num2words(value, lang="ru", case=case)

    def price(match):
        value = Decimal("".join(match[1].split()).replace(",", "."))
        if re.search(r"\b(?:от|до|около|порядка|свыше)\s+$", match.string[:match.start()], re.I):
            def genitive(n, gender, one, many):
                unit = one if n % 10 == 1 and n % 100 != 11 else many
                return num2words(n, lang="ru", case="genitive", gender=gender) + " " + unit

            rubles = int(value)
            cents = int((value - rubles) * 100)
            result = genitive(rubles, "masculine", "рубля", "рублей")
            if cents:
                result += " " + genitive(cents, "feminine", "копейки", "копеек")
            return result
        return num2words(value, lang="ru", to="currency", currency="RUB").removesuffix(", ноль копеек")

    text = re.sub("(" + _AMOUNT + r")\s*(?:₽|руб(?:лей|ля|ль|ли)\b|руб\b\.?)", price, text, flags=re.I)
    text = re.sub(r"(?<=\d)[–—-](?=\d)", " — ", text)
    text = re.sub(r"(?<!\w)[−-](?=\d)", "минус ", text)
    def percent(match):
        number = Decimal("".join(match[1].split()).replace(",", "."))
        ending = "процента" if number % 1 else _percent_unit(int(number))
        return num2words(number, lang="ru") + " " + ending

    text = re.sub("(" + _AMOUNT + r")\s*%", percent, text)
    return re.sub(_AMOUNT, amount, text)


def _percent_unit(value):
    if 11 <= value % 100 <= 14:
        return "процентов"
    return {1: "процент", 2: "процента", 3: "процента", 4: "процента"}.get(value % 10, "процентов")
