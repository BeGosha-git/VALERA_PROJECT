"""Подбор небольших выдержек из таблиц программ приёма для голосового ответа."""
import re
from pathlib import PurePath

_SELECTION = re.compile(
    r"подбор направления в РТУ МИРЭА|куда.{0,30}поступ|(?:каки[ехм]|какую|подбери|посоветуй|порекомендуй|назови|примеры)"
    r".{0,45}(?:направлени|профил|специальност|программ)|"
    r"(?:направлени|профил|программ|специальност).{0,50}(?:подойд|выбра|посовету)|"
    r"подойд.{0,70}(?:платк|обучени|поступ)|(?:посоветуй|порекомендуй).{0,30}куда", re.I)
_LEVEL = re.compile(r"магистратур|бакалавриат|специалитет|колледж|аспирантур|\bспо\b|после девятого|после 9", re.I)
_STOP = {"прогр", "обуче", "напри", "напра", "инсти", "решен", "техно", "подой",
         "подхо", "сколь", "посту", "специ", "профи", "интер", "какие", "униве"}


def asks_programs(question):
    if re.search(r"скачать|установить|для компьютера|для телефона", question, re.I):
        return False
    return bool(_SELECTION.search(question))


def _terms(text):
    words = {word[:5] for word in re.findall(r"[а-яё]{4,}", text.lower())}
    if re.search(r"\bхим", text, re.I):
        words.add("хим")
    if re.search(r"программир|программная инженерия|пишу код", text, re.I):
        words.add("программир")
    if re.search(r"дрон|бпла|беспилот", text, re.I):
        words.update(("беспи", "мульт"))
    if re.search(r"б[эе]к[эе]нд|backend", text, re.I):
        words.add("разра")
    return words - _STOP


def examples(items, question, history):
    previous = [m["content"] for m in reversed((history or [])[-8:])
                if m.get("role") == "user" and m.get("content") != question]
    turns = [question, *previous]
    level = next((m.group().lower() for t in turns if (m := _LEVEL.search(t))), "")
    if level and level not in ("магистратур", "бакалавриат", "специалитет"):
        return []
    master = level == "магистратур"
    table = "programs_master_2026.csv" if master else "programs_bachelor_specialist_2026.csv"
    candidates = []
    for item in items:
        if PurePath(item["title"].replace(" / ", "/")).name != table:
            continue
        for row in re.split(r"(?:Институт / подразделение|Институт)=", item["content"])[1:]:
            code = re.search(r"; Код=(\d{2}\.\d{2}\.\d{2});", row)
            name = re.search(r"; (?:Программа|Направление — профиль)=(.*?); (?:Уровень|Бюджет)", row)
            price = re.search(r"; Стоимость 2026/27[^=]*=(.+)$", row.strip(), re.S)
            if not (code and name and price):
                continue
            degree = code[1].split(".")[1]
            if level and degree != {"бакалавриат": "03", "специалитет": "05", "магистратур": "04"}[level]:
                continue
            scores = re.findall(r"; (?:Проходной балл|Балл) (\d{4})=([^;]+)", row)
            scores = [(year, re.sub(r"\s*\[И\d+\]", "", value).strip()) for year, value in scores]
            results = ", ".join(f"{year} — {value}" for year, value in scores if re.search(r"\d", value))
            institute = row.partition("; Код=")[0]
            tokens = _terms(name[1])
            candidates.append((tokens, code[1], name[1], institute, price[1].strip(), results, item))
    if not candidates:
        return []
    institute_pattern = r"\b(?:ИИТ|ИИИ|ИКБ|ИРИП|ИТУ|ИТХТ|ПИШ)\b"
    institutes = next((names for t in turns if (names := re.findall(institute_pattern, t, re.I))), [])
    if institutes:
        candidates = [r for r in candidates if any(i.upper() in r[3].upper() for i in institutes)]
        if not candidates:
            return []
    code = re.search(r"\b\d{2}\.\d{2}\.\d{2}\b", question)
    if code:
        candidates = [row for row in candidates if row[1] == code[0]]
        if not candidates:
            return []
    scored = []
    for turn in turns:
        terms = _terms(turn)
        for row in candidates:
            score = len(terms & row[0])
            if "между" in turn.lower() and "робот" in terms and "разра" in terms:
                # На стыке разработки и робототехники релевантны программы о
                # соединении ПО с физическим устройством.
                if row[0] & {"кибер", "встра"} and re.search(r"киберфизич|встраиваем", row[2], re.I):
                    score += 2
            if score:
                scored.append((score, row))
        if scored:
            break
    if scored:
        best = max(score for score, _ in scored)
        selected = [r for score, r in scored if score == best]
    else:
        selected = candidates
    first = selected[0]
    second = next((r for r in selected[1:] if r[1] != first[1]), None)
    if second is None:
        second = next((r for r in selected[1:] if r[2] != first[2]), None)
    result = []
    for _, code, name, institute, price, scores, item in [first] + ([second] if second else []):
        fee = f"{price} рублей в год" if re.search(r"\d", price) else "не указана"
        content = f"{institute}. {code}. {name.replace(' — ', '; профиль: ', 1)}. Стоимость на 2026/27 учебный год: {fee}."
        if scores:
            content += f" Проходные баллы по годам: {scores}."
        result.append(dict(item["payload"], content=content))
    return result
