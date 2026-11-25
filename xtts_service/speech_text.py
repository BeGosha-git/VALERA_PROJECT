"""Подготовка текста, который непосредственно произносит голосовой движок."""
import re
import unicodedata

# Названия из глоссария базы МИРЭА. По полному названию поиск различает институты.
INSTITUTES = {
    "ИИТ": "Институт информационных технологий",
    "ИИИ": "Институт искусственного интеллекта",
    "ИКБ": "Институт кибербезопасности и цифровых технологий",
    "ИРИП": "Институт радиоэлектроники и индустриального программирования",
    "ИТУ": "Институт технологий управления",
    "ИТХТ": "Институт тонких химических технологий",
}
# Названия букв нужны до удаления дефисов и скобок. Обычные слова
# в основном тексте и написание ответа на экране сохраняются.
_INITIALS = {"ИИТ": "и и тэ", "ИИИ": "и и и", "ИКБ": "и ка бэ", "ИРИП": "и эр и пэ",
             "ИТУ": "и тэ у", "ИТХТ": "и тэ ха тэ", "СВЧ": "эс вэ чэ"}
_INITIALS_RE = re.compile(r"\b(?:" + "|".join(_INITIALS) + r")\b")

_NAME = r"(?:" + "|".join(INSTITUTES) + r")"
_INSTITUTE_LIST = re.compile(
    r"\b(?:(?P<prep>(?i:в|об|о|из|от|для|с|к))\s+)?(?P<items>" + _NAME
    + r"(?:(?:\s*,\s*(?:и\s+)?|\s+(?:и|или)\s+)" + _NAME + r")+)(?!\w)"
)
_INSTITUTE_TOKEN = re.compile(r"\b" + _NAME + r"\b")


def _read_institute_list(match: re.Match) -> str:
    prep = match["prep"] or ""
    noun = {"в": "институте", "об": "институте", "о": "институте", "из": "института",
            "от": "института", "для": "института", "с": "институтом", "к": "институту"}.get(prep.lower(), "институт")
    names = _INSTITUTE_TOKEN.sub(lambda m: INSTITUTES[m[0]].replace("Институт", noun, 1), match["items"])
    return (prep + " " if prep else "") + names


def words_only(text: str) -> str:
    """Оставляет слова и ударения. Разделитель между словами заменяет пробелом."""
    text = unicodedata.normalize("NFC", text)
    text = _INSTITUTE_LIST.sub(_read_institute_list, text)
    text = _INITIALS_RE.sub(lambda m: _INITIALS[m[0]], text)
    # У серии GoFa стык двух коротких имён у XTTS сливается в один звук.
    text = re.sub(r"\b(?:abb|а[- ]?б[- ]?б)\s+(?:gofa|го[- ]?фа)\b", "АББ серии Гофа", text, flags=re.I)
    text = re.sub(r"\b(?:abb|а[- ]?б[- ]?б)\b", "а бэ бэ", text, flags=re.I)
    text = re.sub(r"\b(?:iiwa|и[- ]и[- ]вэ[- ]а)\b", "ива", text, flags=re.I)
    text = re.sub(r"\b(?:gofa|го[- ]фа)\b", "гофа", text, flags=re.I)
    clean = "".join(ch if ch.isalnum() or unicodedata.category(ch).startswith("M") else " " for ch in text)
    return " ".join(word for word in clean.split() if any(ch.isalnum() for ch in word))
