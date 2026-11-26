import os
import re
import subprocess
import threading

GREETING_RE = re.compile(
    r"^\s*(?:привет|(?:(?:ещ[её] раз|и снова) )?здравствуйте|доброе утро|добрый день|добрый вечер)\b",
    re.IGNORECASE,
)
HANDSHAKE_RE = re.compile(
    r"(?:(?:дава(?:й|йте)\s+)?(?:пожм[её]м\s+руки|обменяемся\s+рукопожатием|"
    r"(?:я\s+)?пожму\s+(?:вам|тебе)\s+руку)|можно\s+пожать\s+(?:вам|тебе)\s+руку)"
    r"[.!?…]*", re.IGNORECASE,
)
_HAND_REQUEST_RE = re.compile(
    r"(?:дава(?:й|йте)\s+)?(?:пожм[её]м\s+(?:руки|друг\s+другу\s+руки)|руки\s+пожм[её]м|"
    r"(?:пожми(?:те)?|дай(?:те)?|протяни(?:те)?|мож(?:ешь|ете)\s+(?:пожать|дать|протянуть))"
    r"\s+(?:мне\s+)?(?:руку|лапу))", re.I,
)
_WAVE_REQUEST_RE = re.compile(
    r"(?:помаши(?:те)?|мож(?:ешь|ете)\s+помахать)(?:\s+(?:мне|нам|ей|ему|им))?(?:\s+рукой)?", re.I,
)
WAVE_REPLY = "Машу Вам рукой!"


def gesture_reply(command: str) -> str:
    """Подбирает произносимую фразу для прямой просьбы гостя."""
    command = command.strip(" .!?…")
    command = re.sub(r"^(?:(?:ну|ладно|тогда|пожалуйста|[аэ](?:-[аэ])+)[,\s]+)+", "", command, flags=re.I)
    command = re.sub(r"[,\s]+пожалуйста$", "", command, flags=re.I)
    if _HAND_REQUEST_RE.fullmatch(command) or HANDSHAKE_RE.fullmatch(command):
        return "Пожмём руки!"
    if _WAVE_REQUEST_RE.fullmatch(command):
        return WAVE_REPLY
    if re.fullmatch(r"поздоровай(?:ся|тесь)|поприветствуй(?:ся|те| нас| меня| гостей)?", command, re.I):
        return "Здравствуйте!"
    return ""


def choose_gesture(phrase: str, *, first_phrase: bool) -> str | None:
    """Выбирает жест для фразы, которую Валера собирается произнести."""
    if HANDSHAKE_RE.fullmatch(phrase.strip()):
        return "shake_hand"
    if phrase == WAVE_REPLY or (first_phrase and GREETING_RE.match(phrase)):
        return "wave_hand"

    return None


def send_gesture(gesture: str) -> bool:
    """Отправляет H1 один жест по SSH и возвращает результат команды."""
    if gesture not in {"wave_hand", "shake_hand"}:
        print(f"На H1 пока не настроен жест: {gesture}")
        return False

    command = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=5",
        "unitree@192.168.1.196",
        f"python3 ~/gesture_runner.py {gesture}",
    ]

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=45 if gesture == "shake_hand" else 10
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"Не удалось отправить жест: {error}")
        return False

    if result.returncode != 0:
        print(f"Ошибка команды жеста: {result.stderr.strip()}")
        return False

    return True


_gesture_lock = threading.Lock()


def gesture_busy() -> bool:
    return _gesture_lock.locked()


def start_gesture(gesture: str) -> None:
    """Запускает отправку жеста в фоне, чтобы голос мог звучать сразу."""
    if not _gesture_lock.acquire(blocking=False):
        print(f"Жест {gesture} пропущен: предыдущая команда ещё выполняется.", flush=True)
        return

    def worker():
        try:
            send_gesture(gesture)
        finally:
            _gesture_lock.release()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()


def handle_reply_gesture(phrase: str, *, first_phrase: bool, sent: set[str]) -> None:
    """Выбирает жест и обрабатывает его один раз за ответ Валеры."""
    gesture = choose_gesture(phrase, first_phrase=first_phrase)
    if gesture is None or gesture in sent:
        return

    sent.add(gesture)
    print(f"Жест: {gesture}", flush=True)

    if os.getenv("VALERA_ROBOT_GESTURES") == "1":
        start_gesture(gesture)
