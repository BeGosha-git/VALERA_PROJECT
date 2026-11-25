import re
import os
import subprocess
import threading
GREETING_RE = re.compile(
    r"^\s*(?:привет|здравствуйте|доброе утро|добрый день|добрый вечер)\b",
    re.IGNORECASE,
)


def choose_gesture(phrase: str, *, first_phrase: bool) -> str | None:
    """Выбирает жест для фразы, которую Валера собирается произнести."""
    if first_phrase and GREETING_RE.match(phrase):
        return "wave_hand"

    return None


def send_gesture(gesture: str) -> bool:
    """Отправляет H1 один жест по SSH и возвращает результат команды."""
    if gesture != "wave_hand":
        print(f"На H1 пока не настроен жест: {gesture}")
        return False

    command = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=5",
        "unitree@192.168.1.196",
        "python3 ~/gesture_runner.py wave_hand",
    ]

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"Не удалось отправить жест: {error}")
        return False

    if result.returncode != 0:
        print(f"Ошибка команды жеста: {result.stderr.strip()}")
        return False

    return True
_gesture_lock = threading.Lock()


def start_gesture(gesture: str) -> None:
    """Запускает отправку жеста в фоне, чтобы голос мог звучать сразу."""
    if not _gesture_lock.acquire(blocking=False):
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