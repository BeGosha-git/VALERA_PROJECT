#!/usr/bin/env python3
"""
valera_reset.py
===============
Сброс состояния робота-экскурсовода «Валера» к чистому виду.

Что чистится
------------
1. **Локальные файлы состояния** (в корне проекта):
   - ``chat_id.txt``     — id текущего чата WebRAgent (контекст разговора);
   - ``questions.log``   — журнал заданных вопросов и ответов;
   - ``triggers.log``    — журнал сработавших триггеров «слово → команда»;
   - ``valera_config.json`` — настройки GUI (тумблеры, пороги, триггеры).

2. **Чаты WebRAgent в MongoDB** — вся история диалогов. При ``--chats``
   удаляются все чаты пользователя (или конкретный чат через ``--chat-id``).
   По умолчанию Mongo *не* трогается: файлы чистятся, а история чата
   остаётся, если вы явно не попросите обратное.

Зачем
-----
Со временем состояние накапливается: ``questions.log`` растёт на сотни
килобайт, в MongoDB копятся десятки чатов от прошлых сессий. Перед демо,
деплоем или отладкой удобно начать с чистого листа.

Использование
-------------
    python valera_reset.py                     # только локальные файлы
    python valera_reset.py --chats             # + удалить ВСЕ чаты из MongoDB
    python valera_reset.py --chat-id <id>      # + удалить один конкретный чат
    python valera_reset.py --keep-config       # файлы, но НЕ трогать настройки GUI
    python valera_reset.py --dry-run           # показать, что будет удалено
    python valera_reset.py --yes               # без вопроса-подтверждения

Через ``run_all.sh``:
    ./run_all.sh reset              # локальные файлы + чаты MongoDB
    ./run_all.sh reset --dry-run    # посмотреть, что удалится
"""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Локальные файлы состояния
STATE_FILES = [
    ("chat_id.txt", "id текущего чата (контекст разговора)"),
    ("questions.log", "журнал заданных вопросов и ответов"),
    ("triggers.log", "журнал сработавших триггеров"),
]
CONFIG_FILE = "valera_config.json"


def human_size(path: str) -> str:
    try:
        size = os.path.getsize(path)
    except OSError:
        return "—"
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} ТБ"


def clear_local_files(keep_config: bool, dry_run: bool) -> int:
    """Удаляет локальные файлы состояния. Возвращает число удалённых."""
    removed = 0
    targets = list(STATE_FILES)
    if not keep_config:
        targets.append((CONFIG_FILE, "настройки GUI (тумблеры, пороги, триггеры)"))

    print("── Локальные файлы состояния ──")
    for name, desc in targets:
        path = os.path.join(PROJECT_DIR, name)
        if not os.path.exists(path):
            print(f"   · {name:22} — нет (пропуск)")
            continue
        size = human_size(path)
        if dry_run:
            print(f"   · {name:22} ({size}) — БУДЕТ УДАЛЁН ({desc})")
        else:
            try:
                os.remove(path)
                print(f"   ✓ {name:22} ({size}) удалён — {desc}")
                removed += 1
            except OSError as exc:
                print(f"   ✗ {name:22} не удалён: {exc}")
    return removed


def clear_mongo_chats(chat_id: str | None, dry_run: bool) -> int:
    """Удаляет чаты в MongoDB через docker exec mongosh. Возвращает число удалённых."""
    import subprocess

    container = os.getenv("MONGO_CONTAINER", "webragent-mongodb-1")
    db = os.getenv("MONGO_DB", "ragapp")

    # Проверяем, что контейнер жив
    ps = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True
    )
    if ps.returncode != 0:
        print("   ✗ Docker недоступен — чаты не тронуты.")
        return 0
    if container not in ps.stdout.split():
        print(f"   ✗ Контейнер {container} не запущен — чаты не тронуты.")
        print("     Запустите стек: ./run_all.sh  (или docker compose up -d в WebRAgent/)")
        return 0

    # Формируем фильтр
    if chat_id:
        chat_filter = f"{{chat_id: '{chat_id}'}}"
        msg_filter = f"{{chat_id: '{chat_id}'}}"
        what = f"чат {chat_id}"
    else:
        chat_filter = "{}"
        msg_filter = "{}"
        what = "ВСЕ чаты"

    count_js = (
        f"print(db.chat_history.countDocuments({chat_filter}) + '|' + "
        f"db.chat_messages.countDocuments({msg_filter}))"
    )
    res = subprocess.run(
        ["docker", "exec", container, "mongosh", db, "--quiet", "--eval", count_js],
        capture_output=True, text=True,
    )
    counts = (res.stdout or "0|0").strip().split("|")
    n_chats = counts[0] if counts else "0"
    n_msgs = counts[1] if len(counts) > 1 else "0"

    print("── Чаты WebRAgent (MongoDB) ──")
    print(f"   найдено: чатов {n_chats}, сообщений {n_msgs}  ({what})")
    if n_chats == "0" and n_msgs == "0":
        return 0

    if dry_run:
        print(f"   БУДУТ УДАЛЕНЫ: {what} и связанные сообщения")
        return 0

    del_js = (
        f"print(db.chat_history.deleteMany({chat_filter}).deletedCount + '|' + "
        f"db.chat_messages.deleteMany({msg_filter}).deletedCount)"
    )
    res = subprocess.run(
        ["docker", "exec", container, "mongosh", db, "--quiet", "--eval", del_js],
        capture_output=True, text=True,
    )
    out = (res.stdout or "").strip().split("|")
    if res.returncode != 0:
        print(f"   ✗ Ошибка удаления: {(res.stderr or '').strip()[:200]}")
        return 0
    deleted = int(out[0]) if out and out[0].isdigit() else 0
    print(f"   ✓ удалено чатов: {out[0] if out else '?'}, "
          f"сообщений: {out[1] if len(out) > 1 else '?'}")
    return deleted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Сброс состояния робота Валеры (файлы + чаты WebRAgent)"
    )
    parser.add_argument("--chats", action="store_true",
                        help="Удалить ВСЕ чаты WebRAgent из MongoDB")
    parser.add_argument("--chat-id", default=None,
                        help="Удалить конкретный чат по id")
    parser.add_argument("--keep-config", action="store_true",
                        help="Не удалять valera_config.json (настройки GUI)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Показать, что будет удалено, ничего не трогая")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Не спрашивать подтверждение")
    parser.add_argument("--mongo-container", default=None,
                        help="Имя контейнера MongoDB (по умолчанию webragent-mongodb-1)")
    args = parser.parse_args()

    if args.mongo_container:
        os.environ["MONGO_CONTAINER"] = args.mongo_container

    do_chats = args.chats or bool(args.chat_id)

    print("=" * 62)
    print("  Сброс состояния робота-экскурсовода «Валера»")
    print(f"  Локальные файлы: ДА" + ("  (кроме настроек GUI)" if args.keep_config else ""))
    print(f"  Чаты WebRAgent:  {'ДА' if do_chats else 'нет'}")
    if args.dry_run:
        print("  РЕЖИМ: dry-run (ничего не удаляется)")
    print("=" * 62)
    print()

    if not args.dry_run and not args.yes:
        answer = input("Удалить перечисленное? [y/N]: ").strip().lower()
        if answer not in ("y", "yes", "д", "да"):
            print("Отменено.")
            return 0
        print()

    removed_files = clear_local_files(args.keep_config, args.dry_run)
    removed_chats = 0
    if do_chats:
        removed_chats = clear_mongo_chats(args.chat_id, args.dry_run)

    print()
    print("=" * 62)
    if args.dry_run:
        print("  Проверка завершена. Запустите без --dry-run, чтобы удалить.")
    else:
        print(f"  Готово. Удалено файлов: {removed_files}, чатов: {removed_chats}")
        print("  Следующий запуск начнётся с чистого контекста.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
