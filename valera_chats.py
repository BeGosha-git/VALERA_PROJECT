"""
valera_chats.py
===============
Управление чатами WebRAgent (контекст диалога робота Валеры).

Зачем отдельный модуль
----------------------
WebRAgent хранит историю разговоров в MongoDB. За долгую работу ассистента
там накапливаются десятки чатов от прошлых сессий — это мусор, который
мешает и путает (легко случайно подключиться к старому контексту).

Здесь собраны операции над чатами:
* :meth:`ChatStore.list` — список чатов (кто, когда, сколько сообщений);
* :meth:`ChatStore.stats` — сводка (сколько всего, сколько «старых»);
* :meth:`ChatStore.delete` — удалить конкретные чаты;
* :meth:`ChatStore.delete_older_than` — удалить всё старше N дней (кнопка
  «Удалить старые чаты»);
* :meth:`ChatStore.delete_all` — вычистить всё (кроме, при желании, активного).

Почему напрямую в Mongo, а не через HTTP
----------------------------------------
В WebRAgent нет API «список всех чатов» и «удалить группу чатов» — только
поштучно и только для чатов текущего пользователя. Прямой доступ к MongoDB
даёт полный контроль и работает быстро, когда чатов много.

Модуль не падает, если Mongo недоступна: возвращает пустые списки и понятную
ошибку — интерфейс продолжает работать.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger("valera.chats")

# Параметры подключения (совпадают с WebRAgent/.env)
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "ragapp")
CHATS_COLLECTION = os.getenv("MONGO_COLLECTION_CHATS", "chat_history")
MESSAGES_COLLECTION = os.getenv("MONGO_COLLECTION_MESSAGES", "chat_messages")


class ChatStore:
    """
    Обёртка над MongoDB-коллекциями чатов WebRAgent.

    Создавайте по экземпляру на запрос (дёшево) либо держите один —
    ``pymongo.MongoClient`` потокобезопасен и сам пулит соединения.
    """

    def __init__(self, mongo_uri: str = MONGO_URI, db_name: str = MONGO_DB):
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self._client = None
        self._db = None
        self.last_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Подключение
    # ------------------------------------------------------------------
    def _ensure_db(self):
        """Ленивое подключение к Mongo. Возвращает БД или None при ошибке."""
        if self._db is not None:
            return self._db
        try:
            import pymongo

            self._client = pymongo.MongoClient(
                self.mongo_uri, serverSelectionTimeoutMS=3000
            )
            # Пинг, чтобы ошибка всплыла сразу, а не при первом запросе
            self._client.admin.command("ping")
            self._db = self._client[self.db_name]
            self.last_error = None
            return self._db
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"MongoDB недоступна: {exc.__class__.__name__}"
            log.warning("MongoDB недоступна: %s", exc)
            return None

    @property
    def available(self) -> bool:
        """True, если Mongo отвечает (для отображения статуса в интерфейсе)."""
        return self._ensure_db() is not None

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None
            self._db = None

    # ------------------------------------------------------------------
    # Чтение
    # ------------------------------------------------------------------
    def list(self, limit: int = 200) -> List[Dict[str, Any]]:
        """
        Возвращает список чатов, самые свежие — первыми.

        Поля каждой записи: ``chat_id``, ``title``, ``created_at``,
        ``updated_at``, ``message_count``, ``age_days`` (сколько дней чату).
        """
        db = self._ensure_db()
        if db is None:
            return []
        try:
            cursor = (
                db[CHATS_COLLECTION]
                .find({}, {"_id": 0})
                .sort("updated_at", -1)
                .limit(max(1, int(limit)))
            )
            now = datetime.now(timezone.utc)
            out: List[Dict[str, Any]] = []
            for doc in cursor:
                updated = doc.get("updated_at")
                age_days = None
                if isinstance(updated, datetime):
                    if updated.tzinfo is None:
                        updated = updated.replace(tzinfo=timezone.utc)
                    age_days = (now - updated).total_seconds() / 86400.0
                out.append({
                    "chat_id": doc.get("chat_id", ""),
                    "title": doc.get("title") or "(без названия)",
                    "created_at": _iso(doc.get("created_at")),
                    "updated_at": _iso(doc.get("updated_at")),
                    "message_count": int(doc.get("message_count", 0) or 0),
                    "age_days": round(age_days, 1) if age_days is not None else None,
                })
            return out
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"Ошибка чтения чатов: {exc}"
            return []

    def stats(self, older_than_days: float = 1.0) -> Dict[str, Any]:
        """Сводка: всего чатов/сообщений и сколько считается «старыми»."""
        db = self._ensure_db()
        if db is None:
            return {
                "available": False,
                "error": self.last_error,
                "total_chats": 0,
                "total_messages": 0,
                "old_chats": 0,
                "older_than_days": older_than_days,
            }
        try:
            total_chats = db[CHATS_COLLECTION].count_documents({})
            total_messages = db[MESSAGES_COLLECTION].count_documents({})
            cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
            old_chats = db[CHATS_COLLECTION].count_documents(
                {"updated_at": {"$lt": cutoff}}
            )
            return {
                "available": True,
                "error": None,
                "total_chats": total_chats,
                "total_messages": total_messages,
                "old_chats": old_chats,
                "older_than_days": older_than_days,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "available": False,
                "error": str(exc),
                "total_chats": 0,
                "total_messages": 0,
                "old_chats": 0,
                "older_than_days": older_than_days,
            }

    # ------------------------------------------------------------------
    # Удаление
    # ------------------------------------------------------------------
    def delete(self, chat_ids: List[str]) -> int:
        """Удаляет указанные чаты вместе с их сообщениями. Возвращает число чатов."""
        if not chat_ids:
            return 0
        db = self._ensure_db()
        if db is None:
            return 0
        try:
            db[MESSAGES_COLLECTION].delete_many({"chat_id": {"$in": list(chat_ids)}})
            res = db[CHATS_COLLECTION].delete_many({"chat_id": {"$in": list(chat_ids)}})
            return int(res.deleted_count)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"Ошибка удаления: {exc}"
            return 0

    def delete_older_than(
        self, days: float, keep_chat_id: Optional[str] = None
    ) -> int:
        """
        Удаляет чаты, не обновлявшиеся дольше ``days`` дней.

        ``keep_chat_id`` — чат, который удалять нельзя (текущий активный).
        Возвращает число удалённых чатов.
        """
        db = self._ensure_db()
        if db is None:
            return 0
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=float(days))
            query: Dict[str, Any] = {"updated_at": {"$lt": cutoff}}
            if keep_chat_id:
                query["chat_id"] = {"$ne": keep_chat_id}

            ids = [d["chat_id"] for d in db[CHATS_COLLECTION].find(query, {"chat_id": 1})]
            if not ids:
                return 0
            db[MESSAGES_COLLECTION].delete_many({"chat_id": {"$in": ids}})
            res = db[CHATS_COLLECTION].delete_many({"chat_id": {"$in": ids}})
            return int(res.deleted_count)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"Ошибка удаления старых чатов: {exc}"
            return 0

    def delete_all(self, keep_chat_id: Optional[str] = None) -> int:
        """Удаляет ВСЕ чаты (кроме ``keep_chat_id``, если он задан)."""
        db = self._ensure_db()
        if db is None:
            return 0
        try:
            query: Dict[str, Any] = {}
            if keep_chat_id:
                query["chat_id"] = {"$ne": keep_chat_id}
            ids = [d["chat_id"] for d in db[CHATS_COLLECTION].find(query, {"chat_id": 1})]
            if not ids:
                return 0
            db[MESSAGES_COLLECTION].delete_many({"chat_id": {"$in": ids}})
            res = db[CHATS_COLLECTION].delete_many({"chat_id": {"$in": ids}})
            return int(res.deleted_count)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"Ошибка полной очистки: {exc}"
            return 0


def _iso(value: Any) -> Optional[str]:
    """Аккуратно приводит дату к ISO-строке (Mongo отдаёт datetime)."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat(timespec="seconds")
    if value is None:
        return None
    return str(value)
