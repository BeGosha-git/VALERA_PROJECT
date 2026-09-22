"""Локальное JSON-хранилище с API, совместимым с pymongo коллекциями.

Зачем: WebRAgent исторически требует MongoDB (пользователи + история чатов).
На Jetson нет Docker, поэтому вместо поднятия сервера MongoDB используем файл на
диске. Классы ниже повторяют минимально необходимый интерфейс pymongo
(insert_one / find_one / find / update_one / delete_one / delete_many),
поэтому код сервисов менять не нужно — достаточно подставить этот клиент.

Если MongoDB доступен (MONGODB_URI и запущенный сервер) — используется он.
"""

import json
import os
import re
import threading
import uuid
from datetime import date, datetime
from pathlib import Path

try:  # ObjectId приходит из pymongo (bson); на Jetson он установлен
    from bson import ObjectId
except ImportError:  # pragma: no cover
    ObjectId = None


def _normalize(value):
    """Приводит значение к виду, пригодному для хранения в JSON."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if ObjectId is not None and isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value


def _match(doc, query):
    """Проверяет соответствие документа простому mongo-запросу."""
    for key, expected in (query or {}).items():
        if key.startswith("$"):
            continue  # логические операторы в этом приложении не используются
        actual = doc.get(key)
        if isinstance(expected, dict):
            # поддержка $in / $ne / $regex — минимум, который нужен
            for op, op_val in expected.items():
                if op == "$in":
                    if actual not in op_val and str(actual) not in [str(v) for v in op_val]:
                        return False
                elif op == "$ne":
                    if actual == op_val:
                        return False
                elif op == "$regex":
                    if not re.search(op_val, str(actual or "")):
                        return False
                else:
                    return False
        elif str(actual) != str(expected):
            return False
    return True


class LocalCursor:
    """Итератор по результатам find() с поддержкой sort/limit/skip."""

    def __init__(self, docs):
        self._docs = list(docs)

    def sort(self, key, direction=1):
        reverse = direction == -1
        self._docs.sort(key=lambda d: (d.get(key) is None, d.get(key)), reverse=reverse)
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    def skip(self, n):
        self._docs = self._docs[n:]
        return self

    def __iter__(self):
        return iter(self._docs)

    def __len__(self):
        return len(self._docs)


class InsertResult:
    def __init__(self, inserted_id):
        self.inserted_id = inserted_id


class UpdateResult:
    def __init__(self, matched, modified):
        self.matched_count = matched
        self.modified_count = modified


class DeleteResult:
    def __init__(self, deleted):
        self.deleted_count = deleted


class LocalCollection:
    """Коллекция документов, хранимая в JSON-файле."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        if not self.path.exists():
            self.path.write_text("[]", encoding="utf-8")

    # --- внутреннее ---------------------------------------------------------
    def _load(self):
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def _save(self, docs):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    # --- pymongo-подобный API ------------------------------------------------
    def insert_one(self, doc):
        with self._lock:
            docs = self._load()
            stored = _normalize(dict(doc))
            new_id = stored.get("_id")
            if new_id is None:
                new_id = str(ObjectId()) if ObjectId is not None else uuid.uuid4().hex
                stored["_id"] = new_id
            docs.append(stored)
            self._save(docs)
            return InsertResult(new_id)

    def find_one(self, query=None, projection=None):
        for doc in self._load():
            if _match(doc, query):
                return dict(doc)
        return None

    def find(self, query=None, projection=None):
        return LocalCursor([dict(d) for d in self._load() if _match(d, query)])

    def update_one(self, query, update, upsert=False):
        with self._lock:
            docs = self._load()
            for doc in docs:
                if _match(doc, query):
                    if "$set" in update:
                        for k, v in update["$set"].items():
                            doc[k] = _normalize(v)
                    if "$inc" in update:
                        for k, v in update["$inc"].items():
                            doc[k] = (doc.get(k) or 0) + v
                    if "$push" in update:
                        for k, v in update["$push"].items():
                            doc.setdefault(k, []).append(_normalize(v))
                    self._save(docs)
                    return UpdateResult(1, 1)
            if upsert:
                new_doc = dict(query or {})
                new_doc.update({k: _normalize(v) for k, v in (update.get("$set") or {}).items()})
                self.insert_one(new_doc)
                return UpdateResult(0, 1)
            return UpdateResult(0, 0)

    def delete_one(self, query):
        with self._lock:
            docs = self._load()
            for i, doc in enumerate(docs):
                if _match(doc, query):
                    docs.pop(i)
                    self._save(docs)
                    return DeleteResult(1)
            return DeleteResult(0)

    def delete_many(self, query):
        with self._lock:
            docs = self._load()
            kept = [d for d in docs if not _match(d, query)]
            self._save(kept)
            return DeleteResult(len(docs) - len(kept))

    def count_documents(self, query=None):
        return sum(1 for d in self._load() if _match(d, query))

    def create_index(self, *args, **kwargs):
        """Индексы не нужны — заглушка для совместимости."""
        return None

    def drop(self):
        self._save([])


class LocalDatabase:
    def __init__(self, base_dir: Path, name: str):
        self.base_dir = Path(base_dir) / name
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, collection_name):
        return LocalCollection(self.base_dir / f"{collection_name}.json")


class LocalClient:
    """Минимальная замена MongoClient, хранящая данные в каталоге."""

    def __init__(self, base_dir=None):
        self.base_dir = Path(
            base_dir or os.getenv("LOCAL_STORE_DIR", "data/local_store")
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, db_name):
        return LocalDatabase(self.base_dir, db_name)

    def close(self):
        return None


def mongo_available(uri=None, timeout_ms=1500):
    """Быстрая проверка доступности MongoDB (чтобы не ждать 30 с таймаута)."""
    try:
        from pymongo import MongoClient

        uri = uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
        client.admin.command("ping")
        return True
    except Exception:
        return False


def get_client(uri=None):
    """Возвращает MongoClient, если MongoDB доступен, иначе локальное хранилище."""
    uri = uri or os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    if os.getenv("FORCE_LOCAL_STORE", "").lower() in ("1", "true", "yes"):
        return LocalClient()
    if mongo_available(uri):
        from pymongo import MongoClient

        return MongoClient(uri)
    return LocalClient()
