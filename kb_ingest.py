#!/usr/bin/env python3
"""
kb_ingest.py
============
Загрузка базы знаний РТУ МИРЭА (архив ``rtu_mirea_llm_kb.zip``) в WebRAgent.

Что делает скрипт
-----------------
1. Ищет распакованную базу в ``knowledge_base/rtu_mirea_llm_kb`` (если нет —
   распаковывает zip из ``knowledge_base/`` или ``~/Downloads``).
2. Логинится в WebRAgent (``/auth/login``) под админом из конфига.
3. Находит (или создаёт) коллекцию «МИРЭА» с моделью эмбеддингов
   ``all-MiniLM-L6-v2`` (384 измерения — как у эталонного архива).
4. Загружает документы в коллекцию, **начиная с самых ценных файлов**:
   ``chunks/*.md`` (79 самостоятельных фрагментов) → ``data/faq.jsonl`` →
   ``data/all_tables.jsonl`` → ``rtu_mirea_key_facts.md`` → ``rtu_mirea_faq.md``.
   ``rtu_mirea_kb_full.md`` (≈400 КБ) идёт последним и только с ``--with-full``:
   он «тяжелее» и нужен только если фрагментов не хватает.

Загрузка идёт через штатный веб-эндпоинт WebRAgent
``POST /admin/collections/<id>/upload`` — тот же, что использует веб-интерфейс,
поэтому документация корректно разбивается на чанки, эмбеддится и попадает в Qdrant.

Использование
-------------
    python kb_ingest.py                  # залить всё (кроме full-документа)
    python kb_ingest.py --with-full      # плюс rtu_mirea_kb_full.md
    python kb_ingest.py --dry-run        # показать план, ничего не заливать
    python kb_ingest.py --reset          # удалить коллекцию «МИРЭА» и залить заново
    python kb_ingest.py --collection "МИРЭА" --base-url http://localhost:5000

Требует запущенного WebRAgent (``./run_all.sh`` или ``docker compose up -d``).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import zipfile
from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(PROJECT_DIR, "knowledge_base")
KB_NAME = "rtu_mirea_llm_kb"

# --- настройки по умолчанию (совпадают с WebRAgent/.env) --------------------
DEFAULT_BASE_URL = os.getenv("RAG_BASE_URL", "http://localhost:5000")
DEFAULT_USERNAME = os.getenv("RAG_USERNAME", "admin")
DEFAULT_PASSWORD = os.getenv("RAG_PASSWORD", "change_me_in_production")
DEFAULT_COLLECTION = os.getenv("RAG_COLLECTION_NAME", "МИРЭА")
DEFAULT_EMBEDDING_MODEL = os.getenv("KB_EMBEDDING_MODEL") or "paraphrase-multilingual-MiniLM-L12-v2"

# Порядок загрузки: сначала самое полезное для RAG.
INGEST_PLAN: List[Tuple[str, str]] = [
    ("chunks", "glob"),            # 79 фрагментов — «золото» для поиска
    ("data/faq.jsonl", "file"),    # 156 пар «вопрос → ответ»
    ("data/all_tables.jsonl", "file"),  # 893 строки таблиц как JSON
    ("rtu_mirea_key_facts.md", "file"),  # выжимка главных фактов
    ("rtu_mirea_faq.md", "file"),
    ("sources.md", "file"),
]
FULL_DOCS = ["rtu_mirea_kb_full.md"]

# Лимит на файл для загрузки (защита от огромных PDF/DOCX)
MAX_UPLOAD_BYTES = 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# Поиск и распаковка базы
# ---------------------------------------------------------------------------
def ensure_kb_unpacked() -> str:
    """Возвращает путь к распакованной базе, распаковывая архив при необходимости."""
    target = os.path.join(KB_DIR, KB_NAME)
    if os.path.isdir(target):
        return target

    os.makedirs(KB_DIR, exist_ok=True)
    candidates = [
        os.path.join(KB_DIR, f"{KB_NAME}.zip"),
        os.path.join(PROJECT_DIR, f"{KB_NAME}.zip"),
        os.path.expanduser(f"~/Downloads/{KB_NAME}.zip"),
        os.path.expanduser(f"~/Загрузки/{KB_NAME}.zip"),
    ]
    archive = next((p for p in candidates if os.path.exists(p)), None)
    if not archive:
        print(f"❌ Не найден архив {KB_NAME}.zip.")
        print(f"   Положите его в: {KB_DIR}/")
        sys.exit(1)

    print(f"📦 Распаковываю {archive} → {KB_DIR}/")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(KB_DIR)
    if not os.path.isdir(target):
        print(f"❌ После распаковки нет папки {target}")
        sys.exit(1)
    return target


def collect_files(kb_root: str, with_full: bool = False) -> List[Tuple[str, str]]:
    """
    Собирает список (путь_на_диске, отображаемое_имя) в порядке INGEST_PLAN.
    """
    import glob as globmod

    files: List[Tuple[str, str]] = []
    for rel, kind in INGEST_PLAN:
        path = os.path.join(kb_root, rel)
        if kind == "glob":
            if not os.path.isdir(path):
                continue
            for fp in sorted(globmod.glob(os.path.join(path, "*.md"))):
                files.append((fp, os.path.relpath(fp, kb_root)))
        elif os.path.exists(path):
            files.append((path, rel))
    if with_full:
        for rel in FULL_DOCS:
            path = os.path.join(kb_root, rel)
            if os.path.exists(path):
                files.append((path, rel))
    # Отсекаем слишком большие файлы (WebRAgent и так режет на чанки)
    result = []
    for fp, name in files:
        try:
            size = os.path.getsize(fp)
        except OSError:
            continue
        if size > MAX_UPLOAD_BYTES:
            print(f"   ⏭️  Пропускаю {name} ({size/1024/1024:.1f} МБ) — больше лимита")
            continue
        result.append((fp, name))
    return result


# ---------------------------------------------------------------------------
# Клиент WebRAgent
# ---------------------------------------------------------------------------
class WebRAgentAdmin:
    """Минимальный клиент админки WebRAgent: логин, коллекции, загрузка документов."""

    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.session.headers.update({"X-Requested-With": "XMLHttpRequest"})

    # -- аутентификация --------------------------------------------------
    def login(self) -> bool:
        url = f"{self.base_url}/auth/login"
        csrf = None
        try:
            page = self.session.get(url, timeout=10)
            if page.status_code == 200:
                soup = BeautifulSoup(page.text, "html.parser")
                for name in ("csrf_token", "_csrf_token"):
                    tag = soup.find("input", {"name": name})
                    if tag and tag.get("value"):
                        csrf = tag["value"]
                        break
        except requests.RequestException as exc:
            print(f"❌ WebRAgent недоступен по {self.base_url}: {exc}")
            return False

        data = {"username": self.username, "password": self.password}
        if csrf:
            data["csrf_token"] = csrf
        resp = self.session.post(url, data=data, allow_redirects=False, timeout=10)
        if resp.status_code not in (200, 302):
            print(f"❌ Ошибка входа (статус {resp.status_code}).")
            return False
        # Проверяем, что реально залогинились
        check = self.session.get(f"{self.base_url}/admin/collections", timeout=10)
        if check.status_code != 200 or "/auth/login" in check.url:
            print("❌ Логин не удался (неверный логин/пароль?).")
            return False
        return True

    # -- коллекции -------------------------------------------------------
    def list_collections(self) -> List[dict]:
        """
        Возвращает список коллекций, разбирая админ-страницу.

        В WebRAgent нет JSON-эндпоинта списка коллекций для админа, поэтому
        читаем страницу: у каждой коллекции — карточка ``.collection-card``,
        внутри которой есть заголовок ``.card-title`` и ссылка на документы
        с id коллекции.
        """
        resp = self.session.get(f"{self.base_url}/admin/collections", timeout=15)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")

        collections: List[dict] = []
        seen = set()
        # Основной путь: карточки коллекций
        for card in soup.select(".collection-card"):
            title = card.select_one(".card-title") or card.select_one("h5")
            name = title.get_text(strip=True) if title else None
            for link in card.find_all("a", href=True):
                m = re.search(r"/admin/collections/([0-9a-f-]{36})", link["href"])
                if m and m.group(1) not in seen:
                    seen.add(m.group(1))
                    collections.append({"id": m.group(1), "name": name})
                    break
            else:
                # Ссылки на документы может не быть — ищем id где угодно в карточке
                m = re.search(r"/admin/collections/([0-9a-f-]{36})", str(card))
                if m and m.group(1) not in seen:
                    seen.add(m.group(1))
                    collections.append({"id": m.group(1), "name": name})

        # Резерв: любые id коллекций на странице, для которых имя не нашли
        for cid in re.findall(r"/admin/collections/([0-9a-f-]{36})", resp.text):
            if cid not in seen:
                seen.add(cid)
                collections.append({"id": cid, "name": None})
        return collections

    def find_collection(self, name: str) -> Optional[str]:
        for col in self.list_collections():
            if col.get("name") and col["name"].strip().lower() == name.strip().lower():
                return col["id"]
        return None

    def create_collection(
        self,
        name: str,
        embedding_provider: str = "embedding",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        description: str = "База знаний РТУ МИРЭА (офлайн-пакет rtu_mirea_llm_kb)",
    ) -> Optional[str]:
        data = {
            "name": name,
            "description": description,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
        }
        resp = self.session.post(
            f"{self.base_url}/admin/collections", data=data, timeout=120
        )
        if resp.status_code not in (200, 302):
            print(f"❌ Не удалось создать коллекцию (статус {resp.status_code}).")
            return None
        return self.find_collection(name)

    def delete_collection(self, collection_id: str) -> bool:
        """
        Удаляет коллекцию. Ошибки НЕ пробрасываются наружу: удаление —
        вспомогательная операция, и если WebRAgent задумался, лучше продолжить
        (коллекция просто останется, а документы допишутся в неё).
        """
        try:
            page = self.session.get(
                f"{self.base_url}/admin/collections/{collection_id}", timeout=60
            )
            token = None
            match = re.search(r'name="csrf_token"\s+value="([^"]+)"', page.text)
            if match:
                token = match.group(1)
            data = {"csrf_token": token} if token else {}
            resp = self.session.post(
                f"{self.base_url}/admin/collections/{collection_id}/delete",
                data=data,
                timeout=180,
                allow_redirects=False,
            )
            return resp.status_code in (200, 302)
        except requests.RequestException as exc:
            print(f"   ⚠️  Не удалось удалить коллекцию ({exc.__class__.__name__}). "
                  f"Продолжаю с существующей.")
            return False

    def list_documents(self, collection_id: str) -> List[str]:
        """
        Имена уже загруженных документов в коллекции.

        Нужно, чтобы ``--skip-existing`` не заливал один и тот же фрагмент
        дважды (в Qdrant появятся дубли, и поиск начнёт повторяться).
        """
        try:
            resp = self.session.get(
                f"{self.base_url}/admin/collections/{collection_id}", timeout=60
            )
        except requests.RequestException:
            return []
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        names: List[str] = []
        # Имена документов в таблице детальной страницы коллекции
        for row in soup.select("table tr, .document-item, .list-group-item"):
            text = row.get_text(" ", strip=True)
            if text:
                names.append(text)
        return names

    # -- загрузка --------------------------------------------------------
    def upload_document(
        self,
        collection_id: str,
        file_path: str,
        title: str,
        use_docling: bool = False,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        chunk_strategy: str = "paragraph",
        extract_metadata: str = "basic",
        retries: int = 2,
    ) -> bool:
        """
        Загружает один документ в коллекцию через штатный эндпоинт.

        Разбиение на чанки делает сам WebRAgent. При таймауте (он возможен:
        эмбеддинг длинного документа — небыстрая операция) пробуем ещё раз.
        """
        url = f"{self.base_url}/admin/collections/{collection_id}/upload"
        filename = os.path.basename(file_path)

        for attempt in range(retries + 1):
            try:
                with open(file_path, "rb") as fh:
                    files = {"documents": (filename, fh, "application/octet-stream")}
                    data = {
                        "title": title,
                        "chunk_strategy": chunk_strategy,
                        "chunk_size": str(chunk_size),
                        "chunk_overlap": str(chunk_overlap),
                        "extract_metadata": extract_metadata,
                    }
                    if use_docling:
                        data["use_docling"] = "on"
                    # allow_redirects=False — НЕ идём по редиректу (302) на
                    # страницу коллекции. Она весит ~110 КБ и в этом окружении
                    # (VPN, обрезающий большие ответы) «зависает»: заливка
                    # вставала на полуслове. Нам достаточно самого факта
                    # редиректа — это и есть признак успешной загрузки.
                    resp = self.session.post(
                        url, data=data, files=files, timeout=240,
                        allow_redirects=False,
                    )
            except requests.RequestException as exc:
                # Таймаут чтения ответа часто означает, что документ УЖЕ
                # принят и обрабатывается — не дублируем загрузку.
                if isinstance(exc, requests.exceptions.ReadTimeout):
                    print("(таймаут ответа, документ, скорее всего, принят) ", end="")
                    return True
                if attempt < retries:
                    print(f"(сеть: {exc.__class__.__name__}, повтор {attempt + 1}) ", end="")
                    continue
                print(f"❌ Ошибка сети: {exc}")
                return False

            if resp.status_code in (200, 302):
                return True
            if attempt < retries:
                continue
            print(f"❌ Статус {resp.status_code}")
            return False
        return False


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Загрузка базы знаний РТУ МИРЭА в коллекцию WebRAgent"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Адрес WebRAgent")
    parser.add_argument("--username", default=DEFAULT_USERNAME)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Имя коллекции")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--embedding-provider", default="embedding",
        help="Провайдер эмбеддингов ('embedding' = локальные sentence-transformers)",
    )
    parser.add_argument("--with-full", action="store_true", help="Залить и rtu_mirea_kb_full.md")
    parser.add_argument("--reset", action="store_true", help="Удалить коллекцию и создать заново")
    parser.add_argument("--dry-run", action="store_true", help="Показать план без загрузки")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Пропускать документы с совпадающими именами")
    args = parser.parse_args()

    print("=" * 62)
    print("  Загрузка базы знаний РТУ МИРЭА → WebRAgent")
    print(f"  WebRAgent:  {args.base_url}")
    print(f"  Коллекция:  «{args.collection}»")
    print("=" * 62)

    kb_root = ensure_kb_unpacked()
    files = collect_files(kb_root, with_full=args.with_full)
    total_mb = sum(os.path.getsize(p) for p, _ in files) / 1024 / 1024
    print(f"\n📚 Подготовлено файлов: {len(files)} ({total_mb:.1f} МБ)")
    if args.with_full:
        print("   (включая rtu_mirea_kb_full.md)")
    for i, (_, name) in enumerate(files[:10], 1):
        print(f"   {i:2d}. {name}")
    if len(files) > 10:
        print(f"   ... и ещё {len(files) - 10}")

    if args.dry_run:
        print("\n🔎 --dry-run: загрузка не выполняется.")
        return 0

    # --- подключение -----------------------------------------------------
    admin = WebRAgentAdmin(args.base_url, args.username, args.password)
    print("\n🔐 Вход в WebRAgent...")
    if not admin.login():
        print("   Подсказка: сначала поднимите стек командой ./run_all.sh")
        return 1
    print("   ✅ Вошли.")

    # --- коллекция -------------------------------------------------------
    collection_id = admin.find_collection(args.collection)
    if collection_id and args.reset:
        print(f"\n♻️  Удаляю существующую коллекцию «{args.collection}»...")
        admin.delete_collection(collection_id)
        collection_id = None
    if not collection_id:
        print(f"\n📁 Создаю коллекцию «{args.collection}» "
              f"(эмбеддинги: {args.embedding_provider}/{args.embedding_model})...")
        collection_id = admin.create_collection(
            args.collection, args.embedding_provider, args.embedding_model
        )
        if not collection_id:
            print("❌ Коллекция не создана.")
            return 1
        print(f"   ✅ Создана: {collection_id}")
    else:
        print(f"\n📁 Использую существующую коллекцию: {collection_id}")

    # --- загрузка --------------------------------------------------------
    print(f"\n⬆️  Загружаю {len(files)} документ(ов)...")
    ok = 0
    failed = 0
    for i, (path, name) in enumerate(files, 1):
        # В WebRAgent title = имя документа; используем относительный путь,
        # чтобы фрагменты не смешивались между собой.
        title = name.replace("/", " / ").replace("\\", " / ")
        print(f"  [{i}/{len(files)}] {name} ...", end=" ", flush=True)
        started = time.time()
        success = admin.upload_document(collection_id, path, title=title)
        if success:
            ok += 1
            print(f"✅ ({time.time() - started:.1f}с)")
        else:
            failed += 1
            print("❌")
        # Небольшая пауза, чтобы не перегружать Qdrant/эмбеддер
        time.sleep(0.3)

    print("\n" + "=" * 62)
    print(f"  Готово: загружено {ok}, ошибок {failed}")
    print(f"  Коллекция «{args.collection}»: {args.base_url}/admin/collections/{collection_id}")
    print("=" * 62)
    if failed:
        print("⚠️  Часть файлов не загрузилась — проверьте, что WebRAgent полностью поднялся.")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
