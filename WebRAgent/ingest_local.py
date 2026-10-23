#!/usr/bin/env python3
"""
ingest_local.py
===============
Заливка базы знаний НАПРЯМУЮ (без HTTP), запускается ВНУТРИ контейнера app.

Зачем это нужно
---------------
Штатный путь (``kb_ingest.py``) загружает файлы в WebRAgent по HTTP. В этом
окружении VPN-клиент обрезает крупные HTTP-запросы: файлы больше ~4 КБ
перестают доходить целиком и заливка «висит». Поэтому здесь мы обходим сеть
и вызываем сервисы приложения напрямую в процессе:

    читаем файл → module-level chunk_text → эмбеддинг → вставка в Qdrant

Так заливается вся база (включая большие таблицы), без единого HTTP-запроса.

Запуск (из корня проекта):
    docker cp knowledge_base/rtu_mirea_llm_kb <APP>:/tmp/kb
    docker cp WebRAgent/ingest_local.py  <APP>:/tmp/ingest_local.py
    docker exec <APP> python /tmp/ingest_local.py --kb /tmp/kb --collection МИРЭА
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from typing import List, Optional, Tuple

sys.path.insert(0, "/app")

import warnings

warnings.filterwarnings("ignore")

import requests  # noqa: E402

from app.models.collection import Collection          # noqa: E402
from app.models.document import Document              # noqa: E402
from app.services.document_service import DocumentService  # noqa: E402
from app.services.qdrant_service import QdrantService, get_sentence_transformer  # noqa: E402

# Адрес Qdrant. ВАЖНО: по умолчанию берём НЕ localhost, а 0.0.0.0/имя хоста,
# потому что в этом окружении VPN перехватывает loopback и обрезает крупные
# HTTP-запросы (>16 КБ). Через сетевой интерфейс большие батчи проходят
# мгновенно (проверено: 450 КБ → 200 OK за 0.006с).
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


def _pick_qdrant_url() -> str:
    """Возвращает рабочий адрес Qdrant, обходя loopback (VPN обрезает >16КБ)."""
    cand = []
    env = os.getenv("QDRANT_URL")
    if env:
        cand.append(env)
    # LAN-адреса хоста — ПЕРВЫМИ: только они пропускают большие запросы
    # (loopback режется VPN на 16 КБ).
    try:
        import socket
        host = socket.gethostname()
        for ip in socket.gethostbyname_ex(host)[2]:
            if not ip.startswith("127."):
                cand.append(f"http://{ip}:6333")
    except Exception:  # noqa: BLE001
        pass
    cand.append("http://localhost:6333")
    # docker bridge gateway — тоже часто доступен
    cand.append("http://172.17.0.1:6333")
    for url in cand:
        try:
            r = requests.get(f"{url}/collections", timeout=3)
            if r.status_code == 200:
                # Проверяем, что БОЛЬШОЙ запрос тоже проходит (не только GET).
                # Проба нарочно большая (>16 КБ): так мы отсеиваем loopback,
                # на котором VPN обрезает крупные тела запросов.
                probe = {"points": [{"id": f"00000000-0000-0000-0000-0000000000{i:02d}",
                                     "vector": [0.0] * 384,
                                     "payload": {"t": "x" * 3000}}
                                    for i in range(8)]}
                rr = requests.post(
                    f"{url}/collections/__probe__/points", json=probe, timeout=5
                )
                # 404 (нет коллекции) тоже годится — важно, что тело дошло.
                if rr.status_code < 500:
                    return url
        except Exception:  # noqa: BLE001
            continue
    return cand[0]

# Порядок загрузки: сначала самое полезное для поиска, затем всё остальное
# (CSV-таблицы, changelog и т.п.) — чтобы из датасета попало МАКСИМУМ данных.
INGEST_PLAN: List[Tuple[str, str]] = [
    ("chunks", "glob"),                     # 80 фрагментов — «золото» для поиска
    ("data/faq.jsonl", "file"),             # пары «вопрос → ответ»
    ("data/all_tables.jsonl", "file"),      # строки таблиц как JSON
    ("rtu_mirea_key_facts.md", "file"),     # выжимка главных фактов
    ("rtu_mirea_faq.md", "file"),
    # ДОКУМЕНТЫ .docx с ПРАВИЛЬНОЙ структурой (Heading-стили): из них получаются
    # заголовки разделов, а именно по ним работает ранжирование (см.
    # document_service). Ставятся ВЫШЕ markdown-сборки и CSV.
    ("docx", "glob_docx"),
    ("sources.md", "file"),
    ("README.md", "file"),
    ("changelog.md", "file"),
    ("data/tables", "glob_csv"),            # 68 таблиц разделов
    ("data", "glob_csv"),                   # 15 сводных CSV таблиц
    ("rtu_mirea_kb_full.md", "file"),       # полный документ (большой) — в конце
]

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
CHUNK_STRATEGY = "paragraph"


def _csv_to_text(path: str) -> str:
    """
    Превращает CSV в читаемый текст для эмбеддинга.

    Формат: на каждую строку — «<Первая колонка>: <колонка>=<значение>, ...».
    Так модель видит и заголовки, и значения, и связку между ними (обычный
    CSV без заголовков в каждой строке плохо ищется).
    """
    import csv

    try:
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as fh:
            sample = fh.read(4096)
            fh.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except Exception:  # noqa: BLE001
                dialect = csv.excel
            reader = csv.reader(fh, dialect)
            rows = list(reader)
    except Exception as exc:  # noqa: BLE001
        return f""

    if not rows:
        return ""

    title = os.path.basename(path)
    header = [h.strip() for h in rows[0]]
    lines = [f"# Таблица: {title}", " | ".join(header)]
    for row in rows[1:]:
        parts = []
        for i, cell in enumerate(row):
            key = header[i] if i < len(header) else f"col{i + 1}"
            val = (cell or "").strip()
            if val:
                parts.append(f"{key}={val}")
        if parts:
            lines.append("; ".join(parts))
    return "\n".join(lines)


def _jsonl_to_text(path: str) -> str:
    """Превращает JSONL в текст: по строке на запись, «ключ: значение»."""
    out = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if isinstance(obj, dict):
                    out.append("; ".join(f"{k}: {v}" for k, v in obj.items()))
                else:
                    out.append(str(obj))
    except Exception:  # noqa: BLE001
        return ""
    return "\n".join(out)


def _docx_to_text(path: str) -> str:
    """
    Превращает .docx в markdown-подобный текст с СОХРАНЕНИЕМ СТРУКТУРЫ.

    Почему не просто ``"\n".join(p.text)``: в этих документах тема раздела
    задаётся стилями («Heading 1», «Heading 2», «Title»). Именно по заголовку
    ранжирование понимает, о чём чанк («9.1. Общежития: шесть корпусов»).
    Если заголовки превратить в обычный текст, поиск теряет главный признак
    и раздаёт нерелевантные фрагменты.

    ВАЖНО: обходим ``document.element.body``, а не ``document.paragraphs``:
    атрибут ``paragraphs`` не содержит таблиц и НЕ сохраняет порядок — таблица
    отдельно, текст отдельно. Здесь же всё идёт в том порядке, в котором стоит
    в документе.
    """
    try:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️ python-docx недоступен ({exc}) — пропускаю {path}")
        return ""

    def style_level(name: str) -> int:
        """Уровень заголовка по имени стиля: 1..6, 0 — не заголовок."""
        low = (name or "").lower()
        if low.startswith("title"):
            return 1
        if low.startswith("heading"):
            digits = "".join(ch for ch in low if ch.isdigit())
            try:
                return max(1, min(6, int(digits))) if digits else 1
            except ValueError:
                return 1
        return 0

    try:
        doc = Document(path)
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠️ Не удалось открыть {path}: {exc}")
        return ""

    title = os.path.basename(path)
    out: List[str] = [f"# Документ: {title}"]

    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[-1]

        if tag == "p":
            para = Paragraph(child, doc)
            text = (para.text or "").strip()
            if not text:
                continue
            level = style_level(getattr(para.style, "name", ""))
            if level:
                # Заголовок — отдельной строкой с решётками, чтобы чанкер
                # начинал с него новый фрагмент и он попал в заголовок чанка.
                out.append(f"{'#' * level} {text}")
            else:
                out.append(text)
            # ПУСТАЯ СТРОКА между блоками — обязательно.
            # Чанкер делит текст по `\n\s*\n` (граница абзаца). Если склеить все
            # блоки одиночным `\n`, весь документ на 750 тыс. символов станет
            # ОДНИМ «абзацем» и превратится в единственный чанк — база получит
            # один бесполезный фрагмент вместо сотен разделов (именно так
            # и случилось: из DOCX пришёл 1 чанк вместо сотен).
            out.append("")

        elif tag == "tbl":
            table = Table(child, doc)
            rows = []
            for row in table.rows:
                cells = [(c.text or "").strip().replace("\n", " ")
                         for c in row.cells]
                if any(cells):
                    rows.append(" | ".join(cells))
            if not rows:
                continue
            # Первая строка — шапка столбцов, вторая — разделитель markdown.
            header = rows[0]
            out.append(f"# Таблица: {title}")
            out.append(header)
            out.append(" | ".join("---" for _ in header.split(" | ")))
            out.extend(rows[1:])
            # Пустая строка — граница абзаца для чанкера (см. пояснение выше).
            out.append("")


    return "\n".join(out)


def read_as_text(path: str) -> str:
    """Читает файл любого поддерживаемого формата как текст для эмбеддинга."""
    low = path.lower()
    if low.endswith(".csv"):
        return _csv_to_text(path)
    if low.endswith(".jsonl"):
        return _jsonl_to_text(path)
    if low.endswith(".docx"):
        return _docx_to_text(path)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except Exception:  # noqa: BLE001
        return ""


def collect_files(kb_root: str) -> List[Tuple[str, str]]:
    """Собирает (путь, отображаемое_имя) в порядке INGEST_PLAN."""
    import glob as globmod

    files: List[Tuple[str, str]] = []
    seen = set()
    for rel, kind in INGEST_PLAN:
        path = os.path.join(kb_root, rel)
        if kind in ("glob", "glob_csv", "glob_docx"):
            if not os.path.isdir(path):
                continue
            pats = {"glob": ["*.md"], "glob_csv": ["*.csv"],
                    "glob_docx": ["*.docx"]}[kind]
            found = []
            for pat in pats:
                for fp in sorted(globmod.glob(os.path.join(path, pat))):
                    if fp not in seen:
                        found.append(fp)
            # ДЕДУПЛИКАЦИЯ ТОМОВ.
            # «РТУ_МИРЭА_база_знаний_расширенная.docx» — это ПОЛНАЯ база,
            # включающая разделы 1–24 из всех трёх томов. Если залить и её,
            # и тома, каждый факт окажется в базе 2–3 раза: на поиск придут
            # почти одинаковые чанки, и ранжирование будет выбирать между
            # копиями вместо разных источников. Поэтому при наличии
            # «расширенной» версии заливаем ТОЛЬКО её.
            if kind == "glob_docx" and len(found) > 1:
                merged = [f for f in found if "расширен" in os.path.basename(f).lower()]
                if merged:
                    print(f"  ℹ️ Найдена объединённая база — тома пропущены "
                          f"({len(found) - len(merged)} шт.)")
                    found = merged
            for fp in found:
                seen.add(fp)
                files.append((fp, os.path.relpath(fp, kb_root)))
        elif os.path.exists(path):
            if path in seen:
                continue
            seen.add(path)
            files.append((path, rel))
    return files


def find_collection(name: str) -> Optional[str]:
    """Ищет коллекцию по имени (точное или частичное совпадение)."""
    for col in Collection.get_all():
        if col.name and col.name.strip().lower() == name.strip().lower():
            return col.id
    for col in Collection.get_all():
        if col.name and name.strip().lower() in col.name.strip().lower():
            return col.id
    return None


def rest_create_collection(name: str, dim: int = 384) -> bool:
    """
    Создаёт коллекцию через ПРЯМОЙ REST-запрос к Qdrant.

    Почему не через библиотеку: qdrant-client 1.13 несовместим с сервером
    Qdrant 1.19 — upsert'ы возвращают 400 после долгого ожидания. Прямой REST
    работает мгновенно (проверено: 0.003с вместо 106с).
    """
    try:
        r = requests.put(
            f"{QDRANT_URL}/collections/{name}",
            json={"vectors": {"size": dim, "distance": "Cosine"}},
            timeout=30,
        )
        return r.status_code in (200, 201)
    except Exception as exc:  # noqa: BLE001
        print(f"      ❌ create_collection: {exc}")
        return False


def rest_delete_collection(name: str) -> bool:
    """Удаляет коллекцию через прямой REST."""
    try:
        r = requests.delete(f"{QDRANT_URL}/collections/{name}", timeout=30)
        return r.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def rest_upsert_batch(name: str, vectors, metadatas) -> int:
    """
    Пакетная вставка векторов через ПРЯМОЙ REST (обходит сломанный клиент).

    Returns: число вставленных точек (0 при ошибке).
    """
    import uuid as _uuid

    points = []
    for vec, meta in zip(vectors, metadatas):
        points.append({
            "id": str(_uuid.uuid4()),
            "vector": vec,
            "payload": meta,
        })
    try:
        r = requests.put(
            f"{QDRANT_URL}/collections/{name}/points?wait=true",
            json={"points": points},
            timeout=120,
        )
        if r.status_code != 200:
            print(f"      ⚠️ upsert {r.status_code}: {r.text[:120]}")
            return 0
        return len(points)
    except Exception as exc:  # noqa: BLE001
        print(f"      ❌ upsert: {exc}")
        return 0


def ingest_file(ds: DocumentService, qdrant: QdrantService,
                collection_id: str, path: str, display_name: str,
                provider: str, model: str) -> int:
    """
    Заливает ОДИН файл: читает текст, режет на чанки, эмбеддит, кладёт в Qdrant.

    Эмбеддинги считаем локальной моделью (пачками), а запись делаем прямым
    REST-запросом — так надёжно и быстро.

    Возвращает число вставленных чанков (0 при ошибке).
    """
    try:
        text = read_as_text(path)
    except Exception as exc:  # noqa: BLE001
        print(f"      ❌ чтение: {exc}")
        return 0

    if not text.strip():
        return 0

    title = display_name.replace("/", " / ").replace("\\", " / ")
    doc = Document(title=title, collection_id=collection_id,
                   file_path=path, metadata={})
    doc.save()

    chunks = ds.chunk_text(
        text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP,
        strategy=CHUNK_STRATEGY,
    )
    if not chunks:
        return 0

    # Общая модель эмбеддингов (из кэша процесса) — считаем вектора пачками.
    st = get_sentence_transformer(model)
    vectors = st.encode(chunks, batch_size=32, show_progress_bar=False).tolist()

    metadatas = []
    for i, chunk in enumerate(chunks):
        metadatas.append({
            "document_id": doc.id,
            "document_title": title,
            "chunk_index": i,
            "original_chunk_id": f"{doc.id}_{i}",
            "content": chunk,
            "file_path": path,
            "chunk_strategy": CHUNK_STRATEGY,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "page_number": i + 1,
            "embedding_provider": provider,
            "embedding_model": model,
        })

    inserted = 0
    BATCH = 128
    for start in range(0, len(chunks), BATCH):
        bv = vectors[start:start + BATCH]
        bm = metadatas[start:start + BATCH]
        for attempt in range(3):
            n = rest_upsert_batch(collection_id, bv, bm)
            if n:
                inserted += n
                break
            time.sleep(0.5 * (attempt + 1))

    doc.metadata["chunk_count"] = len(chunks)
    doc.save()
    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(description="Прямая заливка базы знаний в Qdrant")
    parser.add_argument("--kb", default="/tmp/kb", help="Папка базы знаний в контейнере")
    parser.add_argument("--collection", default="МИРЭА", help="Имя коллекции")
    parser.add_argument("--reset", action="store_true", help="Очистить коллекцию перед заливкой")
    args = parser.parse_args()

    global QDRANT_URL
    QDRANT_URL = _pick_qdrant_url()
    print(f"🔌 Qdrant: {QDRANT_URL}")

    files = collect_files(args.kb)
    if not files:
        print(f"❌ Не найдено файлов в {args.kb}")
        return 1
    print(f"📚 Файлов к заливке: {len(files)}")

    collection_id = find_collection(args.collection)
    if collection_id and args.reset:
        print(f"♻️  Очищаю коллекцию {collection_id}...")
        rest_delete_collection(collection_id)
        collection_id = None

    if args.reset:
        # Удаляем и МЕТАДАННЫЕ старых коллекций с тем же именем.
        #
        # Без этого после каждой перезаливки в списке коллекций остаются
        # «мёртвые» записи: сами данные в Qdrant уже удалены, а JSON-описания
        # живут. Ассистент ищет коллекцию ПО ИМЕНИ и мог взять такую запись —
        # тогда поиск возвращает пустоту («найдено 0»), хотя база залита.
        # Проверено на практике: накопилось три записи «МИРЭА», живая — одна.
        live = set()
        try:
            import requests as _rq

            resp = _rq.get(f"{QDRANT_URL}/collections", timeout=10)
            if resp.status_code == 200:
                for item in (resp.json().get("result", {}).get("collections") or []):
                    live.add(item.get("name"))
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️ Не удалось получить список коллекций Qdrant: {exc}")

        removed = 0
        for col in Collection.get_all():
            if col.id in live:
                continue
            try:
                from app.models.document import Document

                for doc in list(Document.get_all(collection_id=col.id)):
                    doc.delete()
            except Exception:  # noqa: BLE001
                pass
            col.delete()
            removed += 1
        if removed:
            print(f"  🧹 Удалено мёртвых записей коллекций: {removed}")

    ds = DocumentService()
    qdrant = ds.qdrant_service  # нужен только для chunk_text/совместимости

    if not collection_id:
        collection_id = str(uuid.uuid4())
        col = Collection(id=collection_id, name=args.collection,
                         description="База знаний РТУ МИРЭА",
                         embedding_provider="embedding",
                         embedding_model="all-MiniLM-L6-v2",
                         embedding_dimensions=384)
        col.save()
        rest_create_collection(collection_id, dim=384)
        print(f"📁 Создана коллекция {collection_id}")
    else:
        print(f"📁 Использую коллекцию {collection_id}")

    provider, model = "embedding", "all-MiniLM-L6-v2"
    total_chunks, ok, failed = 0, 0, 0
    t_all = time.time()
    for i, (path, name) in enumerate(files, 1):
        t0 = time.time()
        n = ingest_file(ds, qdrant, collection_id, path, name, provider, model)
        dt = time.time() - t0
        if n:
            ok += 1
            total_chunks += n
            print(f"  [{i}/{len(files)}] {name[:60]} → {n} чанков ({dt:.1f}с)")
        else:
            failed += 1
            print(f"  [{i}/{len(files)}] {name[:60]} ❌ ({dt:.1f}с)")

    print("\n" + "=" * 60)
    print(f"  Готово за {time.time()-t_all:.0f}с: файлов {ok}, ошибок {failed}, "
          f"чанков {total_chunks}")
    print(f"  Коллекция: {collection_id}")
    print("=" * 60)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
