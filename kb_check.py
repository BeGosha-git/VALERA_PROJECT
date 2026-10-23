#!/usr/bin/env python3
"""
kb_check.py
===========
Проверка полноты базы знаний «МИРЭА» в Qdrant перед стартом Валеры.

Сверяет файлы, которые ДОЛЖНЫ быть залиты (тот же план, что у
``WebRAgent/ingest_local.py``), с документами, которые РЕАЛЬНО лежат в
коллекции Qdrant. Использует только стандартную библиотеку.

Что проверяется
---------------
* коллекция «МИРЭА» существует и в ней есть точки;
* каждый ожидаемый файл загружен (по ``document_title`` в payload);
* в коллекции нет «лишних» документов, которых уже нет на диске;
* нет дублей (один документ залит дважды — поиск начинает повторяться);
* база не устарела: ни один файл не изменён позже заливки коллекции.

Коды выхода: 0 — всё полно; 1 — база неполная/устарела (нужна перезаливка);
2 — Qdrant недоступен.

Использование
-------------
    python3 kb_check.py            # проверка с отчётом
    python3 kb_check.py --quiet    # только итоговая строка
    python3 kb_check.py --json     # машинно-читаемый результат
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
KB_ROOT = os.path.join(PROJECT_DIR, "knowledge_base", "rtu_mirea_llm_kb")
COLLECTIONS_META = os.path.join(PROJECT_DIR, "WebRAgent", "data", "collections")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/")
COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "МИРЭА")
# Должна совпадать с DEFAULT_EMBEDDING_MODEL в WebRAgent/ingest_local.py.
EMBEDDING_MODEL = os.getenv("KB_EMBEDDING_MODEL") or "paraphrase-multilingual-MiniLM-L12-v2"

# План — копия INGEST_PLAN из WebRAgent/ingest_local.py (держать в синхроне).
INGEST_PLAN: List[Tuple[str, str]] = [
    ("chunks", "glob"),
    ("data/faq.jsonl", "file"),
    ("data/all_tables.jsonl", "file"),
    ("rtu_mirea_kb_full.md", "file"),
    ("rtu_mirea_key_facts.md", "file"),
    ("rtu_mirea_faq.md", "file"),
    ("docx", "glob_docx"),
    ("sources.md", "file"),
    ("data/tables", "glob_csv"),
    ("data", "glob_csv"),
]


def expected_documents() -> Dict[str, str]:
    """{отображаемое_имя_как_в_Qdrant: путь_на_диске}."""
    found: Dict[str, str] = {}
    for rel, kind in INGEST_PLAN:
        path = os.path.join(KB_ROOT, rel)
        if kind in ("glob", "glob_csv", "glob_docx"):
            if not os.path.isdir(path):
                continue
            pat = {"glob": "*.md", "glob_csv": "*.csv", "glob_docx": "*.docx"}[kind]
            files = sorted(glob.glob(os.path.join(path, pat)))
            if kind == "glob_docx" and len(files) > 1:
                merged = [f for f in files if "расширен" in os.path.basename(f).lower()]
                files = merged or files
            for fp in files:
                found[_title(os.path.relpath(fp, KB_ROOT))] = fp
        elif os.path.exists(path):
            found[_title(rel)] = path
    return found


def _title(rel: str) -> str:
    # Так название формирует ingest_local.ingest_file().
    return rel.replace("/", " / ").replace("\\", " / ")


# ---------------------------------------------------------------------------
# Qdrant (REST, stdlib)
# ---------------------------------------------------------------------------
def _request(method: str, path: str, body: Optional[dict] = None, timeout: int = 20):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{QDRANT_URL}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    # Без прокси: Qdrant локальный, а переменные HTTP_PROXY бывают заданы.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def live_collections() -> List[str]:
    res = _request("GET", "/collections")
    return [c["name"] for c in res.get("result", {}).get("collections", [])]


def points_count(collection: str) -> int:
    res = _request("POST", f"/collections/{collection}/points/count", {"exact": True})
    return int(res.get("result", {}).get("count", 0))


def titles_in_collection(collection: str) -> Counter:
    """Сколько чанков у каждого document_title (скролл только по нужным полям)."""
    counts: Counter = Counter()
    offset = None
    while True:
        body = {
            "limit": 256,
            "with_payload": {"include": ["document_title"]},
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset
        res = _request("POST", f"/collections/{collection}/points/scroll", body, timeout=60)
        result = res.get("result", {})
        for p in result.get("points", []):
            counts[(p.get("payload") or {}).get("document_title", "")] += 1
        offset = result.get("next_page_offset")
        if offset is None:
            break
    return counts


def pick_collection() -> Tuple[Optional[str], Optional[dict]]:
    """
    Находит рабочую коллекцию «МИРЭА»: запись в WebRAgent/data/collections
    с нужным именем, для которой реально есть коллекция в Qdrant.
    Если живых несколько — берём ту, где больше точек.
    """
    live = set(live_collections())
    best: Tuple[int, Optional[str], Optional[dict]] = (-1, None, None)
    for fp in glob.glob(os.path.join(COLLECTIONS_META, "*.json")):
        try:
            with open(fp, encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            continue
        if (meta.get("name") or "").strip().lower() != COLLECTION_NAME.lower():
            continue
        cid = meta.get("id")
        if cid in live:
            n = points_count(cid)
            if n > best[0]:
                best = (n, cid, meta)
    return best[1], best[2]


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Проверка полноты базы знаний МИРЭА")
    ap.add_argument("--quiet", action="store_true", help="Только итоговая строка")
    ap.add_argument("--json", action="store_true", help="Результат в JSON")
    args = ap.parse_args()

    def say(msg: str = "") -> None:
        if not args.quiet and not args.json:
            print(msg)

    expected = expected_documents()
    if not expected:
        print(f"❌ Файлы базы не найдены в {KB_ROOT}")
        return 1

    say(f"🔎 Проверка базы знаний «{COLLECTION_NAME}» (Qdrant: {QDRANT_URL})")
    say(f"   Ожидается документов: {len(expected)}")

    try:
        cid, meta = pick_collection()
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"❌ Qdrant недоступен ({exc.__class__.__name__}: {exc})")
        return 2

    report = {"expected": len(expected), "collection": cid, "missing": [],
              "extra": [], "duplicated": [], "stale": [], "chunks": 0,
              "model_mismatch": None}

    if not cid:
        report["missing"] = sorted(expected)
        return _finish(args, report, "коллекция не найдена — база не залита")

    try:
        counts = titles_in_collection(cid)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"❌ Не удалось прочитать коллекцию ({exc})")
        return 2

    report["chunks"] = sum(counts.values())
    say(f"   Коллекция: {cid}  (чанков: {report['chunks']}, документов: {len(counts)})")
    if meta:
        say(f"   Эмбеддинги: {meta.get('embedding_model')} ({meta.get('embedding_dimensions')} изм.)")

    report["missing"] = sorted(t for t in expected if t not in counts)
    report["extra"] = sorted(t for t in counts if t and t not in expected)

    # Дубли: документ залит дважды → кусков вдвое больше, чем обычно. Надёжнее
    # всего ловится по совпадению chunk_index; здесь — по числу документов с
    # одним названием (Counter склеивает их), поэтому смотрим на «Document» ниже.
    report["duplicated"] = _duplicated(cid)

    # Свежесть: файл изменён позже, чем создана коллекция.
    created = _parse_ts((meta or {}).get("created_at"))
    if created:
        for title, path in expected.items():
            if title in counts and datetime.fromtimestamp(os.path.getmtime(path)) > created:
                report["stale"].append(title)
        report["stale"].sort()

    problems = []
    have_model = (meta or {}).get("embedding_model")
    if have_model and have_model != EMBEDDING_MODEL:
        report["model_mismatch"] = have_model
        say(f"\n🧠 Модель эмбеддингов: в коллекции «{have_model}», нужна «{EMBEDDING_MODEL}».")
        problems.append("модель эмбеддингов устарела — нужна перезаливка")
    if report["missing"]:
        problems.append(f"не залито файлов: {len(report['missing'])}")
    if report["stale"]:
        problems.append(f"изменено после заливки: {len(report['stale'])}")
    if report["duplicated"]:
        problems.append(f"дублей документов: {len(report['duplicated'])}")
    if report["extra"]:
        problems.append(f"лишних (нет на диске): {len(report['extra'])}")

    for key, label in (("missing", "❌ Не залиты"), ("stale", "🕒 Устарели"),
                       ("duplicated", "♊ Дубли"), ("extra", "➕ Лишние")):
        if report[key]:
            say(f"\n{label} ({len(report[key])}):")
            for t in report[key][:15]:
                say(f"   • {t}")
            if len(report[key]) > 15:
                say(f"   … и ещё {len(report[key]) - 15}")

    return _finish(args, report, "; ".join(problems))


def _duplicated(collection: str) -> List[str]:
    """Названия, у которых больше одного document_id (документ залит повторно)."""
    ids: Dict[str, set] = {}
    offset = None
    while True:
        body = {"limit": 256, "with_vector": False,
                "with_payload": {"include": ["document_title", "document_id"]}}
        if offset is not None:
            body["offset"] = offset
        res = _request("POST", f"/collections/{collection}/points/scroll", body, timeout=60)
        result = res.get("result", {})
        for p in result.get("points", []):
            pl = p.get("payload") or {}
            ids.setdefault(pl.get("document_title", ""), set()).add(pl.get("document_id"))
        offset = result.get("next_page_offset")
        if offset is None:
            break
    return sorted(t for t, s in ids.items() if t and len(s) > 1)


def _parse_ts(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _finish(args, report: dict, problems: str) -> int:
    ok = not problems
    if args.json:
        report["ok"] = ok
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif ok:
        print(f"✅ База знаний полная: {report['expected']} документов, "
              f"{report['chunks']} чанков.")
    else:
        print(f"⚠️  База знаний неполная: {problems}.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
