#!/usr/bin/env python3
"""
Оценка качества поиска по базе знаний на тестовом наборе (FAQ базы: вопрос → раздел с ответом).

Запуск (внутри контейнера WebRAgent):
    docker cp knowledge_base/rtu_mirea_llm_kb/data/faq.jsonl webragent-app-1:/tmp/faq.jsonl
    docker cp WebRAgent/eval_retrieval.py webragent-app-1:/tmp/eval_retrieval.py
    docker exec -e RAG_VEC_WEIGHT=0.4 webragent-app-1 python /tmp/eval_retrieval.py /tmp/faq.jsonl

Метрики: доля вопросов, где правильный фрагмент (раздел из FAQ) попал в топ-1/3/5, и MRR.
Подбор весов: перебирайте RAG_VEC_WEIGHT (и RAG_RERANK=1 для переранжирования).
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, "/app")
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

from app.models.collection import Collection  # noqa: E402
from app.services.document_service import DocumentService  # noqa: E402


def sections_of(title: str):
    """«chunks / 16_4_1_zamestiteli.md» → «4.1»; «chunks / 15_4_rukov…» → «4»."""
    m = re.search(r"chunks / \d+_((?:\d+_)*\d+)(?=_[a-z])", title)
    return m.group(1).replace("_", ".") if m else None


def is_gold(title: str, gold: str) -> bool:
    sec = sections_of(title)
    if not sec:
        return False
    return any(sec == g or sec.startswith(g + ".") for g in re.findall(r"\d+(?:\.\d+)*", gold))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/faq.jsonl"
    limit = int(os.getenv("EVAL_LIMIT", "10"))
    cid = next((c.id for c in Collection.get_all() if (c.name or "").lower() == "мирэа"), None)
    if not cid:
        print("Коллекция «МИРЭА» не найдена")
        return 1
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    # вопросы «в базе нет» не оцениваем: у них нет правильного раздела
    rows = [r for r in rows if not re.search(r"в\s+базе\s+нет|не\s+найден", r["ответ"][:90], re.I)]
    ds = DocumentService()
    hits = {1: 0, 3: 0, 5: 0}
    mrr, total, t0, misses = 0.0, 0, time.time(), []
    for r in rows:
        res = ds.search_documents(cid, r["вопрос"], limit=limit).get("results", [])
        # Подставленные «лучшие записи FAQ» (document_title == 'faq') в оценку не берём: тестовые вопросы
        # взяты из того же FAQ, и это было бы подсказкой самому себе. Оцениваем только поиск по фрагментам.
        titles = [x.get("document_title", "") for x in res if "faq" not in x.get("document_title", "").lower()]
        rank = next((i + 1 for i, t in enumerate(titles) if is_gold(t, r["раздел"])), None)
        total += 1
        if rank:
            mrr += 1.0 / rank
            for k in hits:
                if rank <= k:
                    hits[k] += 1
        else:
            misses.append((r["вопрос"], r["раздел"], [sections_of(t) or t[-30:] for t in titles[:3]]))
    dt = time.time() - t0
    print(f"вопросов: {total}  | top1 {hits[1]/total:.1%}  top3 {hits[3]/total:.1%}  top5 {hits[5]/total:.1%}  MRR {mrr/total:.3f}  "
          f"| {dt/total*1000:.0f} мс/вопрос  | RAG_VEC_WEIGHT={os.getenv('RAG_VEC_WEIGHT', 'по умолчанию')} RAG_RERANK={os.getenv('RAG_RERANK', '0')}")
    if os.getenv("EVAL_SHOW_MISSES"):
        for q, g, got in misses[:25]:
            print("  ✗", q[:70], "| нужен раздел", g, "| получили", got)
    return 0


if __name__ == "__main__":
    sys.exit(main())
