#!/usr/bin/env python3
"""
Сервис переранжирования фрагментов базы знаний (cross-encoder BAAI/bge-reranker-v2-m3).

Поиск по базе сначала находит кандидатов дёшево (BM25 + векторы), а этот сервис читает вопрос
и каждого кандидата ВМЕСТЕ и ставит оценку «насколько фрагмент отвечает на вопрос». Так
выбирается нужный фрагмент среди похожих. Модель многоязычная (русский — штатно), ~1,1 ГБ в fp16,
работает на отдельной видеокарте (RERANK_GPU), WebRAgent обращается к ней по HTTP.

API:  POST /rerank {"query": "...", "passages": ["...", ...]} → {"scores": [p1, p2, ...]}  (вероятности 0..1)
      GET /health
Запуск: python rerank_service/server.py   (порт RERANK_PORT, по умолчанию 8152)
"""
from __future__ import annotations

import os
import threading
import time
from typing import List

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = os.getenv("RERANK_MODEL_DIR", os.path.expanduser("~/.cache/valera/bge-reranker-v2-m3"))
PORT = int(os.getenv("RERANK_PORT", "8152"))
HOST = os.getenv("RERANK_HOST", "0.0.0.0")      # контейнер WebRAgent ходит на 172.17.0.1
MAX_LEN = int(os.getenv("RERANK_MAX_LEN", "512"))

app = FastAPI(title="Valera reranker")
_tok = _model = None
_dev = "cpu"
_lock = threading.Lock()


class Req(BaseModel):
    query: str
    passages: List[str]


@app.on_event("startup")
def _load():
    global _tok, _model, _dev
    _dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    _tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    _model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16 if _dev == "cuda" else torch.float32).to(_dev).eval()
    print(f"[rerank] модель загружена на {_dev} за {time.time() - t0:.1f} с", flush=True)
    _score("прогрев", ["прогрев"])


def _score(query: str, passages: List[str]) -> List[float]:
    with _lock, torch.inference_mode():
        enc = _tok([query] * len(passages), passages, padding=True, truncation="only_second",
                   max_length=MAX_LEN, return_tensors="pt").to(_dev)
        logits = _model(**enc).logits.view(-1).float()
        return torch.sigmoid(logits).cpu().tolist()


@app.get("/health")
def health():
    return {"status": "ok" if _model is not None else "loading", "device": _dev}


@app.post("/rerank")
def rerank(req: Req):
    if not req.passages:
        return {"scores": []}
    t0 = time.time()
    scores = _score(req.query, [p[:2000] for p in req.passages[:32]])
    return {"scores": scores, "ms": int((time.time() - t0) * 1000)}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
