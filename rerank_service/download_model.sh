#!/bin/bash
# Скачивает модель переранжирования BAAI/bge-reranker-v2-m3 (~2,3 ГБ) в ~/.cache/valera.
# Идёт НАПРЯМУЮ, без прокси (--noproxy): через прокси Hugging Face отдаёт в разы медленнее.
set -e
D="${RERANK_MODEL_DIR:-$HOME/.cache/valera/bge-reranker-v2-m3}"
mkdir -p "$D"; cd "$D"
for f in config.json special_tokens_map.json tokenizer_config.json tokenizer.json sentencepiece.bpe.model model.safetensors; do
    for t in 1 2 3 4 5; do
        curl -sS -L -C - --noproxy '*' --connect-timeout 20 -o "$f" "https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/$f" && break
        sleep 3
    done
done
ls -la "$D"
