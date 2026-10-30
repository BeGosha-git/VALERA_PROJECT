#!/usr/bin/env python3
"""
Раскладка сервисов Валеры по видеокартам. Печатает строки для `eval` в bash.

Роли (по убыванию нагрузки на память и пропускную способность):
  LLM   — языковая модель в Ollama: генерация упирается в пропускную способность памяти,
          поэтому ей достаётся САМАЯ БЫСТРАЯ карта с достаточным запасом памяти;
  XTTS  — синтез речи (≈2–3 ГБ);
  ASR   — распознавание речи + запасной Silero (≈1–2 ГБ) в процессе ассистента.

Карт две и больше: LLM — на самой быстрой, XTTS — на следующей, ASR — на третьей
(если карт две, ASR делит карту с XTTS: распознавание и синтез идут не одновременно).
Одна карта: всё на ней. Переопределение: LLM_GPU, XTTS_GPU, ASR_GPU (индексы nvidia-smi),
LLM_SPREAD=1 — растянуть LLM на все карты (нужно только для моделей, не влезающих в одну).

Скорость карты оцениваем по номеру модели в названии (RTX 5070 Ti быстрее 5060 Ti: у неё шире
шина памяти); при равенстве выигрывает карта с большим объёмом памяти. Свободную память не учитываем:
на старте её могут занимать наши же прошлые процессы.
"""
import os
import re
import subprocess
import sys

# Решаем по ОБЩЕМУ объёму памяти, а не по свободному: на момент запуска карту могут занимать
# наши же прошлые процессы (прогретая модель, XTTS), и свободной памяти там временно мало.
MIN_LLM_TOTAL_MB = int(os.getenv("LLM_MIN_TOTAL_MB", "12000"))


def query():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout
    except Exception:  # noqa: BLE001
        return []
    gpus = []
    for line in out.strip().splitlines():
        idx, name, total, free = [x.strip() for x in line.split(",")]
        m = re.search(r"(\d{3,4})", name)
        base = int(m.group(1)) if m else 0
        bonus = 50 if re.search(r"\bTi\b|SUPER", name, re.I) else 0
        gpus.append({"idx": idx, "name": name, "total": int(float(total)), "free": int(float(free)),
                     "speed": base * 10 + bonus})
    return gpus


BIG_MODEL_RE = re.compile(r"(?i)(?:^|[:\-_])(2[6-9]b|3\d+b|[4-9]\d+b|1\d\ds?b)")


def modelfile_base() -> str:
    """Базовая модель из первой строки FROM в Modelfile (или переменной VALERA_MODELFILE)."""
    path = os.getenv("VALERA_MODELFILE") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "Modelfile")
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip().upper().startswith("FROM "):
                    return line.split(None, 1)[1].strip()
    except OSError:
        pass
    return ""


def main():
    gpus = query()
    if not gpus:
        print("# GPU не найдены")
        return
    by_speed = sorted(gpus, key=lambda g: (-g["speed"], -g["free"]))
    roomy = [g for g in by_speed if g["total"] >= MIN_LLM_TOTAL_MB] or by_speed
    llm = os.getenv("LLM_GPU") or roomy[0]["idx"]
    rest = [g["idx"] for g in sorted(gpus, key=lambda g: (-g["total"], -g["speed"])) if g["idx"] != llm]
    xtts = os.getenv("XTTS_GPU") or (rest[0] if rest else llm)
    rest2 = [i for i in rest if i != xtts]
    asr = os.getenv("ASR_GPU") or (rest2[0] if rest2 else xtts)
    # Большая модель (≥26B параметров) в одну 16-ГБ карту не влезает — растягиваем на все карты.
    # Между картами по слоям ходят только активации (килобайты), поэтому даже узкая шина второй
    # карты (PCIe x4) почти не мешает. Явно: LLM_SPREAD=1 / LLM_SPREAD=0.
    forced = os.getenv("LLM_SPREAD")
    big = bool(BIG_MODEL_RE.search(modelfile_base()))
    spread = len(gpus) > 1 and (forced == "1" or (forced is None and big))
    llm_visible = ",".join(g["idx"] for g in gpus) if spread else llm
    print(f"export GPU_COUNT={len(gpus)} LLM_GPU={llm} XTTS_GPU={xtts} ASR_GPU={asr} "
          f"LLM_VISIBLE={llm_visible} LLM_SPREAD={'1' if spread else '0'}")
    names = {g["idx"]: g["name"] for g in gpus}
    print(f"# LLM → GPU {llm} ({names[llm]}{', растянута на все' if spread else ''}); "
          f"XTTS → GPU {xtts} ({names[xtts]}); ASR+Silero → GPU {asr} ({names[asr]})", file=sys.stderr)


if __name__ == "__main__":
    main()
