#!/usr/bin/env python3
"""Патчи совместимости transformers с Jetson-сборкой PyTorch 2.5.

Проблема 1 — `torch.load` запрещён на torch < 2.6
-------------------------------------------------
transformers/utils/import_utils.py:

    def check_torch_load_is_safe() -> None:
        if not is_torch_greater_or_equal("2.6"):
            raise ValueError("... CVE-2025-32434")

Qwen2.5-Omni читает словарь голосов `spk_dict.pt` через `torch.load`:

    modeling_qwen2_5_omni.py:3724  check_torch_load_is_safe()
    modeling_qwen2_5_omni.py:3725  torch.load(path, weights_only=True)

На Jetson есть только сборка NVIDIA `torch 2.5.0`, поэтому:

    Failed to load model: Due to a serious vulnerability issue in `torch.load`,
    even with `weights_only=True`, we now require users to upgrade torch to at
    least v2.6 ... https://nvd.nist.gov/vuln/detail/CVE-2025-32434

Исправление: порог понижается с 2.6 до 2.5. CVE-2025-32434 касается чтения
НЕДОВЕРЕННЫХ pickle-файлов, а здесь грузятся только файлы из официального
репозитория модели и всегда с `weights_only=True`.

Проблема 2 — SDPA не принимает `enable_gqa`
-------------------------------------------
transformers/integrations/sdpa_attention.py определяет доступность GQA **по
версии torch** (`>= 2.5`), но в сборке NVIDIA для Jetson аргумента `enable_gqa`
нет:

    TypeError: scaled_dot_product_attention() got an unexpected keyword argument 'enable_gqa'

Исправление: вместо проверки версии делается реальная проба — если SDPA не
принимает `enable_gqa`, возвращается False, и transformers сам разворачивает
K/V через `repeat_kv` (см. ниже в том же файле). Результат тот же, ценой
чуть большего расхода памяти на attention.

Запуск (в активированном окружении):
    python patch_transformers.py
    python patch_transformers.py --check   # только проверить

Скрипт идемпотентен: повторный запуск ничего не меняет.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

PATCHES = [
    {
        "name": "torch.load на PyTorch 2.5 (CVE-2025-32434)",
        "file": "utils/import_utils.py",
        "marker": "VALERA: порог понижен",
        "old": '''def check_torch_load_is_safe() -> None:
    if not is_torch_greater_or_equal("2.6"):
''',
        "new": '''def check_torch_load_is_safe() -> None:
    # ── VALERA: порог понижен с 2.6 до 2.5 (Jetson-сборка torch 2.5.0) ───────
    # Оригинал требовал torch >= 2.6 из-за CVE-2025-32434. На Jetson есть
    # только сборка NVIDIA 2.5.0, а загружаются лишь локальные доверенные файлы
    # (spk_dict.pt из официального репозитория Qwen) с weights_only=True.
    if not is_torch_greater_or_equal("2.5"):
''',
    },
    {
        "name": "enable_gqa в SDPA (Jetson-сборка torch 2.5)",
        "file": "integrations/sdpa_attention.py",
        "marker": "VALERA: на Jetson-torch SDPA не умеет enable_gqa",
        "old": '''def use_gqa_in_sdpa(attention_mask: Optional[torch.Tensor], key: torch.Tensor) -> bool:
''',
        "new": '''_VALERA_SDPA_GQA_CACHE: Optional[bool] = None


def _valera_sdpa_supports_gqa() -> bool:
    """Принимает ли SDPA аргумент enable_gqa (проверка не по версии, а пробой).

    В сборке NVIDIA для Jetson (torch 2.5) аргумента нет, хотя версия >= 2.5,
    из-за чего transformers передавал его и падал с TypeError.
    Создано скриптом patch_transformers.py.
    """
    global _VALERA_SDPA_GQA_CACHE
    if _VALERA_SDPA_GQA_CACHE is None:
        try:
            probe = torch.zeros((1, 2, 1, 4))
            torch.nn.functional.scaled_dot_product_attention(
                probe, probe, probe, enable_gqa=True
            )
            _VALERA_SDPA_GQA_CACHE = True
        except Exception:  # noqa: BLE001 — любая ошибка = не поддерживается
            _VALERA_SDPA_GQA_CACHE = False
    return _VALERA_SDPA_GQA_CACHE


def use_gqa_in_sdpa(attention_mask: Optional[torch.Tensor], key: torch.Tensor) -> bool:
    # ── VALERA: на Jetson-torch SDPA не умеет enable_gqa — используем repeat_kv
    if not _valera_sdpa_supports_gqa():
        return False
''',
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="только проверить, ничего не менять")
    args = parser.parse_args()

    print("── Патчи совместимости transformers ───────────────────────────")

    try:
        import transformers
    except Exception as exc:  # pragma: no cover
        print(f"•  transformers недоступен ({exc}) — пропускаю")
        return 0

    root = pathlib.Path(transformers.__file__).resolve().parent
    problems = 0

    for index, patch in enumerate(PATCHES, 1):
        print()
        print(f"{index}. {patch['name']}")
        target = root / patch["file"]

        if not target.exists():
            print(f"•  нет файла {target} — пропускаю")
            continue

        source = target.read_text(encoding="utf-8")

        if patch["marker"] in source:
            print("✓  Патч уже применён")
            continue

        if patch["old"] not in source:
            print(f"•  Ожидаемый код не найден в {target.name} —")
            print("   версия transformers отличается, патч не нужен. Пропускаю.")
            continue

        if args.check:
            print("✗  Требуется патч (запустите без --check)")
            problems += 1
            continue

        target.write_text(
            source.replace(patch["old"], patch["new"], 1), encoding="utf-8"
        )
        print(f"✓  Патч применён: {target.name}")

    print()
    if args.check:
        if problems:
            print("✗ Требуется исправление — запустите скрипт без --check")
            return 1
        print("✓ Все патчи transformers применены")
        return 0

    print("✓ Готово")
    return 0


if __name__ == "__main__":
    sys.exit(main())
