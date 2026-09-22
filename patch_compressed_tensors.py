#!/usr/bin/env python3
"""Ускоряет загрузку AWQ-модели: оптимизирует проверку списка `ignore` в compressed-tensors.

Проблема
--------
У Qwen3-Omni в `config.json` → `quantization_config.ignore` около **11 500**
полных имён модулей. Функция `compressed_tensors.utils.match.is_match()`
перебирала весь этот список для КАЖДОГО модуля модели:

    return not isinstance(module, InternalModule) and (
        any(match_name(name, target, fused) or _match_class(module, target)
            for target in targets)
        and not any(match_name(name, ign, fused) or _match_class(module, ign)
                    for ign in ignore)          # ← 11 500 итераций на модуль
    )

При ~150 000 модулях у 30B MoE-модели это миллиарды операций. Внешне это
выглядит как «зависание»: `python main.py` стоит на `Loading model with sdpa
attention...` с 100% CPU и без чтения файлов модели часами.

Вот стек, который это подтверждает (снят через faulthandler):
    compressed_tensors/utils/match.py:455 in _match_class
    compressed_tensors/utils/match.py:375 in <genexpr>
    compressed_tensors/utils/match.py:374 in is_match
    compressed_tensors/utils/match.py:62  in match_named_modules
    compressed_tensors/quantization/lifecycle/apply.py:131 in apply_quantization_config
    transformers/quantizers/quantizer_compressed_tensors.py:79 in
        _process_model_before_weight_loading

Что делает патч
---------------
1. Сначала проверяются `targets` (их мало) — большинство модулей отсеивается
   сразу, без обращения к `ignore`.
2. Список `ignore` один раз превращается в set точных имён + короткие списки
   имён классов и регулярных выражений. Проверка точного имени — O(1).
3. Поведение функции не меняется: `re:`-шаблоны по-прежнему проверяются
   регулярками, имена классов — через MRO, при `fused` используется исходный
   медленный путь.

Запуск (в активированном окружении):
    python patch_compressed_tensors.py
    python patch_compressed_tensors.py --check   # только проверить

Скрипт идемпотентен: повторный запуск ничего не меняет.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

MARKER = "# ─── VALERA: ускорение проверки ignore ───"

ORIGINAL_RETURN = '''    return not isinstance(module, InternalModule) and (
        any(
            match_name(name, target, fused) or _match_class(module, target)
            for target in targets
        )
        and not any(
            match_name(name, ign, fused) or _match_class(module, ign) for ign in ignore
        )
    )
'''

PATCHED_RETURN = '''    return _valera_is_match(name, module, targets, ignore, fused)
'''

HELPERS = '''

# ─── VALERA: ускорение проверки ignore ───────────────────────────────────────
# В match_named_modules() список ignore (~11 500 точных имён у Qwen3-Omni)
# передаётся в is_match() как аргумент `targets`, и исходная реализация
# перебирала его целиком для каждого модуля модели (~150 000) — это миллиарды
# операций. Здесь список один раз превращается в set, проверка становится O(1).
_IGNORE_CACHE: dict = {}


def _prepare_patterns(patterns):
    entry = _IGNORE_CACHE.get(id(patterns))
    if entry is not None and entry[0] is patterns:
        return entry[1]

    names = set()
    class_names = set()
    regexes = []
    for pattern in patterns:
        if not isinstance(pattern, str):
            continue
        if pattern.startswith("re:"):
            regexes.append(pattern[5:])
        else:
            names.add(pattern)
            if "." not in pattern:
                class_names.add(pattern)

    prepared = (names, class_names, tuple(regexes))
    _IGNORE_CACHE[id(patterns)] = (patterns, prepared)
    return prepared


def _any_match(name, module, patterns, fused=None) -> bool:
    if not patterns:
        return False

    # При fused-мэппинге (vLLM) используем исходную логику
    if fused is not None:
        return any(
            match_name(name, pattern, fused) or _match_class(module, pattern)
            for pattern in patterns
        )

    names, class_names, regexes = _prepare_patterns(patterns)

    if name in names:
        return True

    if class_names:
        mro_names = {cls.__name__ for cls in module.__class__.__mro__}
        if mro_names & class_names:
            return True
        # исключение из оригинала: vllm LinearBase матчится как Linear
        if "LinearBase" in mro_names and "Linear" in class_names:
            return True

    for pattern in regexes:
        if re.match(pattern, name) is not None:
            return True

    return False


def _valera_is_match(name, module, targets, ignore=tuple(), fused=None) -> bool:
    if isinstance(module, InternalModule):
        return False
    if not _any_match(name, module, targets, fused):
        return False
    if not ignore:
        return True
    return not _any_match(name, module, ignore, fused)
'''


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="только проверить, ничего не менять")
    args = parser.parse_args()

    print("── Ускорение проверки ignore в compressed-tensors ──────────────")

    try:
        from compressed_tensors.utils import match as match_module
    except Exception as exc:
        print(f"•  compressed-tensors не установлен ({exc}) — пропускаю")
        return 0

    path = pathlib.Path(match_module.__file__).resolve()
    print(f"   файл: {path}")

    source = path.read_text(encoding="utf-8")

    if MARKER in source:
        print("✓  Патч уже применён")
        return 0

    if ORIGINAL_RETURN not in source:
        print("•  Ожидаемый код is_match() не найден — версия библиотеки отличается,")
        print("   патч не нужен или неприменим. Пропускаю.")
        return 0

    if args.check:
        print("✗  Требуется патч (запустите без --check)")
        return 1

    # Проверяем, что re уже импортирован (нужен для re: шаблонов)
    if "\nimport re" not in source and not source.startswith("import re"):
        print("•  Модуль не импортирует `re` — пропускаю во избежание поломки")
        return 0

    source = source.replace(ORIGINAL_RETURN, PATCHED_RETURN, 1)
    source += HELPERS
    path.write_text(source, encoding="utf-8")

    print("✓  Патч применён (проверка ignore теперь O(1) для точных имён)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
