#!/usr/bin/env python3
"""Патчи совместимости для Jetson-сборки PyTorch (NVIDIA, JetPack 6.x).

NVIDIA-сборка `torch-2.5.0a0+...nv24.08` отличается от обычной PyPI-сборки и
ломает transformers. Скрипт исправляет две проблемы.

Проблема 1 — версия считается «меньше 2.5.0»
--------------------------------------------
NVIDIA собирает wheel с версией вида:

    2.5.0a0+872d972e41.nv24.08

По правилам PEP 440 суффикс `a0` означает ПРЕ-релиз версии 2.5.0, то есть
такая версия считается МЕНЬШЕ, чем 2.5.0. Из-за этого transformers отключает
PyTorch:

    [transformers] Disabling PyTorch because PyTorch >= 2.5 is required
                   but found 2.5.0a0+872d972e41.nv24.8
    [transformers] PyTorch was not found. Models won't be available ...

Проверка: transformers/utils/import_utils.py:194
    if version.parse(torch_version) < version.parse("2.5.0"): ...

Исправление: версия дистрибутива переписывается на `2.5.0+nv24.08`.
`+nv24.08` — локальная версия PEP 440, она сравнивается как БОЛЬШЕ 2.5.0,
поэтому все проверки `torch >= 2.5.0` проходят, а информация о сборке NVIDIA
сохраняется. Правятся: METADATA, RECORD, torch/version.py.

Проблема 2 — вырезан модуль flex_attention
------------------------------------------
В Jetson-сборке нет публичного модуля `torch.nn.attention.flex_attention`
(остался только приватный `_flex_attention.py`). При этом transformers
определяет доступность только по версии:

    def is_torch_flex_attn_available():
        return is_torch_available() and get_torch_version() >= "2.5.0"

и поэтому пытается импортировать несуществующий модуль:

    ModuleNotFoundError: Could not import module 'GenerationMixin'
      ← из-за `from torch.nn.attention.flex_attention import ...`
        в transformers/masking_utils.py:33

Исправление: создаётся модуль-заглушка `torch/nn/attention/flex_attention.py`.
Он повторяет ветку transformers «flex attention недоступна»:
`BlockMask = torch.Tensor`, а вызовы `create_block_mask()`/`flex_attention()`
явно сообщают об ошибке. Проект использует `attn_implementation="sdpa"`,
поэтому эти функции не вызываются.

Запуск (в активированном окружении с torch):
    python patch_torch_jetson.py
    python patch_torch_jetson.py --check     # только проверить, ничего не менять
    python patch_torch_jetson.py --list      # показать текущее состояние

Скрипт идемпотентен: повторный запуск ничего не меняет.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import pathlib
import re
import shutil
import sys

from packaging.version import InvalidVersion
from packaging.version import parse as parse_version

TARGET_VERSION = "2.5.0+nv24.08"
MIN_VERSION = "2.5.0"

BUFFER_STUB_MARKER = "class Buffer(_ValeraTensorBase):"

BUFFER_STUB = '''

# ── VALERA: совместимость с библиотеками, ожидающими torch >= 2.6 ─────────────
# В Jetson-сборке torch 2.5 нет класса torch.nn.Buffer (он появился в 2.6), но
# его используют сторонние библиотеки, например compressed-tensors:
#     isinstance(tensor, (torch.nn.Parameter, torch.nn.Buffer))
# Ниже — класс-маркер: isinstance() работает, а попытка создать буфер даёт
# понятную ошибку вместо AttributeError. Сами буферы в torch 2.5 — обычные
# Tensor, зарегистрированные через Module.register_buffer().
# Создано скриптом patch_torch_jetson.py.
from torch import Tensor as _ValeraTensorBase


class Buffer(_ValeraTensorBase):
    """Заглушка torch.nn.Buffer для Jetson-сборки PyTorch 2.5."""

    _is_buffer = True

    def __new__(cls, *args, **kwargs):
        raise RuntimeError(
            "torch.nn.Buffer недоступен в Jetson-сборке PyTorch 2.5. "
            "Используйте module.register_buffer(name, tensor, persistent=...)"
        )
'''

FLEX_ATTN_SHIM = '''\
"""Модуль-заглушка `torch.nn.attention.flex_attention`.

Создан скриптом patch_torch_jetson.py.

В Jetson-сборке PyTorch публичный модуль `flex_attention` вырезан (остался
только приватный `_flex_attention.py`), но transformers определяет его
доступность только по версии torch и пытается его импортировать.

Этот модуль повторяет поведение ветки transformers «flex attention
недоступна»: `BlockMask = torch.Tensor`, а вызовы функций явно сообщают
об ошибке. Проект использует `attn_implementation="sdpa"`, поэтому
функции ниже не вызываются.
"""

import torch

try:  # константа есть в приватном модуле — берём значение оттуда
    from ._flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE
except Exception:  # pragma: no cover
    _DEFAULT_SPARSE_BLOCK_SIZE = 128

# Тип-заглушка: ровно так делает transformers, когда flex attention недоступна
BlockMask = torch.Tensor

_MESSAGE = (
    "torch.nn.attention.flex_attention недоступен в Jetson-сборке PyTorch "
    "(модуль вырезан NVIDIA). Используйте attn_implementation='sdpa' или 'eager'."
)


def create_block_mask(*args, **kwargs):
    raise NotImplementedError(_MESSAGE)


def flex_attention(*args, **kwargs):
    raise NotImplementedError(_MESSAGE)


__all__ = ["BlockMask", "create_block_mask", "flex_attention", "_DEFAULT_SPARSE_BLOCK_SIZE"]
'''


# ─────────────────────────────────────────────────────────────────────────────
# Утилиты
# ─────────────────────────────────────────────────────────────────────────────


def _sha256_line(path: pathlib.Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).decode().rstrip("=")


def _find_dist_info(site: pathlib.Path) -> pathlib.Path | None:
    infos = sorted(site.glob("torch-*.dist-info"))
    return infos[0] if infos else None


def _metadata_version(metadata: pathlib.Path) -> str | None:
    match = re.search(r"^Version:\s*(.+)$", metadata.read_text(encoding="utf-8"), re.M)
    return match.group(1).strip() if match else None


def _is_ok(version: str) -> bool:
    """Версия корректно сравнивается с MIN_VERSION?"""
    try:
        return parse_version(version) >= parse_version(MIN_VERSION)
    except InvalidVersion:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Патч 1: версия дистрибутива
# ─────────────────────────────────────────────────────────────────────────────


def _patch_version(
    site: pathlib.Path,
    dist_info: pathlib.Path,
    torch_pkg: pathlib.Path,
    current: str,
) -> None:
    # 1. METADATA
    metadata = dist_info / "METADATA"
    metadata.write_text(
        re.sub(
            r"^Version:\s*.*$",
            f"Version: {TARGET_VERSION}",
            metadata.read_text(encoding="utf-8"),
            count=1,
            flags=re.M,
        ),
        encoding="utf-8",
    )

    # 2. RECORD (пути + хэши изменившихся файлов)
    record = dist_info / "RECORD"
    if record.exists():
        new_lines: list[str] = []
        for line in record.read_text(encoding="utf-8").splitlines():
            parts = line.split(",")
            if not parts or not parts[0]:
                new_lines.append(line)
                continue
            rel = parts[0].replace(current, TARGET_VERSION)
            target = site / rel
            if len(parts) >= 3 and parts[1] and target.is_file():
                parts[1] = _sha256_line(target)
                parts[2] = str(target.stat().st_size)
            parts[0] = rel
            new_lines.append(",".join(parts))
        record.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    # 3. torch/version.py
    version_file = torch_pkg / "version.py"
    if version_file.exists():
        version_file.write_text(
            re.sub(
                r"^__version__\s*=\s*.*$",
                f"__version__ = '{TARGET_VERSION}'",
                version_file.read_text(encoding="utf-8"),
                count=1,
                flags=re.M,
            ),
            encoding="utf-8",
        )

    # 4. Переименовать dist-info
    new_dist_info = dist_info.parent / f"torch-{TARGET_VERSION}.dist-info"
    if new_dist_info != dist_info:
        if new_dist_info.exists():
            shutil.rmtree(new_dist_info)
        dist_info.rename(new_dist_info)


# ─────────────────────────────────────────────────────────────────────────────
# Патч 2: модуль-заглушка flex_attention
# ─────────────────────────────────────────────────────────────────────────────


def _patch_flex_attention(torch_pkg: pathlib.Path) -> pathlib.Path:
    target = torch_pkg / "nn" / "attention" / "flex_attention.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(FLEX_ATTN_SHIM, encoding="utf-8")
    return target


def _patch_buffer(nn_init: pathlib.Path) -> None:
    nn_init.write_text(
        nn_init.read_text(encoding="utf-8") + BUFFER_STUB,
        encoding="utf-8",
    )


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="только проверить, ничего не менять")
    parser.add_argument("--list", action="store_true", help="показать текущее состояние и выйти")
    args = parser.parse_args()

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(f"✗ Не удалось импортировать torch: {exc}")
        print("  Сначала установите Jetson-сборку PyTorch.")
        return 1

    torch_pkg = pathlib.Path(torch.__file__).resolve().parent
    site = torch_pkg.parent
    dist_info = _find_dist_info(site)
    problems = 0

    print("── 1. Версия дистрибутива ──────────────────────────────────────")
    if dist_info is None:
        print(f"✗ Не найден torch-*.dist-info в {site}")
        problems += 1
    else:
        metadata = dist_info / "METADATA"
        current = _metadata_version(metadata)
        if current is None:
            print(f"✗ В {metadata} нет поля Version")
            problems += 1
        elif args.list:
            print(f"   метаданные        : {current}")
            print(f"   torch.__version__ : {torch.__version__}")
        elif _is_ok(current):
            print(f"✓  Версия корректная: {current} (PEP 440 считает >= {MIN_VERSION})")
        else:
            print(f"   версия в метаданных : {current}")
            print(f"   torch.__version__   : {torch.__version__}")
            print(f"   проблема            : PEP 440 считает это пре-релизом, т.е. < {MIN_VERSION}")
            print(f"   новая версия        : {TARGET_VERSION}")
            if args.check:
                problems += 1
            else:
                _patch_version(site, dist_info, torch_pkg, current)
                print(f"✓  Версия исправлена на {TARGET_VERSION}")

    print()
    print("── 2. Модуль flex_attention ────────────────────────────────────")
    shim = torch_pkg / "nn" / "attention" / "flex_attention.py"
    if shim.exists():
        print(f"✓  Заглушка уже на месте: {shim}")
    else:
        print("   отсутствует модуль: torch.nn.attention.flex_attention")
        print("   (в Jetson-сборке остался только приватный _flex_attention.py)")
        if args.check:
            problems += 1
        else:
            _patch_flex_attention(torch_pkg)
            print(f"✓  Создана заглушка: {shim}")

    print()
    print("── 3. Класс torch.nn.Buffer ────────────────────────────────────")
    nn_init = torch_pkg / "nn" / "__init__.py"
    if not nn_init.exists():
        print(f"✗ Не найден {nn_init}")
        problems += 1
    else:
        nn_source = nn_init.read_text(encoding="utf-8")
        if BUFFER_STUB_MARKER in nn_source or "torch.nn.Buffer недоступен" in nn_source:
            print("✓  Заглушка torch.nn.Buffer уже добавлена")
        else:
            print("   отсутствует класс torch.nn.Buffer (появился в torch 2.6),")
            print("   но его используют compressed-tensors и другие библиотеки")
            if args.check:
                problems += 1
            else:
                _patch_buffer(nn_init)
                print("✓  Добавлена заглушка torch.nn.Buffer")

    print()
    if args.list:
        return 0

    if problems:
        print("✗ Требуется исправление — запустите скрипт без --check")
        return 1

    print("✓ Все патчи Jetson-совместимости применены")
    return 0


if __name__ == "__main__":
    sys.exit(main())
