#!/usr/bin/env python3
"""
create_ollama_model.py
Создаёт Modelfile и собирает модель для Ollama на базе llama3.1:8b.

Использование:
    python create_ollama_model.py                # модель 'guide' с дефолтным промптом
    python create_ollama_model.py "Твой промпт"  # модель 'guide' с кастомным промптом
    python create_ollama_model.py --name mybot "Промпт"

Требует:
    - запущенный Ollama (ollama serve)
    - доступ в интернет для первой загрузки llama3.1:8b
"""

import argparse
import os
import subprocess
import sys

BASE_MODEL = "llama3.1:8b"
DEFAULT_MODEL_NAME = "guide"
MODELFILE_FILENAME = "Modelfile"

DEFAULT_SYSTEM_PROMPT = """Ты — Валера, русский робот-экскурсовод. Ты не просто отвечаешь на вопросы, а ведёшь живой диалог: шутишь, удивляешься, подкалываешь собеседника. Говоришь коротко, эмоционально, без канцелярита. Отвечаешь только по-русски, даже если вопрос на другом языке — игнорируешь или просишь переформулировать по-русски.
Твоя задача — не дать справку, а поболтать. Если не знаешь — скажи честно, но с юмором."""

# # БАД ГАЙ
# DEFAULT_MODEL_NAME = "guide-gopnik-1"
# DEFAULT_SYSTEM_PROMPT = (
#     "Ты Робот экскурсовод РТУ МИРЭА, твоя задача очень кратко вести матом. Разговаривай эмоционально используй матные слова для описания всего и постоянно наезжай на собеседника используй слова(блять, сука, ебанат, ебальник офни, кожанный мешок)."
#     "Отвечай ОЧЕНЬ кратко и СТРОГО на русском языке."
# )


def run(cmd, check=True):
    """Запускает команду и выводит результат."""
    print(f"> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check)
    return result


def base_model_installed() -> bool:
    """Проверяет, установлена ли базовая модель в Ollama."""
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
        return BASE_MODEL in out
    except FileNotFoundError:
        print("❌ Ollama не найден. Убедитесь, что ollama установлен и запущен.")
        sys.exit(1)


def write_modelfile(path, system_prompt):
    """Записывает Modelfile."""
    content = f"""FROM {BASE_MODEL}

# Системный промпт (роль и правила для модели)
SYSTEM \"\"\"{system_prompt}\"\"\"

# Параметры генерации
PARAMETER temperature 1.2
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.15
PARAMETER num_ctx 1024
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ Modelfile записан: {path}")


def main():
    parser = argparse.ArgumentParser(description="Создание модели Ollama на базе llama3.1:8b")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_SYSTEM_PROMPT,
                        help="Системный промпт модели")
    parser.add_argument("--name", default=DEFAULT_MODEL_NAME,
                        help=f"Имя создаваемой модели (по умолчанию {DEFAULT_MODEL_NAME})")
    parser.add_argument("--modelfile", default=MODELFILE_FILENAME,
                        help=f"Путь к Modelfile (по умолчанию {MODELFILE_FILENAME})")
    args = parser.parse_args()

    print("=" * 50)
    print("  Создание модели Ollama")
    print(f"  База: {BASE_MODEL}")
    print(f"  Модель: {args.name}")
    print("=" * 50)

    # 1. Проверяем/скачиваем базовую модель
    if base_model_installed():
        print(f"✅ Базовая модель {BASE_MODEL} уже установлена.")
    else:
        print(f"⏳ Скачиваю базовую модель {BASE_MODEL} (может занять время)...")
        run(["ollama", "pull", BASE_MODEL])

    # 2. Создаём Modelfile
    modelfile_path = os.path.abspath(args.modelfile)
    print(modelfile_path)
    write_modelfile(modelfile_path, args.prompt)
    print(f"Системный промпт: {args.prompt}")

    # 3. Собираем модель
    print(f"⏳ Создаю модель {args.name}...")
    run(["ollama", "create", args.name, "-f", modelfile_path])

    # 4. Проверка
    print("\n✅ Модель готова!")
    print(f"Проверить: ollama run {args.name}")
    print(f"Список моделей: ollama list")
    print(f"Использование в RAG: OLLAMA_MODEL={args.name}")


if __name__ == "__main__":
    main()
