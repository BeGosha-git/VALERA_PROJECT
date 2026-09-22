"""Download the Qwen-Omni model from HuggingFace.

Модель задаётся в .env (VALERA_MODEL_NAME_OR_PATH). По умолчанию это
Qwen/Qwen2.5-Omni-7B (~20.8 GB, bf16) — она запускается «из коробки»,
без распаковки квантованных весов.

Usage:
    python download_model.py              # download to default location
    python download_model.py --force      # re-download even if exists
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import settings
from core.model import local_model_problem
from loguru import logger


def download_model(force: bool = False):
    """Download the model from HuggingFace."""
    model_dir = settings.model_dir

    if model_dir.exists() and any(model_dir.iterdir()) and not force:
        problem = local_model_problem(model_dir)
        if problem is None:
            logger.info(f"Модель уже скачана полностью: {model_dir}")
            logger.info("Перекачать заново: python download_model.py --force")
            return
        logger.warning(f"Локальная копия неполная ({problem}) — докачиваю...")

    logger.info(f"Downloading {settings.model_name_or_path}...")
    logger.info(f"Destination: {model_dir}")

    model_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    # local_dir_use_symlinks/resume_download убраны в новых версиях
    # huggingface_hub — передаём только актуальные аргументы.
    snapshot_download(
        repo_id=settings.model_name_or_path,
        local_dir=str(model_dir),
        max_workers=4,
    )

    logger.info("Download complete!")
    logger.info(f"Model saved to: {model_dir}")

    # Проверяем, что скачалось ВСЁ (иначе сервер упадёт при загрузке)
    problem = local_model_problem(model_dir)
    if problem:
        logger.error(f"Загрузка неполная: {problem}")
        logger.error("Запустите скрипт ещё раз — он докачает недостающее.")
        return

    # Show size
    total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    logger.info(f"Total size: {total_size / 1024**3:.1f} GB")
    logger.info("✓ Все файлы модели на месте")


def main():
    parser = argparse.ArgumentParser(description="Скачивание Qwen-Omni модели (из .env)")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("QWEN-VALERA Model Download")
    logger.info(f"Model: {settings.model_name_or_path}")
    logger.info(f"Destination: {settings.model_dir}")
    logger.info("=" * 50)

    download_model(args.force)


if __name__ == "__main__":
    main()
