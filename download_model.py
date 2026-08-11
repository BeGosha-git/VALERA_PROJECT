"""Download the AWQ 4-bit quantized model from HuggingFace.

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
from loguru import logger


def download_model(force: bool = False):
    """Download the AWQ 4-bit model from HuggingFace."""
    model_dir = settings.model_dir

    if model_dir.exists() and any(model_dir.iterdir()) and not force:
        logger.info(f"Model already exists at {model_dir}")
        logger.info("Use --force to re-download.")
        return

    logger.info(f"Downloading {settings.model_name_or_path}...")
    logger.info(f"Destination: {model_dir}")

    model_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=settings.model_name_or_path,
        local_dir=str(model_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4,
    )

    logger.info("Download complete!")
    logger.info(f"Model saved to: {model_dir}")

    # Show size
    total_size = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file())
    logger.info(f"Total size: {total_size / 1024**3:.1f} GB")


def main():
    parser = argparse.ArgumentParser(description="Download Qwen3-Omni AWQ 4-bit model")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("QWEN-VALERA Model Download")
    logger.info(f"Model: {settings.model_name_or_path}")
    logger.info(f"Expected size: ~10 GB (AWQ 4-bit)")
    logger.info("=" * 50)

    download_model(args.force)


if __name__ == "__main__":
    main()
