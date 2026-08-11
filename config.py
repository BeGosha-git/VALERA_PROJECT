"""Global configuration for QWEN-VALERA voice assistant."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, loadable from environment variables."""

    # ---- Paths ----
    project_root: Path = Path(__file__).parent.resolve()
    model_dir: Path = project_root / "models" / "Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit"
    data_dir: Path = project_root / "data"
    db_path: Path = data_dir / "valera.db"
    chroma_path: Path = data_dir / "chroma"

    # ---- Model ----
    model_name_or_path: str = "cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit"
    model_device: str = "auto"  # "auto" for device_map, "cuda:0" for single GPU
    model_dtype: str = "auto"  # auto-detect from config
    # On Jetson: "sdpa" (flash-attn is not available for ARM64).
    # On x86+GPU with flash-attn installed: "flash_attention_2" is faster.
    attn_implementation: str = "sdpa"
    speaker_voice: str = "Ethan"  # model's built-in voice name

    # ---- Server ----
    api_host: str = "0.0.0.0"
    api_port: int = 8765
    max_audio_length_seconds: int = 120
    sample_rate: int = 24000  # model native sample rate

    # ---- Generation ----
    max_new_tokens: int = 2048
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20

    # ---- Search ----
    search_enabled: bool = True
    search_region: str = "ru-ru"
    # Enable the model to call external tools (search, DB) via system prompt
    tools_enabled: bool = True

    # ---- Database ----
    db_echo: bool = False  # SQLAlchemy echo for debugging

    model_config = {"env_prefix": "VALERA_", "env_file": ".env"}

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "audio").mkdir(parents=True, exist_ok=True)


settings = Settings()
