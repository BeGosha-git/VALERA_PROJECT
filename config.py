"""Global configuration for QWEN-VALERA voice assistant."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, loadable from environment variables."""

    # ---- Paths ----
    project_root: Path = Path(__file__).parent.resolve()
    data_dir: Path = project_root / "data"
    db_path: Path = data_dir / "valera.db"
    chroma_path: Path = data_dir / "chroma"

    # ---- Model ----
    # Quality options (all work on Jetson AGX Orin, Ampere sm_87):
    #   "cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-8bit"  (~42 GB, best quality, fits 64 GB)
    #   "cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit"  (~27 GB, default, more headroom)
    # NOTE: NVFP4 (25 GB) does NOT work on Jetson — requires Blackwell FP4 hardware.
    model_name_or_path: str = "cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-8bit"
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

    # ---- Documents (knowledge base from .doc/.docx/.pdf) ----
    documents_dir: Path = data_dir / "documents"  # where uploaded files are stored
    chunk_size: int = 800  # characters per chunk
    chunk_overlap: int = 150  # overlap between chunks
    embedding_model: str = "intfloat/multilingual-e5-small"  # local, supports RU
    embedding_device: str = "cpu"  # keep embeddings on CPU to save GPU memory
    rag_top_k: int = 5  # how many document chunks to inject into prompt
    rag_enabled: bool = True  # auto-search documents on every chat request

    model_config = {"env_prefix": "VALERA_", "env_file": ".env"}

    @property
    def model_dir(self) -> Path:
        """Model directory derived from model name (unique per variant)."""
        # Take last path segment of the model id
        name = self.model_name_or_path.rstrip("/").split("/")[-1]
        return self.project_root / "models" / name

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "audio").mkdir(parents=True, exist_ok=True)


settings = Settings()
