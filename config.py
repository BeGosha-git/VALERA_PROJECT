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
    # Модель должна запускаться «из коробки»: transformers всё равно
    # материализует веса квантованных моделей в fp16, поэтому на 61 GB RAM
    # помещается максимум ~10B параметров.
    #   "Qwen/Qwen2.5-Omni-7B"  (~20.8 GB, bf16, DEFAULT)
    #   "Qwen/Qwen2.5-Omni-3B"  (~11.2 GB, быстрее, качество ниже)
    # Классы модели/процессора подбираются автоматически по architectures
    # из config.json (см. core/model.py), так что подходит и Qwen3-Omni.
    # NOTE: NVFP4/FP8 модели НЕ работают на Jetson — нет аппаратных ядер
    #       в Ampere (sm_87), они есть только в Blackwell.
    model_name_or_path: str = "Qwen/Qwen2.5-Omni-7B"
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
    # Текстовому чату озвучка не нужна — там лимит больше.
    # У Qwen2.5-Omni это thinker_max_new_tokens (по умолчанию был 1024!).
    # 96 токенов ≈ 40 слов — хватает персоне, а генерация ограничена
    # ~15-20 с (скорость ~4.3 ток/с на Jetson AGX Orin в режиме MAXN).
    max_new_tokens: int = 96
    # Голосовой режим: КАЖДАЯ секунда синтезированной речи стоит ~13 с
    # генерации на Jetson. Поэтому реплика должна быть короткой:
    # ~80 токенов ≈ 40–50 слов ≈ 15–20 с речи.
    voice_max_new_tokens: int = 80
    # Штраф за повторы: без него модель зацикливается
    # («Там много возможностей. Там много возможностей. …»)
    repetition_penalty: float = 1.15

    # Жёсткий предел на длину озвучки (в кадрах кодек-токенов, ~12.5 кадр/с).
    # Страховка от «монологов»: 400 кадров ≈ 32 с максимум.
    talker_max_new_tokens: int = 400

    # ---- Синтез речи (TTS) ----    # Чем озвучивать ответ:
    #   "russian_tts" (по умолчанию) — Silero v3.1_ru, русский голос, CPU,
    #                  ~2 с на секунду речи и GPU свободен
    #   "model"      — встроенный Talker Qwen2.5-Omni, GPU, ~13 с на секунду речи
    tts_backend: str = "russian_tts"
    # Голос Silero: xenia (жен.), eugene / aidar / baya (муж.), kseniya, random
    silero_speaker: str = "eugene"
    # Путь к model.pt (по умолчанию russian_text_to_speech/model.pt)
    silero_model_path: str = ""
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20

    # ---- Нормализация речи (фильтр из ветки PC) ----
    # ASR часто слышит «МИРЭА» как «мир», «мире», «мирэ». При true заменяем и
    # обычные падежи слова «мир» — фраза «во всём мире» станет «во всём МИРЭА».
    mirea_aggressive: bool = False

    # ---- Язык ответов ----
    # true (ПО УМОЛЧАНИЮ) — только русский: модель не отвечает по-английски
    #                        даже на английский вопрос («тег запрета англ.»)
    # false               — двуязычно: вопрос по-английски → ответ по-английски
    russian_only: bool = True

    # ---- Обрезка служебных концовок ----
    # false (ПО УМОЛЧАНИЮ) — текст ИИ отдаётся целиком, ничего не режется;
    #                        краткость и запрет концовок — только в промпте.
    # true                 — выбрасывает финальные фразы вроде «если есть
    #                        вопросы, обращайтесь» и задаёт краткость жёстко
    #                        (полезно, если модель игнорирует промпт)
    strip_courtesy: bool = False

    # ---- Числа словами ----
    # true (ПО УМОЛЧАНИЮ) — перед озвучкой цифры заменяются словами
    #                        («в 1958 году» → «в тысяча девятьсот пятьдесят
    #                        восьмом году»). Silero многозначные цифры молча
    #                        пропускает, поэтому год иначе просто не звучит.
    #                        Сам ответ при этом НЕ меняется — цифры остаются.
    speak_numbers_as_words: bool = True

    # ---- Персона ----
    #   guide (по умолчанию) — экскурсовод МИРЭА, разговорно, кратко
    #   mat                  — то же, но с матом для выразительности
    #   default              — нейтральный «Валера» без экскурсоводческой темы
    persona_mode: str = "guide"
    # Полный текст промпта (пусто — берётся встроенный из core/personas.py)
    system_prompt: str = ""
    system_prompt_mat: str = ""

    # ---- Язык озвучки моделью (Talker Qwen) ----
    # Язык речи задаётся языком канонического системного промпта:
    # true — русский промпт (речь по-русски), false — английский.
    qwen_speech_russian: bool = True
    # Свой канонический промпт (пусто — берётся из core/personas.py)
    qwen_system_prompt: str = ""

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
