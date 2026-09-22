"""Model loading and inference using Qwen-Omni models with Transformers.

Qwen-Omni models are natively end-to-end: they accept audio/text/images/video
and output BOTH text and speech audio. No separate ASR/TTS needed.

Поддерживаются семейства:
  • Qwen3-Omni  (Qwen3OmniMoe*)  — 30B-A3B, требует много памяти
  • Qwen2.5-Omni (Qwen2_5Omni*)  — 7B/3B, работает «из коробки»

Классы выбираются автоматически по `architectures` из config.json модели,
поэтому достаточно поменять VALERA_MODEL_NAME_OR_PATH в .env.
"""

import json
import time
from pathlib import Path
from typing import Optional, Generator

import numpy as np
import torch
from loguru import logger

from config import settings
from core.personas import get_qwen_canonical_prompt
from core.text_filters import normalize_mirea, strip_courtesy
from core.tts import synthesize, tts_uses_model

# префикс в architectures → (класс модели, класс процессора)
MODEL_FAMILIES: dict[str, tuple[str, str]] = {
    "qwen3omnimoe": ("Qwen3OmniMoeForConditionalGeneration", "Qwen3OmniMoeProcessor"),
    "qwen25omni": ("Qwen2_5OmniForConditionalGeneration", "Qwen2_5OmniProcessor"),
    "qwen2omni": ("Qwen2_5OmniForConditionalGeneration", "Qwen2_5OmniProcessor"),
}

# Qwen2.5-Omni синтезирует голос ТОЛЬКО если первым сообщением идёт ИМЕННО этот
# системный промпт (проверка: transformers/models/qwen2_5_omni/
# processing_qwen2_5_omni.py:330). Персона ассистента добавляется ВТОРЫМ
# system-сообщением — шаблон это допускает.
#
# ВАЖНО: язык речи Talker'а задаётся языком этого промпта. Для русского
# ответа нужен русский вариант (см. core/personas.py,
# VALERA_QWEN_SPEECH_RUSSIAN). Процессор при этом печатает предупреждение
# «System prompt modified» — это только предупреждение, озвучка работает.
def _prepare_conversation(conversation: list[dict], family: str) -> list[dict]:
    """Подгоняет сообщения под требования семейства модели.

    Для Qwen2.5-Omni канонический системный промпт обязателен для голосового
    вывода, поэтому исходный system-промпт (персона) переносится во второе
    system-сообщение.
    """
    if not family.startswith("qwen25") and not family.startswith("qwen2omni"):
        return conversation

    messages = list(conversation)
    persona = None
    if messages and messages[0].get("role") == "system":
        content = messages.pop(0).get("content") or []
        parts = [c.get("text", "") for c in content if c.get("type") == "text"]
        persona = "\n".join(p for p in parts if p).strip() or None

    prepared = [
        {
            "role": "system",
            "content": [{"type": "text", "text": get_qwen_canonical_prompt()}],
        }
    ]
    if persona:
        prepared.append({"role": "system", "content": [{"type": "text", "text": persona}]})
    return prepared + messages


def _normalize(name: str) -> str:
    """'Qwen2_5OmniModel' → 'qwen25omnimodel' (для сравнения префиксов)."""
    return (name or "").lower().replace("_", "").replace("-", "")


def _read_architectures(path: str) -> list[str]:
    """Читает architectures из config.json (локально или с HuggingFace)."""
    cfg_file = Path(path) / "config.json"
    try:
        if cfg_file.exists():
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
        else:
            from huggingface_hub import hf_hub_download

            cfg = json.loads(
                Path(hf_hub_download(path, "config.json")).read_text(encoding="utf-8")
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Не удалось прочитать config.json ({exc})")
        return []
    return cfg.get("architectures") or []


def resolve_model_classes(path: str) -> tuple[type, type]:
    """Подбирает классы модели и процессора по architectures из config.json."""
    import transformers

    archs = _read_architectures(path)
    logger.info(f"architectures: {archs or '—'}")

    for prefix, (model_name, proc_name) in MODEL_FAMILIES.items():
        if any(prefix in _normalize(a) for a in archs):
            model_cls = getattr(transformers, model_name, None)
            proc_cls = getattr(transformers, proc_name, None)
            if model_cls is not None and proc_cls is not None:
                logger.info(f"Классы модели: {model_name} / {proc_name}")
                return model_cls, proc_cls

    logger.warning(
        "Неизвестное семейство модели — использую Qwen3OmniMoe по умолчанию"
    )
    return (
        transformers.Qwen3OmniMoeForConditionalGeneration,
        transformers.Qwen3OmniMoeProcessor,
    )


def local_model_problem(local_dir: Path) -> Optional[str]:
    """Описывает проблему, если локальная копия модели неполная, иначе None.

    Нужно, чтобы не пытаться загрузить модель, которая ЕЩЁ СКАЧИВАЕТСЯ: у неё
    уже может быть config.json, но не быть весов или preprocessor_config.json,
    и падение получается невнятным, например:
        Can't load image processor for '.../models/Qwen2.5-Omni-7B'
    """
    if not (local_dir / "config.json").exists():
        return "нет config.json"

    # Если есть индекс шардов — проверяем, что все шарды скачаны
    index = local_dir / "model.safetensors.index.json"
    if index.exists():
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            missing = sorted(
                {shard for shard in weight_map.values()
                 if not (local_dir / shard).exists()}
            )
        except Exception:  # noqa: BLE001 — индекс может быть ещё не дописан
            missing = ["model.safetensors.index.json"]
        if missing:
            return (
                f"не хватает {len(missing)} файлов весов "
                f"(например, {missing[0]})"
            )
    elif not (
        any(local_dir.glob("*.safetensors")) or any(local_dir.glob("*.bin"))
    ):
        return "не скачаны веса модели (*.safetensors)"

    if not (local_dir / "preprocessor_config.json").exists():
        return "нет preprocessor_config.json (нужен процессору модели)"

    return None


class QwenOmniModel:
    """Manages the Qwen-Omni model lifecycle and inference."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.family = ""
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

    def load(self, model_path: Optional[str] = None) -> None:
        """Load the model and processor.

        Классы модели/процессора выбираются по config.json, поэтому подходят и
        Qwen3-Omni, и Qwen2.5-Omni. On Jetson, falls back to sdpa attention if
        flash-attn is unavailable.
        """
        path = model_path or settings.model_name_or_path

        # Если есть ПОЛНАЯ локальная копия — берём её, иначе — репозиторий HF
        local_dir = settings.model_dir
        if local_dir.exists() and any(local_dir.iterdir()):
            problem = local_model_problem(local_dir)
            if problem:
                raise RuntimeError(
                    f"Локальная копия модели неполная: {problem}.\n"
                    f"  Путь: {local_dir}\n"
                    f"  Похоже, загрузка ещё идёт или прервалась. Дождитесь её\n"
                    f"  окончания или запустите загрузку заново:\n"
                    f"      python download_model.py"
                )
            path = str(local_dir)
            logger.info(f"Using local model from {path}")
        else:
            logger.info(f"Model will be downloaded from HuggingFace: {path}")

        ModelClass, ProcessorClass = resolve_model_classes(path)
        self.family = "qwen25omni" if "Qwen2" in ModelClass.__name__ else "qwen3omnimoe"
        logger.info(f"Семейство модели: {self.family}")

        logger.info("Loading processor...")
        self.processor = ProcessorClass.from_pretrained(path)

        # Choose attention implementation based on availability
        attn = settings.attn_implementation
        if attn == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                logger.warning(
                    "flash-attn not available, falling back to 'sdpa' attention"
                )
                attn = "sdpa"

        logger.info(f"Loading model with {attn} attention...")
        load_kwargs = {
            "dtype": settings.model_dtype,
            "device_map": settings.model_device,
            "trust_remote_code": True,
        }
        if attn:
            load_kwargs["attn_implementation"] = attn

        self.model = ModelClass.from_pretrained(path, **load_kwargs)
        self._loaded = True
        logger.info("Model loaded successfully!")

        # Log GPU memory
        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(
                    f"GPU memory: allocated={allocated:.1f}GB, reserved={reserved:.1f}GB"
                )
            except AttributeError:
                # Jetson unified memory — this attr may not exist
                logger.info("GPU memory: unified memory (Jetson)")

    def generate_response(
        self,
        conversation: list[dict],
        speaker: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        with_audio: bool = True,
        tts_backend: Optional[str] = None,
    ) -> tuple[str, Optional[np.ndarray]]:
        """Generate text and audio response from a conversation.

        Args:
            conversation: List of messages in Qwen3-Omni chat format.
            speaker: Voice name for audio output (e.g., "Ethan").
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            with_audio: Синтезировать речь. Для текстового чата выключаем —
                ветка Talker на Jetson работает в разы медленнее текста.

        Returns:
            Tuple of (text_response, audio_waveform).
            audio_waveform is None if audio generation was disabled.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        from qwen_omni_utils import process_mm_info

        t0 = time.time()

        conversation = _prepare_conversation(conversation, self.family)

        # Build chat template
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        # Process multimodal data (audio, images, video)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)

        # Tokenize
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        )

        # Move to model device
        inputs = inputs.to(self.model.device)
        if self.model.dtype != torch.int8:
            inputs = inputs.to(self.model.dtype)

        # Generate
        # Озвучка встроенным Talker'ом включается только если выбран бэкенд
        # "model". При "russian_tts" модель генерирует только текст, а речь
        # синтезирует Silero на CPU — так в разы быстрее и GPU свободен.
        use_model_talker = with_audio and tts_uses_model(tts_backend)
        gen_kwargs = {
            "speaker": speaker or settings.speaker_voice,
            "use_audio_in_video": True,
            "return_audio": use_model_talker,
        }
        limit = max_new_tokens or (
            settings.voice_max_new_tokens if with_audio else settings.max_new_tokens
        )
        if self.family.startswith("qwen3"):
            # Qwen3-Omni: текст возвращается объектом с .sequences только с этим флагом
            gen_kwargs["thinker_return_dict_in_generate"] = True
            gen_kwargs["max_new_tokens"] = limit
        else:
            # Qwen2.5-Omni: длина текста задаётся thinker_max_new_tokens (по умолч. 1024!)
            gen_kwargs["thinker_max_new_tokens"] = limit
        if use_model_talker:
            # Синтез речи — самая дорогая часть (~13 с GPU на 1 с речи).
            # Жёстко ограничиваем длину озвучки.
            gen_kwargs["talker_max_new_tokens"] = settings.talker_max_new_tokens
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        # Защита от зацикливания — модель без штрафа повторяет одну фразу
        gen_kwargs["repetition_penalty"] = settings.repetition_penalty

        t_generate = time.time()
        with torch.no_grad():
            result = self.model.generate(**inputs, **gen_kwargs)
        t_generated = time.time()

        # Семейства возвращают по-разному:
        #   Qwen3-Omni   → (GenerateOutput(.sequences), audio)
        #   Qwen2.5-Omni → (Tensor, audio)
        if isinstance(result, (tuple, list)):
            text_ids = result[0]
            audio_tensor = result[1] if len(result) > 1 else None
        else:
            text_ids, audio_tensor = result, None
        sequences = getattr(text_ids, "sequences", text_ids)

        # Decode text
        decoded = self.processor.batch_decode(
            sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        text_response = decoded[0] if isinstance(decoded, list) else decoded

        # Сколько токенов реально сгенерировано (до обрезки хвостов) — по этой
        # цифре видно, тратится ли время на служебные фразы.
        gen_tokens = int(sequences.shape[1] - inputs["input_ids"].shape[1])

        # Убираем служебные «хвосты» («если у вас есть ещё вопросы, задавайте»):
        # это экономит и генерацию, и синтез речи. Заодно приводим написание
        # университета к «МИРЭА» — модель часто пишет «МИРЕА».
        text_response = strip_courtesy(text_response)
        text_response = normalize_mirea(text_response)

        # Convert audio tensor to numpy
        audio_waveform = None
        if audio_tensor is not None:
            if hasattr(audio_tensor, "detach"):  # torch.Tensor
                audio_waveform = audio_tensor.reshape(-1).detach().cpu().numpy()
            else:  # numpy.ndarray — Qwen2.5-Omni отдаёт именно его
                audio_waveform = np.asarray(audio_tensor).reshape(-1)

        # Бэкенд "russian_tts": модель дала только текст, речь делаем сами
        # (Silero, CPU, русский голос).
        if (
            with_audio
            and not use_model_talker
            and audio_waveform is None
            and text_response.strip()
        ):
            try:
                tts_started = time.time()
                audio_waveform = synthesize(text_response)
                logger.info(
                    f"Silero TTS: {len(audio_waveform) / settings.sample_rate:.1f} с "
                    f"речи за {time.time() - tts_started:.1f} с"
                )
            except Exception as exc:  # озвучка не должна ронять ответ
                logger.error(f"Ошибка синтеза речи (russian_tts): {exc}")

        elapsed = time.time() - t0
        audio_dur = len(audio_waveform) / settings.sample_rate if audio_waveform is not None else 0
        logger.info(
            f"Inference: {elapsed:.2f}s | "
            f"text_len={len(text_response)} | "
            f"audio_dur={audio_dur:.1f}s | "
            f"токенов={gen_tokens} ({gen_tokens / max(elapsed, 1e-6):.1f} ток/с) | "
            f"фазы: подготовка={t_generate - t0:.2f}s "
            f"генерация={t_generated - t_generate:.2f}s "
            f"хвост={time.time() - t_generated:.2f}s"
        )

        return text_response, audio_waveform

    def generate_response_stream(
        self,
        conversation: list[dict],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """Отдаёт текст ответа по мере генерации (только текст, без Talker'а).

        Используется для стриминга: клиент получает первые слова через ~0.6 с
        и может начать озвучку, пока модель дописывает остальное.

        Yields:
            str: очередной фрагмент текста.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        import threading

        from qwen_omni_utils import process_mm_info
        from transformers import TextIteratorStreamer

        conversation = _prepare_conversation(conversation, self.family)
        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        )
        inputs = inputs.to(self.model.device)
        if self.model.dtype != torch.int8:
            inputs = inputs.to(self.model.dtype)

        limit = max_new_tokens or settings.max_new_tokens
        gen_kwargs: dict = {"return_audio": False}
        if self.family.startswith("qwen3"):
            gen_kwargs["thinker_return_dict_in_generate"] = True
            gen_kwargs["max_new_tokens"] = limit
        else:
            gen_kwargs["thinker_max_new_tokens"] = limit
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        gen_kwargs["repetition_penalty"] = settings.repetition_penalty

        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        def _run() -> None:
            with torch.no_grad():
                self.model.generate(**inputs, streamer=streamer, **gen_kwargs)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        try:
            for delta in streamer:
                if delta:
                    yield delta
        finally:
            thread.join()

    def unload(self) -> None:
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False
        torch.cuda.empty_cache()
        logger.info("Model unloaded.")


# Global model singleton
model = QwenOmniModel()
