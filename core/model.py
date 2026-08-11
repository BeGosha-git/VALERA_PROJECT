"""Model loading and inference using Qwen3-Omni with Transformers.

The Qwen3-Omni model is natively end-to-end: it accepts audio/text/images/video
and outputs BOTH text and speech audio. No separate ASR/TTS needed.
"""

import time
from pathlib import Path
from typing import Optional, Generator

import numpy as np
import torch
from loguru import logger

from config import settings


class QwenOmniModel:
    """Manages the Qwen3-Omni model lifecycle and inference."""

    def __init__(self):
        self.model = None
        self.processor = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None

    def load(self, model_path: Optional[str] = None) -> None:
        """Load the model and processor.

        Uses AWQ 4-bit quantized model (~10 GB) to fit in 64 GB VRAM.
        On Jetson, falls back to sdpa attention if flash-attn is unavailable.
        """
        path = model_path or settings.model_name_or_path

        # If local path exists, use it; otherwise download from HuggingFace
        local_dir = settings.model_dir
        if local_dir.exists() and any(local_dir.iterdir()):
            path = str(local_dir)
            logger.info(f"Using local model from {path}")
        else:
            logger.info(f"Model will be downloaded from HuggingFace: {path}")

        from transformers import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeProcessor,
        )

        logger.info("Loading processor...")
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(path)

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

        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            path, **load_kwargs
        )
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
    ) -> tuple[str, Optional[np.ndarray]]:
        """Generate text and audio response from a conversation.

        Args:
            conversation: List of messages in Qwen3-Omni chat format.
            speaker: Voice name for audio output (e.g., "Ethan").
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Tuple of (text_response, audio_waveform).
            audio_waveform is None if audio generation was disabled.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        from qwen_omni_utils import process_mm_info

        t0 = time.time()

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
        gen_kwargs = {
            "speaker": speaker or settings.speaker_voice,
            "thinker_return_dict_in_generate": True,
            "use_audio_in_video": True,
            "max_new_tokens": max_new_tokens or settings.max_new_tokens,
        }
        if temperature is not None:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            text_ids, audio_tensor = self.model.generate(**inputs, **gen_kwargs)

        # Decode text
        decoded = self.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        text_response = decoded[0] if isinstance(decoded, list) else decoded

        # Convert audio tensor to numpy
        audio_waveform = None
        if audio_tensor is not None:
            audio_waveform = audio_tensor.reshape(-1).detach().cpu().numpy()

        elapsed = time.time() - t0
        audio_dur = len(audio_waveform) / settings.sample_rate if audio_waveform is not None else 0
        logger.info(
            f"Inference: {elapsed:.2f}s | "
            f"text_len={len(text_response)} | "
            f"audio_dur={audio_dur:.1f}s"
        )

        return text_response, audio_waveform

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
