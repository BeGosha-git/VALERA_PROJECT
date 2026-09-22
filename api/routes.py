"""FastAPI routes for the voice assistant."""

import asyncio
import base64
import io
import json
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from loguru import logger

from api.schemas import (
    AudioResponse,
    DevicesResponse,
    HealthResponse,
    KnowledgeCreate,
    KnowledgeEntry,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    LLMRequest,
    LLMResponse,
    TextOnlyResponse,
    TextRequest,
)
from config import settings
from core.audio_io import audio_to_wav_bytes, list_audio_devices, load_audio
from core.conversation import Conversation, conversation, create_new_conversation
from core.model import model
from core.search import search_and_format
from core.text_filters import (
    is_courtesy_sentence,
    is_mirea_related,
    normalize_mirea,
    strip_courtesy,
)
from core.tts import describe_backend, synthesize
from core.tts import describe_backend
from db.database import db
from db.documents import search_documents_formatted
from db.knowledge_base import vector_store

router = APIRouter()

# Conversation registry: session_id -> Conversation
sessions: dict[str, Conversation] = {"default": conversation}


def get_or_create_session(
    session_id: Optional[str], persona: Optional[str] = None
) -> tuple[str, Conversation]:
    """Get existing session or create a new one.

    ``persona`` переключает персону на лету ("mat" — режим с матом).
    """
    if session_id and session_id in sessions:
        conv = sessions[session_id]
        if persona:
            conv.apply_persona(persona)
        return session_id, conv

    new_id = session_id or str(uuid.uuid4())[:8]
    if new_id not in sessions:
        sessions[new_id] = create_new_conversation()
    if persona:
        sessions[new_id].apply_persona(persona)
    return new_id, sessions[new_id]


# ═══════════════════════════════════════════════════════════════════════════════
# Health & Info
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Server health and GPU status."""
    import torch

    gpu_available = torch.cuda.is_available()
    allocated = torch.cuda.memory_allocated(0) / 1024**3 if gpu_available else None
    # В torch >= 2.0 атрибут называется total_memory (старый total_mem убран)
    if gpu_available:
        props = torch.cuda.get_device_properties(0)
        total = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1024**3
    else:
        total = None

    return HealthResponse(
        status="ok",
        model_loaded=model.is_loaded,
        gpu_available=gpu_available,
        gpu_memory_used_gb=round(allocated, 2) if allocated else None,
        gpu_memory_total_gb=round(total, 2) if total else None,
        tts_backend=describe_backend(),
    )


@router.get("/devices", response_model=DevicesResponse)
async def list_devices():
    """List available audio devices."""
    from core.audio_io import get_default_microphone, get_default_speaker

    return DevicesResponse(
        devices=list_audio_devices(),
        default_input=get_default_microphone(),
        default_output=get_default_speaker(),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Text Chat
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/chat/text", response_model=TextOnlyResponse)
async def chat_text(req: TextRequest):
    """Send text message, get text response.

    Automatically searches:
      1. Uploaded documents (RAG) — always if rag_enabled
      2. Internet (DuckDuckGo) — if the query looks like a web search
    """
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    session_id, conv = get_or_create_session(req.session_id, req.persona)

    # 1. RAG: search uploaded documents
    doc_context = ""
    rag_used = False
    if settings.rag_enabled:
        _t0 = time.time()
        doc_context = search_documents_formatted(req.text, settings.rag_top_k)
        rag_used = bool(doc_context)
        logger.info(f"Поиск по БД: {time.time() - _t0:.2f}s (найдено: {rag_used})")

    # 2. Internet search if enabled and query looks like a search
    search_context = ""
    search_used = False
    if req.enable_search and _is_search_query(req.text):
        _t0 = time.time()
        search_context = search_and_format(req.text)
        search_used = True
        logger.info(f"Поиск в интернете: {time.time() - _t0:.2f}s")

    # Build user message with optional contexts
    user_text = req.text
    if doc_context and search_context:
        user_text = (
            f"{req.text}\n\n{doc_context}\n\n{search_context}\n\n"
            f"Ответь на вопрос пользователя, используя информацию из документов "
            f"и результаты поиска."
        )
    elif doc_context:
        user_text = (
            f"{req.text}\n\n{doc_context}\n\n"
            f"Ответь на вопрос пользователя, используя информацию из документов."
        )
    elif search_context:
        user_text = (
            f"{req.text}\n\n{search_context}\n\n"
            f"Ответь на вопрос пользователя, используя результаты поиска."
        )

    conv.add_user_message(text=user_text)

    t0 = time.time()
    try:
        # Текстовому чату озвучка не нужна: ветка Talker на Jetson работает
        # в разы медленнее, поэтому отключаем её (return_audio=False)
        response_text, audio = model.generate_response(
            conv.to_model_format(), with_audio=False
        )
    except Exception as e:
        logger.exception(f"Inference error: {e}")
        conv.history.pop()  # remove failed user message
        raise HTTPException(status_code=500, detail=str(e))

    inference_ms = (time.time() - t0) * 1000

    conv.add_assistant_message(text=response_text)

    # Log to DB
    db.log_conversation(
        session_id=session_id,
        role="user",
        text=req.text,
        inference_time=None,
    )
    db.log_conversation(
        session_id=session_id,
        role="assistant",
        text=response_text,
        inference_time=inference_ms / 1000,
    )

    return TextOnlyResponse(
        session_id=session_id,
        text=response_text,
        inference_time_ms=round(inference_ms, 1),
        search_used=search_used or rag_used,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Voice Chat
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/chat/voice", response_model=AudioResponse)
async def chat_voice(
    audio: UploadFile = File(..., description="Audio file (WAV, MP3, etc.)"),
    session_id: Optional[str] = Form(None),
    text_hint: Optional[str] = Form(None, description="Optional text context"),
    enable_search: bool = Form(True),
    tts_backend: Optional[str] = Form(
        None, description="russian_tts (Silero, по умолчанию) или model"
    ),
    persona: Optional[str] = Form(
        None, description="guide (по умолч.), mat — с матом, default"
    ),
):
    """Send audio message, get text + audio response.

    The model natively understands speech (ASR) and generates speech (TTS).
    """
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    session_id, conv = get_or_create_session(session_id, persona)

    # Save uploaded audio
    audio_dir = settings.data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_filename = f"user_{session_id}_{uuid.uuid4().hex[:8]}.wav"
    audio_path = audio_dir / audio_filename

    audio_bytes = await audio.read()
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    # Нормализация «МИРЭА»: ASR слышит «мир», «мире» и т.п. (фильтр из ветки PC)
    if text_hint:
        normalized_hint = normalize_mirea(text_hint)
        if normalized_hint != text_hint:
            logger.info(f"МИРЭА-нормализация: {text_hint!r} → {normalized_hint!r}")
            text_hint = normalized_hint

    # Build user message
    user_text = text_hint or ""

    # RAG: search uploaded documents (based on text hint if provided)
    doc_context = ""
    rag_used = False
    if settings.rag_enabled and user_text:
        doc_context = search_documents_formatted(user_text, settings.rag_top_k)
        rag_used = bool(doc_context)

    # Internet search based on text hint if provided
    search_context = ""
    search_used = False
    if enable_search and text_hint and _is_search_query(text_hint):
        search_context = search_and_format(text_hint)
        search_used = True

    # Combine contexts into the prompt
    if doc_context or search_context:
        parts = []
        if doc_context:
            parts.append(doc_context)
        if search_context:
            parts.append(search_context)
        combined = "\n\n".join(parts)
        user_text = (
            f"{user_text}\n\n{combined}\n\n"
            f"Ответь на вопрос пользователя, используя предоставленную информацию."
        )

    conv.add_user_message(text=user_text, audio_path=str(audio_path))

    t0 = time.time()
    try:
        response_text, audio_waveform = model.generate_response(
            conv.to_model_format(), tts_backend=tts_backend
        )
    except Exception as e:
        logger.exception(f"Inference error: {e}")
        conv.history.pop()
        raise HTTPException(status_code=500, detail=str(e))

    inference_ms = (time.time() - t0) * 1000

    conv.add_assistant_message(text=response_text)

    # Save assistant audio
    assistant_audio_filename = f"assistant_{session_id}_{uuid.uuid4().hex[:8]}.wav"
    assistant_audio_path = audio_dir / assistant_audio_filename

    audio_url = None
    if audio_waveform is not None:
        sf.write(
            str(assistant_audio_path),
            audio_waveform,
            settings.sample_rate,
        )
        audio_url = f"/audio/{assistant_audio_filename}"

    # Log to DB
    db.log_conversation(
        session_id=session_id,
        role="user",
        text=user_text or "[audio]",
        audio_path=str(audio_path),
    )
    db.log_conversation(
        session_id=session_id,
        role="assistant",
        text=response_text,
        audio_path=str(assistant_audio_path) if audio_waveform is not None else None,
        inference_time=inference_ms / 1000,
    )

    return AudioResponse(
        session_id=session_id,
        text=response_text,
        audio_url=audio_url,
        inference_time_ms=round(inference_ms, 1),
        search_used=search_used or rag_used,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Audio file serving
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/audio/{filename}")
async def get_audio(filename: str):
    """Download generated audio file."""
    audio_path = settings.data_dir / "audio" / filename
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(audio_path, media_type="audio/wav")


# ═══════════════════════════════════════════════════════════════════════════════
# Raw audio endpoint (returns WAV bytes directly)
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/chat/voice/raw")
async def chat_voice_raw(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    text_hint: Optional[str] = Form(None),
    tts_backend: Optional[str] = Form(
        None, description="russian_tts (Silero, по умолчанию) или model"
    ),
    persona: Optional[str] = Form(
        None, description="guide (по умолч.), mat — с матом, default"
    ),
):
    """Send audio, get raw WAV audio bytes back. For programmatic use."""
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    session_id, conv = get_or_create_session(session_id, persona)

    audio_bytes = await audio.read()
    audio_dir = settings.data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"temp_{uuid.uuid4().hex[:8]}.wav"
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    conv.add_user_message(text=text_hint or "", audio_path=str(audio_path))

    # Нормализация «МИРЭА» (фильтр из ветки PC)
    if text_hint:
        normalized_hint = normalize_mirea(text_hint)
        if normalized_hint != text_hint:
            logger.info(f"МИРЭА-нормализация: {text_hint!r} → {normalized_hint!r}")
            conv.history[-1].text = normalized_hint

    response_text, audio_waveform = model.generate_response(
        conv.to_model_format(), tts_backend=tts_backend
    )
    conv.add_assistant_message(text=response_text)

    if audio_waveform is None:
        raise HTTPException(status_code=500, detail="Model did not generate audio.")

    wav_bytes = audio_to_wav_bytes(audio_waveform)

    # Сохраняем озвученный ответ на диск — его можно переслушать/переслать
    audio_filename = f"assistant_{session_id}_{uuid.uuid4().hex[:8]}.wav"
    audio_dir = settings.data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / audio_filename).write_bytes(wav_bytes)

    # HTTP-заголовки допускают только latin-1, а ответ модели — Unicode (кириллица).
    # Поэтому кодируем в percent-encoded UTF-8, клиент делает unquote().
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Session-Id": session_id,
            "X-Response-Text": quote(response_text[:500], safe=""),
            "X-Audio-Path": str(audio_dir / audio_filename),
            "X-Audio-Url": f"/api/v1/audio/{audio_filename}",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming: текст + озвучка порциями (клиент слышит первые слова через ~1 с)
# ═══════════════════════════════════════════════════════════════════════════════

#: Предложение закончилось (по нему решаем, не служебное ли оно)
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s")
#: Буфер заканчивается законченным предложением
_SENTENCE_DONE = re.compile(r"[.!?…]\s*$")
#: Слова вместе с разделителем — чтобы не озвучивать обрывок слова
_COMPLETE_WORD = re.compile(r"\S+\s+")


def _sse(payload: dict) -> str:
    """Формат Server-Sent Events."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat/voice/stream")
async def chat_voice_stream(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    text_hint: Optional[str] = Form(None),
    tts_backend: Optional[str] = Form(None),
    persona: Optional[str] = Form(None),
    chunk_words: int = Form(
        4, description="Сколько слов озвучивать в первом куске (быстрый старт)"
    ),
    tail_words: int = Form(
        10, description="Размер следующих кусков озвучки (плавнее интонация)"
    ),
):
    """Голос → поток текста + озвучка порциями.

    Отдаёт SSE-поток событий:

    * ``{"type": "text", "delta": "..."}``  — фрагмент текста ответа
    * ``{"type": "audio", "seq": N, "wav": "<base64>", "text": "..."}``
      — озвученный кусок (~``chunk_words`` слов), готов к немедленному проигрыванию
    * ``{"type": "done", "text": "..."}``   — полный текст ответа

    Смысл: не ждать весь ответ (~2-6 с), а начинать говорить почти сразу —
    пока модель генерирует дальше.
    """
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    session_id, conv = get_or_create_session(session_id, persona)

    audio_bytes = await audio.read()
    audio_dir = settings.data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"temp_{uuid.uuid4().hex[:8]}.wav"
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    if text_hint:
        text_hint = normalize_mirea(text_hint)
    conv.add_user_message(text=text_hint or "", audio_path=str(audio_path))
    messages = conv.to_model_format()

    chunk_words = max(1, min(int(chunk_words or 4), 50))
    tail_words = max(chunk_words, min(int(tail_words or 10), 80))

    async def event_stream():
        t0 = time.time()
        full_text = ""        # всё, что сказала модель (для истории)
        speak_buffer = ""     # текст, накопленный для озвучки
        sentence_buffer = ""  # текст для поиска конца предложения
        seq = 0
        first_audio_ms = None

        # ВАЖНО: генерация — блокирующая, и если крутить её прямо в event loop,
        # сервер не успевает отправлять события (клиент получает всё в конце).
        # Поэтому генерация идёт в отдельном потоке, а сюда приходит через
        # asyncio.Queue — тогда текст и звук уходят по мере готовности.
        loop = asyncio.get_running_loop()
        pipe: asyncio.Queue = asyncio.Queue()

        def producer() -> None:
            try:
                for delta in model.generate_response_stream(messages):
                    loop.call_soon_threadsafe(pipe.put_nowait, ("text", delta))
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(pipe.put_nowait, ("error", str(exc)))
            finally:
                loop.call_soon_threadsafe(pipe.put_nowait, ("end", None))

        threading.Thread(target=producer, daemon=True).start()

        try:
            while True:
                kind, payload = await pipe.get()
                if kind == "end":
                    break
                if kind == "error":
                    logger.error(f"Ошибка генерации: {payload}")
                    yield _sse({"type": "error", "message": payload})
                    return

                delta = payload
                full_text += delta
                yield _sse({"type": "text", "delta": delta})

                sentence_buffer += delta

                # Отрезаем по одному готовому предложению за раз
                while True:
                    parts = _SENTENCE_END.split(sentence_buffer, maxsplit=1)
                    if len(parts) < 2:
                        # Последнее предложение в ответе не имеет пробела после
                        # точки, поэтому отдельно проверяем «буфер заканчивается
                        # знаком конца»
                        if _SENTENCE_DONE.search(sentence_buffer):
                            sentence, sentence_buffer = sentence_buffer, ""
                            sentence = sentence.strip()
                            if sentence and not is_courtesy_sentence(sentence):
                                speak_buffer = f"{speak_buffer} {sentence}".strip()
                        break
                    sentence, sentence_buffer = parts[0], parts[1]
                    sentence = sentence.strip()
                    # Служебные фразы («если есть вопросы, спрашивай») не озвучиваем
                    if sentence and not is_courtesy_sentence(sentence):
                        speak_buffer = f"{speak_buffer} {sentence}".strip()

                # Длинное предложение начинает звучать, не дожидаясь точки:
                # иначе ответ из одной фразы озвучивался бы только в самом конце.
                # Последние chunk_words слов остаются в буфере, чтобы никогда
                # не озвучить недописанное слово.
                buf_words = _COMPLETE_WORD.findall(sentence_buffer)
                if len(buf_words) >= chunk_words * 2:
                    head = "".join(buf_words[: len(buf_words) - chunk_words])
                    sentence_buffer = sentence_buffer[len(head):].lstrip()
                    speak_buffer = f"{speak_buffer} {head.strip()}".strip()

                # Озвучиваем порциями. Первый кусок — маленький (чтобы звук
                # пошёл почти сразу), дальше крупнее: Silero синтезирует каждый
                # кусок заново, и на 4 словах интонация рвётся.
                # Законченное предложение озвучиваем сразу, не дожидаясь
                # накопления tail_words — так интонация естественнее.
                while True:
                    need = chunk_words if seq == 0 else tail_words
                    words = _COMPLETE_WORD.findall(speak_buffer)
                    sentence_done = bool(_SENTENCE_DONE.search(speak_buffer))
                    if len(words) < chunk_words:
                        break
                    if len(words) < need and not sentence_done:
                        break
                    # Первый кусок — строго chunk_words слов: Silero синтезирует
                    # ~2 с на каждую секунду речи, и чем короче первый кусок,
                    # тем раньше звучит ответ.
                    if seq == 0:
                        take = min(len(words), chunk_words)
                    else:
                        take = len(words) if sentence_done else need
                    taken = "".join(words[:take])
                    chunk = taken.strip()
                    speak_buffer = speak_buffer[len(taken):].lstrip()
                    if not chunk:
                        continue
                    try:
                        # Silero — CPU-задача, уводим её из event loop
                        wav = await asyncio.to_thread(synthesize, chunk)
                        if first_audio_ms is None:
                            first_audio_ms = (time.time() - t0) * 1000
                            logger.info(
                                f"Стриминг: первый звук через {first_audio_ms / 1000:.2f} с"
                            )
                        yield _sse({
                            "type": "audio",
                            "seq": seq,
                            "text": chunk,
                            "wav": base64.b64encode(audio_to_wav_bytes(wav)).decode("ascii"),
                        })
                        seq += 1
                    except Exception as exc:  # озвучка не должна рвать поток
                        logger.error(f"Ошибка озвучки фрагмента: {exc}")

            # Хвост: сначала остаток озвучки, потом незакрытое предложение
            # (порядок важен — иначе слова в озвучке перепутаются)
            tail = f"{speak_buffer} {sentence_buffer}".strip()
            tail_parts = [p.strip() for p in _SENTENCE_END.split(tail) if p.strip()]
            tail = " ".join(p for p in tail_parts if not is_courtesy_sentence(p))
            if tail:
                try:
                    wav = await asyncio.to_thread(synthesize, tail)
                    if first_audio_ms is None:
                        first_audio_ms = (time.time() - t0) * 1000
                    yield _sse({
                        "type": "audio",
                        "seq": seq,
                        "text": tail,
                        "wav": base64.b64encode(audio_to_wav_bytes(wav)).decode("ascii"),
                    })
                except Exception as exc:
                    logger.error(f"Ошибка озвучки хвоста: {exc}")

            final_text = strip_courtesy(normalize_mirea(full_text))
            conv.add_assistant_message(text=final_text)
            elapsed = time.time() - t0
            logger.info(
                f"Стриминг завершён: {elapsed:.2f} с, кусков озвучки {seq}, "
                f"первый звук на {first_audio_ms / 1000 if first_audio_ms else 0:.2f} с"
            )
            yield _sse({
                "type": "done",
                "text": final_text,
                "inference_time_ms": elapsed * 1000,
                "first_audio_ms": first_audio_ms,
                "chunks": seq,
            })
        except Exception as exc:  # noqa: BLE001
            logger.exception(f"Ошибка стриминга: {exc}")
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Raw LLM endpoint (для внешних приложений, напр. WebRAgent)
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/llm/generate", response_model=LLMResponse)
async def llm_generate(req: LLMRequest):
    """Сырая генерация текста: без персоны, RAG и истории диалога.

    Используется внешними приложениями (WebRAgent и др.) как LLM-бэкенд —
    контекст (RAG, документы) формирует само приложение.
    """
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    conversation = [
        {
            "role": m.role if m.role in ("system", "user", "assistant") else "user",
            "content": [{"type": "text", "text": m.content}],
        }
        for m in req.messages
    ]

    t0 = time.time()
    try:
        response_text, _ = model.generate_response(
            conversation,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            with_audio=False,
        )
    except Exception as e:
        logger.exception(f"LLM inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return LLMResponse(
        text=response_text,
        inference_time_ms=(time.time() - t0) * 1000,
        model=settings.model_name_or_path,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Conversation management
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/sessions")
async def list_sessions():
    """List active and recent sessions."""
    active = list(sessions.keys())
    recent = db.get_recent_sessions()
    return {"active": active, "recent": recent}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Clear a conversation session."""
    if session_id in sessions:
        sessions[session_id].clear()
        del sessions[session_id]
        return {"status": "deleted", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


@router.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str, limit: int = 50):
    """Get conversation history for a session."""
    history = db.get_conversation_history(session_id, limit)
    return {
        "session_id": session_id,
        "turns": [
            {
                "role": h.role,
                "text": h.text,
                "timestamp": h.timestamp.isoformat() if h.timestamp else None,
                "inference_time": h.inference_time,
            }
            for h in reversed(history)
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Knowledge Base
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/knowledge", response_model=KnowledgeEntry)
async def add_knowledge(entry: KnowledgeCreate):
    """Add a knowledge entry to both SQL and vector store."""
    # SQL
    db_entry = db.add_knowledge(
        title=entry.title,
        content=entry.content,
        tags=entry.tags,
    )

    # Vector store (semantic search)
    try:
        vector_store.add(
            texts=[f"{entry.title}\n{entry.content}"],
            metadatas=[{"id": str(db_entry.id), "title": entry.title, "tags": entry.tags or ""}],
            ids=[f"kb_{db_entry.id}"],
        )
    except Exception as e:
        logger.warning(f"Failed to add to vector store: {e}")

    return db_entry


@router.get("/knowledge", response_model=list[KnowledgeEntry])
async def list_knowledge(limit: int = 50):
    """List all knowledge entries."""
    return db.get_all_knowledge(limit)


@router.post("/knowledge/search", response_model=KnowledgeSearchResponse)
async def search_knowledge(req: KnowledgeSearchRequest):
    """Search knowledge base (SQL LIKE + semantic)."""
    sql_results = db.search_knowledge(req.query, req.limit)

    semantic_results = []
    if req.semantic:
        semantic_results = vector_store.search(req.query, req.limit)

    return KnowledgeSearchResponse(
        sql_results=sql_results,
        semantic_results=semantic_results,
    )


@router.delete("/knowledge/{entry_id}")
async def delete_knowledge(entry_id: int):
    """Delete a knowledge entry."""
    ok = db.delete_knowledge(entry_id)
    if ok:
        try:
            vector_store.delete_by_ids([f"kb_{entry_id}"])
        except Exception:
            pass
        return {"status": "deleted", "id": entry_id}
    raise HTTPException(status_code=404, detail="Entry not found")


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _is_search_query(text: str) -> bool:
    """Heuristic: does this look like a search query?"""
    search_triggers = [
        "найди", "поищи", "расскажи о", "что такое", "кто такой",
        "сколько", "когда", "где находится", "как работает",
        "новости", "погода", "курс", "цена",
        "search", "find", "what is", "who is", "how to",
        "ищи", "загугли", "проверь", "узнай",
    ]
    text_lower = text.lower()
    return any(trigger in text_lower for trigger in search_triggers)
