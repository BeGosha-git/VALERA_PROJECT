"""FastAPI routes for the voice assistant."""

import io
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from loguru import logger

from api.schemas import (
    AudioResponse,
    DevicesResponse,
    HealthResponse,
    KnowledgeCreate,
    KnowledgeEntry,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    TextOnlyResponse,
    TextRequest,
)
from config import settings
from core.audio_io import audio_to_wav_bytes, list_audio_devices, load_audio
from core.conversation import Conversation, conversation, create_new_conversation
from core.model import model
from core.search import search_and_format
from db.database import db
from db.documents import search_documents_formatted
from db.knowledge_base import vector_store

router = APIRouter()

# Conversation registry: session_id -> Conversation
sessions: dict[str, Conversation] = {"default": conversation}


def get_or_create_session(session_id: Optional[str]) -> tuple[str, Conversation]:
    """Get existing session or create a new one."""
    if session_id and session_id in sessions:
        return session_id, sessions[session_id]

    new_id = session_id or str(uuid.uuid4())[:8]
    if new_id not in sessions:
        sessions[new_id] = create_new_conversation()
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
    total = torch.cuda.get_device_properties(0).total_mem / 1024**3 if gpu_available else None

    return HealthResponse(
        status="ok",
        model_loaded=model.is_loaded,
        gpu_available=gpu_available,
        gpu_memory_used_gb=round(allocated, 2) if allocated else None,
        gpu_memory_total_gb=round(total, 2) if total else None,
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

    session_id, conv = get_or_create_session(req.session_id)

    # 1. RAG: search uploaded documents
    doc_context = ""
    rag_used = False
    if settings.rag_enabled:
        doc_context = search_documents_formatted(req.text, settings.rag_top_k)
        rag_used = bool(doc_context)

    # 2. Internet search if enabled and query looks like a search
    search_context = ""
    search_used = False
    if req.enable_search and _is_search_query(req.text):
        search_context = search_and_format(req.text)
        search_used = True

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
        response_text, audio = model.generate_response(conv.to_model_format())
    except Exception as e:
        logger.error(f"Inference error: {e}")
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
):
    """Send audio message, get text + audio response.

    The model natively understands speech (ASR) and generates speech (TTS).
    """
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    session_id, conv = get_or_create_session(session_id)

    # Save uploaded audio
    audio_dir = settings.data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_filename = f"user_{session_id}_{uuid.uuid4().hex[:8]}.wav"
    audio_path = audio_dir / audio_filename

    audio_bytes = await audio.read()
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

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
            conv.to_model_format()
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
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
):
    """Send audio, get raw WAV audio bytes back. For programmatic use."""
    if not model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    session_id, conv = get_or_create_session(session_id)

    audio_bytes = await audio.read()
    audio_dir = settings.data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"temp_{uuid.uuid4().hex[:8]}.wav"
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    conv.add_user_message(text=text_hint or "", audio_path=str(audio_path))

    response_text, audio_waveform = model.generate_response(conv.to_model_format())
    conv.add_assistant_message(text=response_text)

    if audio_waveform is None:
        raise HTTPException(status_code=500, detail="Model did not generate audio.")

    wav_bytes = audio_to_wav_bytes(audio_waveform)
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Session-Id": session_id,
            "X-Response-Text": response_text[:500],
        },
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
