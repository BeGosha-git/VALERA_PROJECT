"""WebSocket endpoints for real-time voice interaction.

Flow:
1. Client connects to /ws/chat
2. Client sends audio (WAV bytes) or text
3. Server responds with text + audio
"""

import io
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from config import settings
from core.audio_io import audio_to_wav_bytes
from core.conversation import Conversation, create_new_conversation
from core.model import model
from core.search import search_and_format

ws_router = APIRouter()

# Active WebSocket sessions: ws_id -> {conversation, session_id}
ws_sessions: dict[str, dict] = {}


@ws_router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    """Real-time voice chat over WebSocket.

    Message types (JSON envelope with base64 or raw audio):
    - {"type": "audio", "audio_b64": "..."}       # user audio (WAV)
    - {"type": "text", "text": "..."}             # user text
    - {"type": "reset"}                           # clear conversation
    - {"type": "search", "query": "..."}          # do a search, returns result

    Server responses:
    - {"type": "text", "text": "...", "session_id": "..."}
    - {"type": "audio", "audio_b64": "...", "text": "..."}  # assistant audio
    - {"type": "error", "message": "..."}
    """
    await websocket.accept()

    session_id = str(uuid.uuid4())[:8]
    conv: Conversation = create_new_conversation()
    ws_sessions[session_id] = {"conversation": conv, "ws": websocket}

    logger.info(f"WebSocket connected: session={session_id}")

    try:
        while True:
            # Receive message
            msg = await websocket.receive_json()

            msg_type = msg.get("type", "")

            if msg_type == "reset":
                conv.clear()
                await websocket.send_json({"type": "ok", "message": "Conversation cleared"})
                continue

            if msg_type == "search":
                query = msg.get("query", "")
                if not query:
                    await websocket.send_json({"type": "error", "message": "Empty query"})
                    continue
                result = search_and_format(query)
                await websocket.send_json({"type": "search_result", "text": result})
                continue

            if not model.is_loaded:
                await websocket.send_json({
                    "type": "error",
                    "message": "Model not loaded yet",
                })
                continue

            # Build user message
            user_text = msg.get("text", "")
            user_audio_path = None

            if msg_type == "audio":
                # Save received audio
                import base64
                audio_b64 = msg.get("audio_b64", "")
                if not audio_b64:
                    await websocket.send_json({"type": "error", "message": "No audio data"})
                    continue

                audio_bytes = base64.b64decode(audio_b64)
                audio_dir = settings.data_dir / "audio"
                audio_dir.mkdir(parents=True, exist_ok=True)
                user_audio_path = str(
                    audio_dir / f"ws_{session_id}_{uuid.uuid4().hex[:8]}.wav"
                )
                with open(user_audio_path, "wb") as f:
                    f.write(audio_bytes)

            # Search enhancement
            if settings.search_enabled and user_text:
                from api.routes import _is_search_query
                if _is_search_query(user_text):
                    search_ctx = search_and_format(user_text)
                    if search_ctx:
                        user_text = (
                            f"{user_text}\n\n{search_ctx}\n\n"
                            f"Ответь на вопрос пользователя, используя результаты поиска."
                        )

            conv.add_user_message(text=user_text, audio_path=user_audio_path)

            # Generate
            t0 = time.time()
            response_text, audio_waveform = model.generate_response(
                conv.to_model_format()
            )
            inference_ms = (time.time() - t0) * 1000

            conv.add_assistant_message(text=response_text)

            # Send text response
            await websocket.send_json({
                "type": "text",
                "text": response_text,
                "session_id": session_id,
                "inference_time_ms": round(inference_ms, 1),
            })

            # Send audio response if available
            if audio_waveform is not None:
                import base64
                wav_bytes = audio_to_wav_bytes(audio_waveform)
                audio_b64 = base64.b64encode(wav_bytes).decode("utf-8")
                await websocket.send_json({
                    "type": "audio",
                    "audio_b64": audio_b64,
                    "text": response_text,
                    "sample_rate": settings.sample_rate,
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session={session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        ws_sessions.pop(session_id, None)
