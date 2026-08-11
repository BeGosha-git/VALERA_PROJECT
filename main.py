"""QWEN-VALERA — Voice Assistant with Qwen3-Omni.

End-to-end voice assistant with:
- Native audio understanding (ASR) and speech generation (TTS)
- Internet search (DuckDuckGo)
- Internal SQL + vector database
- REST API for integration
"""

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routes import router
from api.ws import ws_router
from config import settings
from core.model import model
from db.database import db

# ── Logging setup ────────────────────────────────────────────────────────────

logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
)
logger.add(
    settings.data_dir / "valera.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG",
)


# ── Application lifecycle ────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown."""
    logger.info("=" * 60)
    logger.info("QWEN-VALERA Voice Assistant starting...")
    logger.info(f"Model: {settings.model_name_or_path}")
    logger.info(f"Speaker: {settings.speaker_voice}")
    logger.info("=" * 60)

    # Init database
    settings.ensure_dirs()
    db.init_db()

    # Load model
    try:
        model.load()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("Server will start but model is not loaded.")
        logger.warning("Call POST /admin/reload to retry.")

    yield

    # Shutdown
    logger.info("Shutting down...")
    model.unload()


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="QWEN-VALERA Voice Assistant",
    description="End-to-end voice assistant powered by Qwen3-Omni-30B-A3B",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router, prefix="/api/v1")

# WebSocket routes
app.include_router(ws_router)


# ── Admin endpoints ──────────────────────────────────────────────────────────

@app.post("/admin/reload")
async def reload_model():
    """Reload the model (e.g., after download)."""
    model.unload()
    try:
        model.load()
        return {"status": "model reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "name": "QWEN-VALERA",
        "version": "1.0.0",
        "model_loaded": model.is_loaded,
        "docs": "/docs",
    }


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level="info",
    )
