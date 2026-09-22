"""QWEN-VALERA — Voice Assistant with Qwen3-Omni.

End-to-end voice assistant with:
- Native audio understanding (ASR) and speech generation (TTS)
- Internet search (DuckDuckGo)
- Internal SQL + vector database
- REST API for integration
"""

import sys
from contextlib import asynccontextmanager

from loguru import logger

from config import settings

# ── Pre-flight: порт занят? ──────────────────────────────────────────────────
# Загрузка модели занимает 30 с – 3 мин. Если порт уже занят другим
# экземпляром, uvicorn узнает об этом только ПОСЛЕ загрузки и упадёт.
# Поэтому проверяем порт заранее — ещё ДО импорта torch/transformers,
# чтобы остановиться за секунду, а не за три минуты.


def port_is_busy(port: int, host: str = "0.0.0.0") -> bool:
    """Занят ли порт (пробуем забиндиться и сразу отпускаем)."""
    import socket

    addr = "" if host in ("0.0.0.0", "::", "*") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((addr, port))
            return False
        except OSError:
            return True


def find_port_owner(port: int) -> tuple:
    """Вернуть (pid, cmdline) процесса, слушающего порт."""
    import re
    import subprocess

    for cmd in (["ss", "-lptnH"], ["ss", "-ltnp"]):
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5
            ).stdout
        except Exception:
            continue
        for line in out.splitlines():
            fields = line.split()
            if len(fields) < 4 or not fields[3].endswith(f":{port}"):
                continue
            match = re.search(r"pid=(\d+)", line)
            if not match:
                return None, None
            pid = int(match.group(1))
            try:
                raw = open(f"/proc/{pid}/cmdline", "rb").read()
                cmdline = raw.replace(b"\x00", b" ").decode(errors="replace").strip()
            except Exception:
                cmdline = ""
            return pid, cmdline or None
    return None, None


def ensure_port_free(port: int, host: str, force: bool) -> None:
    """Остановиться сразу, если порт занят (или убить старый сервер)."""
    import os
    import signal
    import time

    if not port_is_busy(port, host):
        return

    pid, cmdline = find_port_owner(port)
    who = f"PID {pid} — {cmdline}" if pid else "процесс не определён (нет прав?)"

    if force and pid:
        logger.warning(f"Порт {port} занят ({who}). Останавливаю старый сервер...")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(30):
            time.sleep(0.5)
            if not port_is_busy(port, host):
                logger.info(f"✓ Старый сервер (PID {pid}) остановлен")
                return
        logger.warning(f"PID {pid} не ответил на SIGTERM — SIGKILL")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for _ in range(20):
            time.sleep(0.5)
            if not port_is_busy(port, host):
                logger.info(f"✓ Старый сервер (PID {pid}) снят")
                return
        logger.error(f"Порт {port} всё ещё занят — освободите его вручную")
        sys.exit(1)

    logger.error("=" * 60)
    logger.error(f"ОСТАНОВКА: порт {port} уже занят!")
    logger.error(f"  Кто: {who}")
    logger.error("  Модель загружаться не будет (иначе потеряете 1–3 минуты).")
    logger.error("")
    logger.error("  Что делать:")
    logger.error("    python main.py --force   # остановить старый сервер и запустить")
    logger.error(f"    kill {pid}" if pid else "    kill <PID>")
    logger.error("=" * 60)
    sys.exit(1)


# Проверка ДО тяжёлых импортов (torch/transformers ≈ 10 с)
if __name__ == "__main__":
    import os as _os

    ensure_port_free(
        settings.api_port,
        settings.api_host,
        "--force" in sys.argv or _os.getenv("VALERA_FORCE_KILL", "") == "1",
    )

# ── Тяжёлые импорты (torch, transformers, fastapi) ───────────────────────────

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from api.document_routes import router as document_router  # noqa: E402
from api.routes import router  # noqa: E402
from api.ws import ws_router  # noqa: E402
from core.model import model  # noqa: E402
from db.database import db  # noqa: E402

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


# ── Pre-flight: порт занят? ────────────────────────────────────
# Загрузка модели занимает 30 с – 3 мин. Если порт уже занят другим
# экземпляром, uvicorn узнает об этом только ПОСЛЕ загрузки и упадёт.
# Поэтому проверяем порт заранее и останавливаемся сразу.


def port_is_busy(port: int, host: str = "0.0.0.0") -> bool:
    """Занят ли порт (пробуем забиндиться и сразу отпускаем)."""
    import socket

    addr = "" if host in ("0.0.0.0", "::", "*") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((addr, port))
            return False
        except OSError:
            return True


def find_port_owner(port: int) -> tuple[int | None, str | None]:
    """Вернуть (pid, cmdline) процесса, слушающего порт."""
    import re
    import subprocess

    for cmd in (["ss", "-lptnH"], ["ss", "-ltnp"]):
        try:
            out = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5
            ).stdout
        except Exception:
            continue
        for line in out.splitlines():
            fields = line.split()
            if len(fields) < 4 or not fields[3].endswith(f":{port}"):
                continue
            match = re.search(r"pid=(\d+)", line)
            if not match:
                return None, None
            pid = int(match.group(1))
            try:
                raw = open(f"/proc/{pid}/cmdline", "rb").read()
                cmdline = raw.replace(b"\x00", b" ").decode(errors="replace").strip()
            except Exception:
                cmdline = ""
            return pid, cmdline or None
    return None, None


def ensure_port_free(port: int, host: str, force: bool) -> None:
    """Остановиться сразу, если порт занят (или убить старый сервер)."""
    import os
    import signal
    import time

    if not port_is_busy(port, host):
        return

    pid, cmdline = find_port_owner(port)
    who = f"PID {pid} — {cmdline}" if pid else "процесс не определён (нет прав?)"

    if force and pid:
        logger.warning(f"Порт {port} занят ({who}). Останавливаю старый сервер...")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(30):
            time.sleep(0.5)
            if not port_is_busy(port, host):
                logger.info(f"✓ Старый сервер (PID {pid}) остановлен")
                return
        logger.warning(f"PID {pid} не ответил на SIGTERM — SIGKILL")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for _ in range(20):
            time.sleep(0.5)
            if not port_is_busy(port, host):
                logger.info(f"✓ Старый сервер (PID {pid}) снят")
                return
        logger.error(f"Порт {port} всё ещё занят — освободите его вручную")
        sys.exit(1)

    logger.error("=" * 60)
    logger.error(f"ОСТАНОВКА: порт {port} уже занят!")
    logger.error(f"  Кто: {who}")
    logger.error("  Модель загружаться не будет (иначе потеряете 1–3 минуты).")
    logger.error("")
    logger.error("  Что делать:")
    logger.error(f"    python main.py --force   # остановить старый сервер и запустить")
    logger.error(f"    kill {pid}                  # либо вручную" if pid else "    kill <PID>")
    logger.error("=" * 60)
    sys.exit(1)


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

    # Прогрев: первый запрос не должен платить за загрузку моделей.
    # Без прогрева первый поиск по базе занимал ~6 с (загрузка эмбеддингов),
    # а первый синтез — ~1 с (загрузка Silero).
    import time as _time

    try:
        from db.documents import search_documents_formatted

        _t0 = _time.time()
        search_documents_formatted("прогрев", top_k=1)
        logger.info(f"✓ Эмбеддинги прогреты за {_time.time() - _t0:.1f} с")
    except Exception as e:  # прогрев не должен мешать запуску
        logger.warning(f"Не удалось прогреть поиск: {e}")

    try:
        from core.tts import tts_uses_model

        if not tts_uses_model():
            from core.tts import get_silero

            _t0 = _time.time()
            get_silero()._ensure_loaded()
            logger.info(f"✓ Silero TTS прогрет за {_time.time() - _t0:.1f} с")
    except Exception as e:
        logger.warning(f"Не удалось прогреть TTS: {e}")

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

# Document management routes
app.include_router(document_router, prefix="/api/v1")

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

    # Порт уже проверен выше (до импорта torch)
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level="info",
    )
