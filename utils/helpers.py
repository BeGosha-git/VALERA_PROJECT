"""Helper utilities for the voice assistant."""

import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


def generate_session_id() -> str:
    """Generate a short unique session ID."""
    return str(uuid.uuid4())[:8]


def hash_text(text: str) -> str:
    """Create a short hash of text for caching/dedup."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


def format_timestamp(dt: Optional[datetime] = None) -> str:
    """Format a datetime as ISO string."""
    dt = dt or datetime.utcnow()
    return dt.isoformat()


def get_file_size_mb(filepath: Path) -> float:
    """Get file size in megabytes."""
    return filepath.stat().st_size / (1024 * 1024)


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path
