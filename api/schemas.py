"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---- Conversation ----

class TextRequest(BaseModel):
    """Request with text input."""
    text: str = Field(..., description="User text message")
    session_id: Optional[str] = Field(None, description="Session ID for multi-turn")
    enable_search: bool = Field(True, description="Enable internet search")


class AudioResponse(BaseModel):
    """Response with text and audio."""
    session_id: str
    text: str
    audio_url: Optional[str] = None  # relative URL to download audio
    inference_time_ms: float
    search_used: bool = False


class TextOnlyResponse(BaseModel):
    """Response with text only."""
    session_id: str
    text: str
    inference_time_ms: float
    search_used: bool = False


# ---- Knowledge Base ----

class KnowledgeCreate(BaseModel):
    """Create a knowledge entry."""
    title: str = Field(..., min_length=1, max_length=256)
    content: str = Field(..., min_length=1)
    tags: Optional[str] = None


class KnowledgeEntry(BaseModel):
    """Knowledge entry in responses."""
    id: int
    title: str
    content: str
    tags: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class KnowledgeSearchRequest(BaseModel):
    """Search knowledge base."""
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    semantic: bool = Field(default=True, description="Use vector semantic search")


class KnowledgeSearchResult(BaseModel):
    """Single search result."""
    text: str
    metadata: dict = {}
    distance: Optional[float] = None


class KnowledgeSearchResponse(BaseModel):
    """Response with search results."""
    sql_results: list[KnowledgeEntry] = []
    semantic_results: list[KnowledgeSearchResult] = []


# ---- Health / Info ----

class HealthResponse(BaseModel):
    """Server health status."""
    status: str = "ok"
    model_loaded: bool
    gpu_available: bool
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None


class DeviceInfo(BaseModel):
    """Audio device information."""
    id: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float


class DevicesResponse(BaseModel):
    """List of audio devices."""
    devices: list[DeviceInfo]
    default_input: Optional[int] = None
    default_output: Optional[int] = None
