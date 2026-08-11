"""Document upload and management API endpoints.

Supports: .doc, .docx, .pdf, .txt, .md
Documents are parsed, chunked, and embedded into ChromaDB for semantic search.
"""

import threading
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from loguru import logger
from pydantic import BaseModel

from config import settings
from db.database import db
from db.documents import (
    delete_document_chunks,
    extract_text,
    get_document_count,
    index_document,
    search_documents,
)

router = APIRouter()

ALLOWED_EXTENSIONS = {".doc", ".docx", ".pdf", ".txt", ".md", ".markdown", ".rtf", ".log"}

# Processing lock to avoid concurrent ChromaDB writes
_processing_lock = threading.Lock()


class DocumentOut(BaseModel):
    """Document metadata in responses."""

    id: int
    filename: str
    file_type: str
    file_size: int
    num_chunks: int
    status: str
    error: Optional[str] = None
    created_at: str

    model_config = {"from_attributes": True}


class DocumentSearchRequest(BaseModel):
    """Search query across documents."""

    query: str
    top_k: int = 5


class DocumentSearchHit(BaseModel):
    """A single search hit."""

    text: str
    filename: str
    doc_id: str
    chunk_index: int
    distance: Optional[float] = None


class DocumentSearchResponse(BaseModel):
    """Search response with hits."""

    query: str
    total_chunks: int
    results: list[DocumentSearchHit]


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/documents/upload", response_model=DocumentOut)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document. It will be parsed, chunked, and embedded.

    Returns immediately with the document record (status="processing").
    Check the status via GET /documents/{id} — indexing happens in background.
    """
    filename = file.filename or "unnamed"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # Save file
    settings.ensure_dirs()
    stored_name = f"{file.filename or 'doc'}"
    stored_path = settings.documents_dir / stored_name

    # Avoid collisions
    counter = 1
    while stored_path.exists():
        stored_path = settings.documents_dir / f"{Path(stored_name).stem}_{counter}{ext}"
        counter += 1

    content = await file.read()
    stored_path.write_bytes(content)

    doc = db.add_document(
        filename=filename,
        stored_path=str(stored_path),
        file_type=ext,
        file_size=len(content),
    )

    # Process in background
    threading.Thread(
        target=_process_document,
        args=(doc.id, stored_path, filename),
        daemon=True,
    ).start()

    return _to_out(doc)


@router.get("/documents", response_model=list[DocumentOut])
async def list_documents():
    """List all uploaded documents."""
    docs = db.get_all_documents()
    return [_to_out(d) for d in docs]


@router.get("/documents/{doc_id}", response_model=DocumentOut)
async def get_document(doc_id: int):
    """Get a single document by ID (check processing status)."""
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return _to_out(doc)


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: int):
    """Delete a document and all its embedded chunks."""
    doc = db.get_document(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Remove from vector store
    delete_document_chunks(doc_id)

    # Remove file from disk
    try:
        Path(doc.stored_path).unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Could not delete file: {e}")

    db.delete_document(doc_id)
    return {"status": "deleted", "id": doc_id, "filename": doc.filename}


@router.post("/documents/search", response_model=DocumentSearchResponse)
async def search_documents_endpoint(req: DocumentSearchRequest):
    """Semantic search across all uploaded documents."""
    results = search_documents(req.query, req.top_k)
    return DocumentSearchResponse(
        query=req.query,
        total_chunks=get_document_count(),
        results=[
            DocumentSearchHit(
                text=r["text"],
                filename=r["filename"],
                doc_id=r["doc_id"],
                chunk_index=int(r["chunk_index"]),
                distance=r["distance"],
            )
            for r in results
        ],
    )


@router.get("/documents/stats")
async def document_stats():
    """Statistics about the document knowledge base."""
    docs = db.get_all_documents()
    return {
        "documents": len(docs),
        "total_chunks": get_document_count(),
        "ready": sum(1 for d in docs if d.status == "ready"),
        "processing": sum(1 for d in docs if d.status == "processing"),
        "errors": sum(1 for d in docs if d.status == "error"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _process_document(doc_id: int, stored_path: Path, filename: str):
    """Background task: parse, chunk, embed a document."""
    with _processing_lock:
        try:
            logger.info(f"Processing document: {filename} (id={doc_id})")
            text = extract_text(stored_path)

            if not text.strip():
                db.update_document(doc_id, num_chunks=0, status="error",
                                   error="Документ пуст или текст не извлечён")
                return

            num_chunks = index_document(doc_id, filename, text)
            db.update_document(doc_id, num_chunks=num_chunks, status="ready")
            logger.info(f"Document ready: {filename} ({num_chunks} chunks)")

        except Exception as e:
            logger.error(f"Document processing failed: {filename}: {e}")
            db.update_document(
                doc_id, num_chunks=0, status="error", error=str(e)
            )


def _to_out(doc) -> DocumentOut:
    """Convert DB document to response model."""
    return DocumentOut(
        id=doc.id,
        filename=doc.filename,
        file_type=doc.file_type,
        file_size=doc.file_size,
        num_chunks=doc.num_chunks,
        status=doc.status,
        error=doc.error,
        created_at=doc.created_at.isoformat() if doc.created_at else "",
    )
