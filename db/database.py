"""SQLite database for storing conversation history, settings, and knowledge."""

from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import settings


class Base(DeclarativeBase):
    pass


class ConversationLog(Base):
    """Stored conversation turns."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), index=True, nullable=False)
    role = Column(String(16), nullable=False)  # "user" or "assistant"
    text = Column(Text, nullable=True)
    audio_path = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    inference_time = Column(Float, nullable=True)  # seconds


class KnowledgeEntry(Base):
    """User-saved knowledge entries (facts, notes, etc.)."""

    __tablename__ = "knowledge"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(256), nullable=False)
    content = Column(Text, nullable=False)
    tags = Column(String(512), nullable=True)  # comma-separated tags
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Document(Base):
    """Uploaded documents (.doc, .docx, .pdf, .txt, .md) for the knowledge base.

    The content is chunked and embedded into ChromaDB for semantic retrieval.
    """

    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(512), nullable=False)       # original file name
    stored_path = Column(String(1024), nullable=False)   # path on disk
    file_type = Column(String(32), nullable=False)       # .doc / .docx / .pdf / ...
    file_size = Column(Integer, nullable=False)          # bytes
    num_chunks = Column(Integer, default=0)              # chunks stored in vector DB
    status = Column(String(32), default="processing")    # processing | ready | error
    error = Column(Text, nullable=True)                  # error message if failed
    created_at = Column(DateTime, default=datetime.utcnow)


class Database:
    """SQLite database manager."""

    def __init__(self, db_path: Optional[Path] = None):
        db_path = db_path or settings.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(
            f"sqlite:///{db_path}",
            echo=settings.db_echo,
            connect_args={"check_same_thread": False},
        )
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def init_db(self) -> None:
        """Create all tables."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created.")

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    # ---- Conversation methods ----

    def log_conversation(
        self,
        session_id: str,
        role: str,
        text: Optional[str] = None,
        audio_path: Optional[str] = None,
        inference_time: Optional[float] = None,
    ) -> ConversationLog:
        """Log a conversation turn."""
        with self.get_session() as sess:
            entry = ConversationLog(
                session_id=session_id,
                role=role,
                text=text,
                audio_path=audio_path,
                inference_time=inference_time,
            )
            sess.add(entry)
            sess.commit()
            sess.refresh(entry)
            return entry

    def get_conversation_history(
        self, session_id: str, limit: int = 50
    ) -> list[ConversationLog]:
        """Retrieve conversation history for a session."""
        with self.get_session() as sess:
            return (
                sess.query(ConversationLog)
                .filter(ConversationLog.session_id == session_id)
                .order_by(ConversationLog.timestamp.desc())
                .limit(limit)
                .all()
            )

    def get_recent_sessions(self, limit: int = 10) -> list[str]:
        """Get list of recent session IDs."""
        with self.get_session() as sess:
            rows = (
                sess.query(ConversationLog.session_id)
                .distinct()
                .order_by(ConversationLog.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [r[0] for r in rows]

    # ---- Knowledge methods ----

    def add_knowledge(
        self, title: str, content: str, tags: Optional[str] = None
    ) -> KnowledgeEntry:
        """Add a knowledge entry."""
        with self.get_session() as sess:
            entry = KnowledgeEntry(title=title, content=content, tags=tags)
            sess.add(entry)
            sess.commit()
            sess.refresh(entry)
            return entry

    def search_knowledge(self, query: str, limit: int = 10) -> list[KnowledgeEntry]:
        """Search knowledge entries by title/content/tags (LIKE search)."""
        with self.get_session() as sess:
            pattern = f"%{query}%"
            return (
                sess.query(KnowledgeEntry)
                .filter(
                    (KnowledgeEntry.title.like(pattern))
                    | (KnowledgeEntry.content.like(pattern))
                    | (KnowledgeEntry.tags.like(pattern))
                )
                .order_by(KnowledgeEntry.updated_at.desc())
                .limit(limit)
                .all()
            )

    def get_all_knowledge(self, limit: int = 50) -> list[KnowledgeEntry]:
        """Get all knowledge entries."""
        with self.get_session() as sess:
            return (
                sess.query(KnowledgeEntry)
                .order_by(KnowledgeEntry.updated_at.desc())
                .limit(limit)
                .all()
            )

    def delete_knowledge(self, entry_id: int) -> bool:
        """Delete a knowledge entry by ID."""
        with self.get_session() as sess:
            entry = sess.query(KnowledgeEntry).filter(
                KnowledgeEntry.id == entry_id
            ).first()
            if entry:
                sess.delete(entry)
                sess.commit()
                return True
            return False

    # ---- Document methods ----

    def add_document(
        self,
        filename: str,
        stored_path: str,
        file_type: str,
        file_size: int,
    ) -> Document:
        """Register a new uploaded document."""
        with self.get_session() as sess:
            doc = Document(
                filename=filename,
                stored_path=stored_path,
                file_type=file_type,
                file_size=file_size,
            )
            sess.add(doc)
            sess.commit()
            sess.refresh(doc)
            return doc

    def update_document(
        self,
        doc_id: int,
        num_chunks: Optional[int] = None,
        status: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Optional[Document]:
        """Update document processing status."""
        with self.get_session() as sess:
            doc = sess.query(Document).filter(Document.id == doc_id).first()
            if not doc:
                return None
            if num_chunks is not None:
                doc.num_chunks = num_chunks
            if status is not None:
                doc.status = status
            if error is not None:
                doc.error = error
            sess.commit()
            sess.refresh(doc)
            return doc

    def get_document(self, doc_id: int) -> Optional[Document]:
        """Get a document by ID."""
        with self.get_session() as sess:
            return sess.query(Document).filter(Document.id == doc_id).first()

    def get_all_documents(self, limit: int = 100) -> list[Document]:
        """List all uploaded documents."""
        with self.get_session() as sess:
            return (
                sess.query(Document)
                .order_by(Document.created_at.desc())
                .limit(limit)
                .all()
            )

    def delete_document(self, doc_id: int) -> bool:
        """Delete a document record."""
        with self.get_session() as sess:
            doc = sess.query(Document).filter(Document.id == doc_id).first()
            if doc:
                sess.delete(doc)
                sess.commit()
                return True
            return False


# Global database instance
db = Database()
