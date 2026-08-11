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


# Global database instance
db = Database()
