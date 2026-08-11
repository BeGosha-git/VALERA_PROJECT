"""ChromaDB vector store for semantic search over saved knowledge."""

import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from config import settings


class VectorKnowledgeBase:
    """Vector store for semantic search using ChromaDB.

    Uses sentence-transformers for embeddings (runs locally).
    """

    COLLECTION_NAME = "valera_knowledge"

    def __init__(self, persist_dir: Optional[Path] = None):
        persist_dir = persist_dir or settings.chroma_path
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._embedding_fn = None
        self._collection = None

    @property
    def embedding_fn(self):
        """Lazy-load the embedding function."""
        if self._embedding_fn is None:
            from chromadb.utils import embedding_functions
            # Use a small multilingual model that runs locally
            self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="intfloat/multilingual-e5-small",
                device="cpu",  # CPU to save GPU memory
            )
        return self._embedding_fn

    @property
    def collection(self):
        """Lazy-load or create the collection."""
        if self._collection is None:
            try:
                self._collection = self.client.get_collection(
                    name=self.COLLECTION_NAME,
                    embedding_function=self.embedding_fn,
                )
            except Exception:
                self._collection = self.client.create_collection(
                    name=self.COLLECTION_NAME,
                    embedding_function=self.embedding_fn,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(f"Created ChromaDB collection: {self.COLLECTION_NAME}")
        return self._collection

    def add(
        self,
        texts: list[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[list[str]] = None,
    ) -> None:
        """Add documents to the vector store."""
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in texts]

        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"Added {len(texts)} documents to vector store.")

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        """Semantic search for similar documents.

        Returns:
            List of dicts with 'text', 'metadata', 'distance' keys.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        formatted = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                formatted.append({
                    "text": results["documents"][0][i],
                    "metadata": (
                        results["metadatas"][0][i]
                        if results["metadatas"] and results["metadatas"][0]
                        else {}
                    ),
                    "distance": (
                        results["distances"][0][i]
                        if results["distances"] and results["distances"][0]
                        else None
                    ),
                })

        return formatted

    def search_formatted(self, query: str, n_results: int = 5) -> str:
        """Search and return formatted text for the model."""
        results = self.search(query, n_results)
        if not results:
            return "В базе знаний ничего не найдено."

        lines = ["Найдено в локальной базе знаний:"]
        for i, r in enumerate(results, 1):
            text_short = r["text"][:300] + "..." if len(r["text"]) > 300 else r["text"]
            lines.append(f"{i}. {text_short}")

        return "\n".join(lines)

    def delete_by_ids(self, ids: list[str]) -> None:
        """Delete documents by IDs."""
        self.collection.delete(ids=ids)

    def count(self) -> int:
        """Number of documents in the collection."""
        return self.collection.count()


# Global vector store instance
vector_store = VectorKnowledgeBase()
