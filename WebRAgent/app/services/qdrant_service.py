import os
import json
import requests
import threading
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from app.services.model_service import ModelService


# ---------------------------------------------------------------------------
# ГЛОБАЛЬНЫЙ КЭШ МОДЕЛЕЙ ЭМБЕДДИНГОВ
# ---------------------------------------------------------------------------
# Раньше SentenceTransformer создавался В КАЖДОМ экземпляре QdrantService, а
# QdrantService → DocumentService → RAGService создаются заново на КАЖДЫЙ
# HTTP-запрос. Поэтому модель грузилась с диска по нескольку раз в секунду
# (1.7-2.6с на тёплом кэше и до 17с на холодном) — поиск по базе казался
# «долгим и неработающим», хотя находил результаты.
#
# Теперь модель грузится ОДИН раз на процесс и переиспользуется всеми
# экземплярами. Повторный запрос — ~0.01с вместо ~2с.
_ST_MODEL_CACHE: dict = {}
_ST_MODEL_LOCK = threading.Lock()


def get_sentence_transformer(name: str) -> "SentenceTransformer":
    """
    Возвращает SentenceTransformer из общего кэша процесса.

    Загружает модель только при первом обращении к данному имени; дальше
    отдаёт уже готовый объект. Потокобезопасно.
    """
    with _ST_MODEL_LOCK:
        model = _ST_MODEL_CACHE.get(name)
        if model is None:
            model = SentenceTransformer(name)
            _ST_MODEL_CACHE[name] = model
        return model


class QdrantService:
    """Service for interacting with Qdrant vector database"""
    
    def __init__(self):
        """Initialize Qdrant client and embedding model"""
        self.host = os.getenv('QDRANT_HOST', 'localhost')
        self.port = int(os.getenv('QDRANT_PORT', 6333))
        # timeout=60: при заливке больших документов (сотни чанков подряд)
        # дефолтные 5 секунд слишком малы — вставки падали с «timed out».
        self.client = QdrantClient(host=self.host, port=self.port, timeout=60)
        
        # Initialize embedding model dimensions lookup - will be populated dynamically
        self.model_dimensions = {}
        
        # Load known dimensions from config file
        self._load_model_dimensions()
        
        # Load model configuration from JSON config
        self.model_service = ModelService()
        self.config = self.model_service.config
        
        # Initialize embedding model based on active configuration
        self.active_config = self.config["active"]
        self.llm_provider = self.active_config["llm_provider"]
        self.vector_size = 384  # Default for sentence-transformers
        
        if self.llm_provider == 'openai':
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
            openai_model = self.active_config["models"]["openai_embedding"] or 'text-embedding-3-small'
            self.vector_size = self.get_vector_size('openai', openai_model)
        elif self.llm_provider == 'ollama':
            self.ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
            self.ollama_model = self.active_config["models"]["ollama_embedding"] or 'nomic-embed-text'
            self.vector_size = self.get_vector_size('ollama', self.ollama_model)
        else:
            # For Claude or fallback to sentence-transformers
            model_name = self.active_config["models"]["embedding"] or 'all-MiniLM-L6-v2'
            self.embedding_model = get_sentence_transformer(model_name)
            self.embedding_model_name = model_name
            self.vector_size = self.get_vector_size('embedding', model_name)
    
    def get_embedding(self, text, provider=None, model=None):
        """
        Generate embedding for text based on provided parameters or configured provider
        
        Args:
            text (str): Text to embed
            provider (str, optional): The provider to use (overrides self.llm_provider)
            model (str, optional): The specific model to use
            
        Returns:
            list: Vector embedding
        """
        # Use provided parameters if available, otherwise use defaults
        provider = provider or self.llm_provider
        
        # Ensure we have required attributes for the chosen provider
        self._ensure_provider_initialized(provider)
        
        # OpenAI embeddings
        if provider == 'openai':
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_api_key)
                # Use specific model if provided, otherwise use default
                embedding_model = model or "text-embedding-3-small"
                response = client.embeddings.create(
                    input=text,
                    model=embedding_model
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"Error with OpenAI embedding: {e}")
                # Fallback to sentence-transformers
                fallback_model = 'all-MiniLM-L6-v2'
                if not hasattr(self, 'embedding_model') or self.embedding_model_name != fallback_model:
                    self.embedding_model = get_sentence_transformer(fallback_model)
                    self.embedding_model_name = fallback_model
                return self.embedding_model.encode(text).tolist()
                
        # Ollama embeddings
        elif provider == 'ollama':
            # Если Ollama-эмбеддинги уже не отвечали (модели нет), не долбимся
            # в неё на КАЖДЫЙ вопрос: это лишний сетевой заход — 404 — и задержка.
            if getattr(self, '_ollama_embed_dead', False):
                return self._fallback_embed(text)
            try:
                # Use model from parameters, instance variable, or get from JSON config
                if model:
                    embedding_model = model
                elif hasattr(self, 'ollama_model'):
                    embedding_model = self.ollama_model
                elif hasattr(self, 'active_config') and self.active_config["models"]["ollama_embedding"]:
                    embedding_model = self.active_config["models"]["ollama_embedding"]
                else:
                    embedding_model = 'nomic-embed-text'

                response = requests.post(
                    f"{self.ollama_host}/api/embeddings",
                    json={
                        "model": embedding_model,
                        "prompt": text
                    },
                    timeout=10,
                )
                response.raise_for_status()
                vec = response.json().get('embedding', [])
                if not vec:
                    raise ValueError("пустой вектор от Ollama")
                return vec
            except Exception as e:
                print(f"Error with Ollama embedding: {e} — перехожу на локальную модель")
                # Запоминаем и больше не пробуем в этом процессе.
                self._ollama_embed_dead = True
                return self._fallback_embed(text)
        
        # Default: use sentence-transformers (embedding provider)
        else:
            # Fix model name by removing tags like ":latest"
            if model and ":" in model:
                model = model.split(":")[0]
            
            # Use model from parameters, instance variable, or get from JSON config
            if model:
                embedding_model = model
            elif hasattr(self, 'embedding_model_name'):
                embedding_model = self.embedding_model_name
            elif hasattr(self, 'active_config') and self.active_config["models"]["embedding"]:
                embedding_model = self.active_config["models"]["embedding"]
            else:
                embedding_model = 'all-MiniLM-L6-v2'
                
            try:
                if not hasattr(self, 'embedding_model') or self.embedding_model_name != embedding_model:
                    self.embedding_model = get_sentence_transformer(embedding_model)
                    self.embedding_model_name = embedding_model
                return self.embedding_model.encode(text).tolist()
            except Exception as e:
                print(f"Error with model {embedding_model}: {e}")
                # Fall back to known working model
                fallback_model = 'all-MiniLM-L6-v2'
                if not hasattr(self, 'embedding_model') or self.embedding_model_name != fallback_model:
                    self.embedding_model = get_sentence_transformer(fallback_model)
                    self.embedding_model_name = fallback_model
                return self.embedding_model.encode(text).tolist()
                
    def _fallback_embed(self, text):
        """Локальная модель эмбеддингов — когда внешний провайдер недоступен."""
        fallback_model = 'all-MiniLM-L6-v2'
        if not hasattr(self, 'embedding_model') or self.embedding_model_name != fallback_model:
            self.embedding_model = get_sentence_transformer(fallback_model)
            self.embedding_model_name = fallback_model
        return self.embedding_model.encode(text).tolist()

    def _load_model_dimensions(self):
        """Load model dimensions from config file"""
        config_path = os.path.join('data', 'models', 'dimensions.json')
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.model_dimensions = json.load(f)
            else:
                # Create initial dimensions file with some common models
                initial_dimensions = {
                    # OpenAI models
                    'text-embedding-3-small': 1536,
                    'text-embedding-3-large': 3072,
                    'text-embedding-ada-002': 1536,
                    
                    # Ollama models
                    'nomic-embed-text': 768,
                    'all-minilm': 384,
                    'mxbai-embed-large': 1024,
                    'mxbai-embed-base': 768,
                    
                    # Sentence transformers models
                    'all-MiniLM-L6-v2': 384,
                    'all-mpnet-base-v2': 768,
                    'paraphrase-multilingual-MiniLM-L12-v2': 384,
                    'paraphrase-albert-small-v2': 768,
                    'multi-qa-mpnet-base-dot-v1': 768,
                    'multi-qa-MiniLM-L6-cos-v1': 384,
                }
                self.model_dimensions = initial_dimensions
                self._save_model_dimensions()
        except Exception as e:
            print(f"Error loading model dimensions: {e}")
            # Start with empty dict, will be populated dynamically
            self.model_dimensions = {}
    
    def _save_model_dimensions(self):
        """Save model dimensions to config file"""
        config_path = os.path.join('data', 'models', 'dimensions.json')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self.model_dimensions, f, indent=2)
        except Exception as e:
            print(f"Error saving model dimensions: {e}")
    
    def get_vector_size(self, provider, model_name):
        """
        Get vector size for a specific model by making a test embedding request
        
        Args:
            provider (str): The embedding provider (openai, ollama, embedding)
            model_name (str): The specific model name
            
        Returns:
            int: Vector size for the model
            
        Raises:
            ValueError: If dimensions cannot be determined
        """
        # Make sure provider attributes are initialized
        self._ensure_provider_initialized(provider)
        
        # Check if we've cached this model's dimensions
        if model_name in self.model_dimensions:
            return self.model_dimensions[model_name]
            
        # Determine dimensions dynamically
        dynamic_size = self._determine_dimensions_dynamically(provider, model_name)
        
        if dynamic_size and dynamic_size > 0:
            # Cache this for future use
            self.model_dimensions[model_name] = dynamic_size
            # Save to persistent storage
            self._save_model_dimensions()
            return dynamic_size
        
        # If we got here, we failed to determine dimensions
        raise ValueError(f"Could not determine embedding dimensions for {provider}/{model_name}")
                
    def _determine_dimensions_dynamically(self, provider, model_name):
        """
        Determine embedding dimensions by making a test request
        
        Args:
            provider (str): The provider to use
            model_name (str): The model to test
            
        Returns:
            int: The determined vector size, or None if couldn't determine
        """
        test_text = "hello world"
        
        try:
            if provider == 'openai':
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_api_key)
                response = client.embeddings.create(
                    input=test_text,
                    model=model_name
                )
                return len(response.data[0].embedding)
                
            elif provider == 'ollama':
                response = requests.post(
                    f"{self.ollama_host}/api/embeddings",
                    json={
                        "model": model_name,
                        "prompt": test_text
                    }
                )
                response.raise_for_status()
                embedding = response.json().get('embedding', [])
                if embedding:
                    return len(embedding)
                
            elif provider == 'embedding':
                # Clean up model name if needed
                if model_name and ':' in model_name:
                    model_name = model_name.split(':')[0]
                
                # Определяем размерность через ОБЩИЙ кэш модели (не грузим
                # отдельную копию только ради подсчёта размеров).
                temp_model = get_sentence_transformer(model_name)
                embedding = temp_model.encode(test_text)
                return len(embedding)
                
        except Exception as e:
            # Just capture the error for raising at a higher level
            import traceback
            error_details = traceback.format_exc()
            raise ValueError(f"Error testing {provider}/{model_name}: {e}\n{error_details}")
            
        # If we reach here with no result, signal error
        raise ValueError(f"No embedding returned from {provider}/{model_name} test request")
    
    def _ensure_provider_initialized(self, provider):
        """Ensure provider-specific attributes are initialized"""
        # Initialize OpenAI attributes if needed
        if provider == 'openai' and not hasattr(self, 'openai_api_key'):
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
            
        # Initialize Ollama attributes if needed
        elif provider == 'ollama' and not hasattr(self, 'ollama_host'):
            self.ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
            
            # Use the model from JSON config file instead of environment variable
            if hasattr(self, 'active_config') and self.active_config["models"]["ollama_embedding"]:
                self.ollama_model = self.active_config["models"]["ollama_embedding"]
            else:
                self.ollama_model = 'nomic-embed-text'
    
    def create_collection(self, collection_name, vector_size=None, provider=None, model=None):
        """
        Create a collection in Qdrant
        
        Args:
            collection_name (str): Name of the collection
            vector_size (int, optional): Size of the embedding vectors
            provider (str, optional): The provider to use for embedding
            model (str, optional): The specific model to use for embedding
            
        Returns:
            bool: True if collection was created successfully
        """
        try:
            # Determine vector size based on arguments
            if vector_size is not None:
                # Use explicitly provided size
                pass
            elif provider and model:
                # Determine size from provider and model
                vector_size = self.get_vector_size(provider, model)
            else:
                # Fallback to provider default
                vector_size = self.vector_size
                
            # Only log this in debug contexts, not during normal operation
                
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
            return True
        except Exception as e:
            print(f"Error creating collection: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def list_collections(self):
        """
        List all collections in Qdrant
        
        Returns:
            list: List of collection names
        """
        try:
            collections = self.client.get_collections().collections
            return [collection.name for collection in collections]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []
    
    def collection_exists(self, collection_name):
        """
        Check if a collection exists
        
        Args:
            collection_name (str): Name of the collection to check
            
        Returns:
            bool: True if collection exists
        """
        return collection_name in self.list_collections()
    
    def delete_collection(self, collection_name):
        """
        Delete a collection from Qdrant
        
        Args:
            collection_name (str): Name of the collection to delete
            
        Returns:
            bool: True if collection was deleted successfully
        """
        try:
            if self.collection_exists(collection_name):
                self.client.delete_collection(collection_name=collection_name)
                return True
            else:
                print(f"Collection {collection_name} not found in Qdrant")
                return False
        except Exception as e:
            print(f"Error deleting collection: {e}")
            return False
            
    def delete_document_points(self, collection_name, document_id):
        """
        Delete all vector points related to a specific document from Qdrant
        
        Args:
            collection_name (str): Name of the collection
            document_id (str): ID of the document to delete points for
            
        Returns:
            bool: True if points were deleted successfully
        """
        try:
            if not self.collection_exists(collection_name):
                print(f"Collection {collection_name} not found in Qdrant")
                return False
                
            # Use search with filter instead of scroll (which has different API in v1.7.0)
            # First create a filter to find all points with this document_id
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id)
                    )
                ]
            )
            
            # Search for points in batches
            batch_size = 100
            deleted_count = 0
            
            while True:
                # Find points matching the document_id
                results = self.client.search(
                    collection_name=collection_name,
                    query_vector=[0.0] * self.vector_size,  # Dummy vector, results will be based on filter
                    query_filter=search_filter,
                    limit=batch_size,
                    with_payload=True,
                    with_vectors=False,
                    score_threshold=0.0  # Accept any score since we're using filter
                )
                
                # If no more results, we're done
                if not results:
                    break
                
                # Get IDs from results
                point_ids = [point.id for point in results]
                
                if point_ids:
                    # Delete the found points
                    self.client.delete(
                        collection_name=collection_name,
                        points_selector=models.PointIdsList(
                            points=point_ids
                        )
                    )
                    deleted_count += len(point_ids)
                    print(f"Deleted batch of {len(point_ids)} points for document {document_id}")
                    
                # If we got fewer than the batch size, we're done
                if len(results) < batch_size:
                    break
            
            print(f"Total deleted: {deleted_count} points for document {document_id}")
            # Коллекция изменилась — снимок в памяти устарел. Сбрасываем кэш
            # здесь, чтобы удаление документа сразу влияло на поиск.
            try:
                from app.services.document_service import invalidate_snapshot
                invalidate_snapshot(collection_name)
            except Exception:  # noqa: BLE001
                pass
            return True
            
        except Exception as e:
            print(f"Error deleting document points: {e}")
            import traceback
            print(traceback.format_exc())
            return False
    
    def insert_document(self, collection_name, document_id, text, metadata=None, provider=None, model=None):
        """
        Insert a document into Qdrant
        
        Args:
            collection_name (str): Name of the collection
            document_id (str): Unique ID for the document (must be UUID string or integer)
            text (str): Text content to be embedded
            metadata (dict, optional): Additional metadata
            provider (str, optional): The provider to use for embedding
            model (str, optional): The specific model to use for embedding
            
        Returns:
            bool: True if document was inserted successfully
        """
        try:
            # Generate a new UUID as string
            import uuid
            # Always use a string representation of UUID
            point_id = str(uuid.uuid4())
            
            # Store original ID in metadata if provided
            if document_id != point_id:
                if metadata is None:
                    metadata = {}
                metadata['original_id'] = document_id
            
            # Clean up model name if needed (especially for sentence transformers)
            if provider == 'embedding' and model and ':' in model:
                model = model.split(':')[0]
                
            # Ensure we have required attributes for the chosen provider
            self._ensure_provider_initialized(provider)
                
            # Generate embedding using the specified provider and model
            embedding = self.get_embedding(text, provider=provider, model=model)
            
            # Store the provider and model used in metadata
            if provider and model:
                if metadata is None:
                    metadata = {}
                metadata['embedding_provider'] = provider
                metadata['embedding_model'] = model
            
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=metadata or {}
                    )
                ]
            )
            return True
        except Exception as e:
            print(f"Error inserting document: {e}")
            import traceback
            print(f"Raw response content:\n{traceback.format_exc()}")
            return False
    
    def insert_documents_batch(self, collection_name, items, provider=None, model=None,
                               wait=False):
        """
        Пакетная вставка: пачка текстов → один вызов эмбеддера и один upsert.

        Зачем: ``insert_document`` на каждый чанк делает отдельный HTTP-запрос
        к Qdrant. При заливке больших документов (сотни чанков) это очень
        медленно и упирается в таймауты. Здесь мы эмбеддим весь список разом
        (``model.encode(list)``) и отправляем один upsert, а ``wait=False``
        избавляет от ожидания синхронной индексации на каждом шаге.

        Args:
            items: список словарей ``{"text": ..., "metadata": ...}``

        Returns:
            int: сколько точек вставлено (0 при ошибке)
        """
        if not items:
            return 0
        try:
            import uuid as _uuid

            provider = provider or self.llm_provider
            self._ensure_provider_initialized(provider)

            texts = [it["text"] for it in items]

            # Эмбеддинги пачкой — за один проход по модели это быстрее, чем
            # N отдельных encode().
            if provider == "embedding":
                if model and ":" in model:
                    model = model.split(":")[0]
                embedding_model = model or getattr(self, "embedding_model_name",
                                                   "all-MiniLM-L6-v2")
                st = get_sentence_transformer(embedding_model)
                vectors = st.encode(texts, batch_size=32).tolist()
            else:
                # Прочие провайдеры (ollama/openai) — по одному, но это редкий путь.
                vectors = [self.get_embedding(t, provider=provider, model=model)
                           for t in texts]

            points = []
            for it, vec in zip(items, vectors):
                meta = dict(it.get("metadata") or {})
                meta["embedding_provider"] = provider
                meta["embedding_model"] = model
                points.append(models.PointStruct(
                    id=str(_uuid.uuid4()), vector=vec, payload=meta,
                ))

            self.client.upsert(collection_name=collection_name, points=points,
                               wait=wait)
            return len(points)
        except Exception as e:  # noqa: BLE001
            print(f"Error inserting batch: {e}")
            return 0

    def rest_base_url(self, collection_name=None):
        """
        Рабочий адрес Qdrant для ПРЯМЫХ REST-запросов (не через клиент).

        ПОЧЕМУ НЕ localhost: в этом окружении VPN перехватывает loopback и
        обрывает HTTP-ответы больше ~16 КБ. Полная выгрузка коллекции — это
        мегабайты, поэтому через localhost она просто виснет до таймаута.
        Сетевой адрес (docker bridge 172.17.0.1 или LAN) отдаёт всё мгновенно.

        ВАЖНО: проба нарочно БОЛЬШАЯ. Маленький GET проходит и на loopback, так
        что по нему нельзя отличить рабочий адрес от сломанного (в контейнере
        QDRANT_URL=http://localhost:6333 и он «отвечает» на любую мелочь).
        Поэтому проверяем именно выгрузку крупной страницы.

        Результат кэшируется в ``self._rest_base``.
        """
        cached = getattr(self, "_rest_base", None)
        if cached:
            return cached

        import socket

        cands = []
        # 1) docker bridge gateway — самый надёжный путь в этом окружении
        cands.append("http://172.17.0.1:6333")
        # 2) LAN-адреса хоста
        try:
            for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
                if not ip.startswith("127."):
                    cands.append(f"http://{ip}:6333")
        except Exception:  # noqa: BLE001
            pass
        # 3) то, что пришло из окружения (обычно localhost — проверяем в конце)
        if os.getenv("QDRANT_URL"):
            cands.append(os.getenv("QDRANT_URL"))
        cands.append(f"http://{self.host}:{self.port}")
        cands.append("http://localhost:6333")

        # Проба: крупная страница (≈ сотни КБ). На loopback она отваливается
        # по таймауту — именно это нам и нужно отличить.
        probe_path = (
            f"/collections/{collection_name}/points/scroll"
            if collection_name else "/collections"
        )
        probe_body = (
            {"limit": 200, "with_payload": True, "with_vector": False}
            if collection_name else None
        )

        for url in cands:
            try:
                if probe_body is None:
                    r = requests.get(f"{url}{probe_path}", timeout=4)
                    ok = r.status_code == 200
                else:
                    r = requests.post(f"{url}{probe_path}", json=probe_body,
                                      timeout=6)
                    # 404 = коллекции тут нет, но канал рабочий.
                    ok = r.status_code in (200, 404)
                if ok:
                    self._rest_base = url
                    print(f"[Qdrant] REST-адрес: {url}")
                    return url
            except Exception:  # noqa: BLE001
                continue

        self._rest_base = cands[-1]
        print(f"[Qdrant] REST-адрес (под вопросом): {self._rest_base}")
        return self._rest_base

    def scroll_all(self, collection_name, max_points=5000, page=512,
                   with_vector=False):
        """
        Возвращает ВСЕ точки коллекции через REST (Можно вместе с векторами).

        Зачем: модель эмбеддингов англоязычная и на русских вопросах ранжирует
        плохо — нужный чанк может не попасть даже в расширенный топ векторного
        поиска. База вуза маленькая (сотни чанков), поэтому дешевле выгрузить
        её целиком и переранжировать уже по всем чанкам.

        Идём страницами (``next_page_offset``) — так ответы остаются небольшими
        и не упираются в лимиты прокси. Если ``with_vector=True``, векторные
        скоры считаются ЛОКАЛЬНО (см. ``document_service``): обычный поиск через
        клиент на localhost возвращает мегабайты payload и виснет по таймауту.
        """
        base = self.rest_base_url(collection_name)
        points = []
        offset = None
        try:
            while True:
                body = {
                    "limit": page,
                    "with_payload": True,
                    "with_vector": bool(with_vector),
                }
                if offset is not None:
                    body["offset"] = offset
                r = requests.post(
                    f"{base}/collections/{collection_name}/points/scroll",
                    json=body,
                    timeout=30,
                )
                if r.status_code != 200:
                    print(f"scroll {collection_name}: HTTP {r.status_code} "
                          f"{r.text[:200]}")
                    break
                result = (r.json() or {}).get("result") or {}
                for p in result.get("points") or []:
                    item = {
                        "id": str(p.get("id")),
                        "score": 0.0,
                        "payload": p.get("payload") or {},
                    }
                    if with_vector:
                        item["vector"] = p.get("vector")
                    points.append(item)
                offset = result.get("next_page_offset")
                if not offset or len(points) >= max_points:
                    break
            return points
        except Exception as e:
            print(f"Error scrolling collection {collection_name}: {e}")
            return points

    def search(self, collection_name, query, limit=5, provider=None, model=None):
        """
        Search for similar documents in Qdrant
        
        Args:
            collection_name (str): Name of the collection to search in
            query (str): Query text
            limit (int, optional): Maximum number of results to return
            provider (str, optional): The provider to use for embedding
            model (str, optional): The specific model to use for embedding
            
        Returns:
            list: List of search results with documents and similarity scores
        """
        try:
            # Clean up model name if needed (especially for sentence transformers)
            if provider == 'embedding' and model and ':' in model:
                model = model.split(':')[0]
                
            # Ensure we have required attributes for the chosen provider
            self._ensure_provider_initialized(provider)
            
            # Generate embedding using the specified provider and model
            query_embedding = self.get_embedding(query, provider=provider, model=model)
            
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'id': result.id,
                    'score': result.score,
                    'payload': result.payload
                })
            
            return formatted_results
        except Exception as e:
            print(f"Error searching: {e}")
            import traceback
            print(f"Search error details:\n{traceback.format_exc()}")
            return []