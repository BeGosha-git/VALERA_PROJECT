import os
import json
import requests
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from app.services.model_service import ModelService

class QdrantService:
    """Service for interacting with Qdrant vector database"""
    
    def __init__(self):
        """Initialize Qdrant client and embedding model"""
        self.host = os.getenv('QDRANT_HOST', 'localhost')
        self.port = int(os.getenv('QDRANT_PORT', 6333))
        self.client = QdrantClient(host=self.host, port=self.port)
        
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
            self.embedding_model = SentenceTransformer(model_name)
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
                    self.embedding_model = SentenceTransformer(fallback_model)
                    self.embedding_model_name = fallback_model
                return self.embedding_model.encode(text).tolist()
                
        # Ollama embeddings
        elif provider == 'ollama':
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
                    }
                )
                response.raise_for_status()
                return response.json().get('embedding', [])
            except Exception as e:
                print(f"Error with Ollama embedding: {e}")
                # Fallback to sentence-transformers
                fallback_model = 'all-MiniLM-L6-v2'
                if not hasattr(self, 'embedding_model') or self.embedding_model_name != fallback_model:
                    self.embedding_model = SentenceTransformer(fallback_model)
                    self.embedding_model_name = fallback_model
                return self.embedding_model.encode(text).tolist()
        
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
                    self.embedding_model = SentenceTransformer(embedding_model)
                    self.embedding_model_name = embedding_model
                return self.embedding_model.encode(text).tolist()
            except Exception as e:
                print(f"Error with model {embedding_model}: {e}")
                # Fall back to known working model
                fallback_model = 'all-MiniLM-L6-v2'
                if not hasattr(self, 'embedding_model') or self.embedding_model_name != fallback_model:
                    self.embedding_model = SentenceTransformer(fallback_model)
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
                
                # Create a temporary model instance just for dimension check
                temp_model = SentenceTransformer(model_name)
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