import os
import json
import logging
from pathlib import Path
import requests
from openai import OpenAI
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelService:
    """Service for managing LLM and embedding models"""
    
    # Class variable for singleton instance
    _instance = None
    
    # Initial configuration structure for new installations
    INITIAL_CONFIG = {
        "providers": {
            "embedding": {
                "models": []
            }
        },
        "active": {
            "llm_provider": "",
            "models": {
                "openai_llm": "",
                "openai_embedding": "",
                "claude_llm": "",
                "ollama_llm": "",
                "ollama_embedding": "",
                "embedding": ""
            }
        }
    }
    
    def __new__(cls):
        """Implement singleton pattern"""
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize model service and load available models (only once)"""
        # Skip initialization if already done
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.config_path = Path('data/models/config.json')
        self.config_dir = self.config_path.parent
        
        # Create config directory if it doesn't exist
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize configuration
        self.config = self._load_config()
        
        # Update with available models from providers
        self._update_available_models()
        
        # Save updated config
        self._save_config()
        
        # Mark as initialized
        self._initialized = True
    
    def _load_config(self):
        """Load configuration from file or initialize with empty structure"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    
                    # Initialize provider entries if environment variables are set
                    # but only if they don't already exist in the config
                    if os.getenv('OPENAI_API_KEY') and 'openai' not in config['providers']:
                        config['providers']['openai'] = {
                            "models": [],
                            "config": {"api_key_env": "OPENAI_API_KEY"}
                        }
                        
                    if os.getenv('CLAUDE_API_KEY') and 'claude' not in config['providers']:
                        config['providers']['claude'] = {
                            "models": [],
                            "config": {"api_key_env": "CLAUDE_API_KEY"}
                        }
                        
                    if os.getenv('OLLAMA_HOST') and 'ollama' not in config['providers']:
                        config['providers']['ollama'] = {
                            "models": [],
                            "config": {"host_env": "OLLAMA_HOST"}
                        }
                        
                    return config
            except Exception as e:
                logger.error(f"Error loading model configuration: {e}")
                return self._create_initial_config()
        else:
            return self._create_initial_config()
            
    def _create_initial_config(self):
        """Create initial configuration based on available environment variables"""
        config = {
            "providers": {
                "embedding": {
                    "models": []
                }
            },
            "active": {
                "llm_provider": "",
                "models": {
                    "openai_llm": "",
                    "openai_embedding": "",
                    "claude_llm": "",
                    "ollama_llm": "",
                    "ollama_embedding": "",
                    "embedding": ""
                }
            }
        }
        
        # Only add providers if their API keys are set
        if os.getenv('OPENAI_API_KEY'):
            config['providers']['openai'] = {
                "models": [],
                "config": {"api_key_env": "OPENAI_API_KEY"}
            }
            # Set as default provider if available
            config['active']['llm_provider'] = 'openai'
            
        if os.getenv('CLAUDE_API_KEY'):
            config['providers']['claude'] = {
                "models": [],
                "config": {"api_key_env": "CLAUDE_API_KEY"}
            }
            # Set as default provider if OpenAI not available
            if not config['active']['llm_provider']:
                config['active']['llm_provider'] = 'claude'
                
        if os.getenv('OLLAMA_HOST'):
            config['providers']['ollama'] = {
                "models": [],
                "config": {"host_env": "OLLAMA_HOST"}
            }
            # Set as default provider if no others are available
            if not config['active']['llm_provider']:
                config['active']['llm_provider'] = 'ollama'
                
        return config
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving model configuration: {e}")
    
    def _update_available_models(self):
        """Update available models from providers"""
        # Only update providers that have their environment variables set
        # This allows the system to adapt when keys are added or removed
        
        # First, check if we need to remove any providers whose env vars are no longer set
        self._cleanup_unavailable_providers()
        
        # Check for OpenAI models if API key is available
        if os.getenv('OPENAI_API_KEY') and 'openai' in self.config["providers"]:
            try:
                self._update_openai_models()
            except Exception as e:
                logger.warning(f"Error updating OpenAI models: {e}")
        
        # Check for Claude models if API key is available
        if os.getenv('CLAUDE_API_KEY') and 'claude' in self.config["providers"]:
            try:
                self._update_claude_models()
            except Exception as e:
                logger.warning(f"Error updating Claude models: {e}")
        
        # Check for Ollama models if host is available
        if os.getenv('OLLAMA_HOST') and 'ollama' in self.config["providers"]:
            try:
                self._update_ollama_models()
            except Exception as e:
                logger.warning(f"Error updating Ollama models: {e}")
                
        # Update embedding models (sentence transformers) - always available
        try:
            self._update_embedding_models()
        except Exception as e:
            logger.warning(f"Error updating embedding models: {e}")
            
    def _cleanup_unavailable_providers(self):
        """Remove providers whose environment variables are no longer set"""
        providers_to_remove = []
        
        # Check if OpenAI API key is no longer available
        if 'openai' in self.config["providers"] and not os.getenv('OPENAI_API_KEY'):
            providers_to_remove.append('openai')
            
        # Check if Claude API key is no longer available
        if 'claude' in self.config["providers"] and not os.getenv('CLAUDE_API_KEY'):
            providers_to_remove.append('claude')
            
        # Check if Ollama host is no longer available
        if 'ollama' in self.config["providers"] and not os.getenv('OLLAMA_HOST'):
            providers_to_remove.append('ollama')
            
        # Remove unavailable providers
        for provider in providers_to_remove:
            logger.info(f"Removing provider {provider} as its environment variable is not set")
            del self.config["providers"][provider]
            
        # If the active provider was removed, set a new one
        if providers_to_remove and self.config["active"]["llm_provider"] in providers_to_remove:
            # Find a new provider
            available_providers = []
            if 'openai' in self.config["providers"]:
                available_providers.append('openai')
            if 'claude' in self.config["providers"]:
                available_providers.append('claude')
            if 'ollama' in self.config["providers"]:
                available_providers.append('ollama')
                
            if available_providers:
                self.config["active"]["llm_provider"] = available_providers[0]
            else:
                self.config["active"]["llm_provider"] = ""
    
    def _update_openai_models(self):
        """Update available OpenAI models"""
        try:
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            models = client.models.list()
            
            # Filter for chat completion models
            chat_models = [
                {"id": model.id, "name": model.id, "type": "llm"}
                for model in models
                if model.id.startswith(("gpt-"))
            ]
            
            # Filter for embedding models
            embedding_models = [
                {"id": model.id, "name": model.id, "type": "embedding"}
                for model in models
                if "embedding" in model.id
            ]
            
            # Preserve defaults from existing models
            for model in chat_models + embedding_models:
                for existing_model in self.config["providers"]["openai"]["models"]:
                    if existing_model["id"] == model["id"]:
                        model["default"] = existing_model.get("default", False)
                        break
            
            # Update models
            self.config["providers"]["openai"]["models"] = chat_models + embedding_models
            
        except Exception as e:
            logger.warning(f"Could not fetch OpenAI models: {e}")
    
    def _update_claude_models(self):
        """Update available Claude models from Anthropic API"""
        try:
            client = Anthropic(api_key=os.getenv('CLAUDE_API_KEY'))
            
            # Get current models to maintain defaults
            current_models = self.config["providers"]["claude"]["models"]
            
            # Fetch models from the API
            response = client.models.list(limit=20)
            
            # Process models from API
            new_models = []
            for model in response.data:
                model_data = {
                    "id": model.id,
                    "name": model.display_name if hasattr(model, 'display_name') and model.display_name else model.id,
                    "type": "llm"
                }
                
                # Preserve default setting if model already exists
                for existing_model in current_models:
                    if existing_model["id"] == model.id:
                        model_data["default"] = existing_model.get("default", False)
                        break
                
                new_models.append(model_data)
            
            # Set the first model as default if no default exists
            if not any(model.get("default") for model in new_models) and new_models:
                new_models[0]["default"] = True
                
            # Update the config
            self.config["providers"]["claude"]["models"] = new_models
            
        except Exception as e:
            logger.error(f"Error updating Claude models: {e}")
            raise
    
    def _update_ollama_models(self):
        """Update available Ollama models"""
        try:
            ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
            response = requests.get(f"{ollama_host}/api/tags")
            
            if response.status_code == 200:
                data = response.json()
                models = []
                
                for model in data.get('models', []):
                    model_name = model.get('name')
                    if model_name:
                        # Check if this is already in our config to preserve defaults
                        is_default = False
                        for existing_model in self.config["providers"]["ollama"]["models"]:
                            if existing_model["id"] == model_name:
                                is_default = existing_model.get("default", False)
                                break
                                
                        models.append({
                            "id": model_name, 
                            "name": model_name, 
                            "type": "llm" if not ("embed" in model_name.lower() or "bge" in model_name.lower()) else "embedding",
                            "default": is_default
                        })
                
                # Check if we found embedding models, if not add known embedding models
                embedding_models = [m for m in models if m["type"] == "embedding"]
                if not embedding_models:
                    # Add known Ollama embedding models
                    known_embedding_models = [
                        {"id": "nomic-embed-text", "name": "Nomic Embed Text", "type": "embedding"},
                        {"id": "all-minilm", "name": "All MiniLM", "type": "embedding"}
                    ]
                    for model in known_embedding_models:
                        # Check if this model already exists in our config to preserve defaults
                        for existing_model in self.config["providers"]["ollama"]["models"]:
                            if existing_model["id"] == model["id"]:
                                model["default"] = existing_model.get("default", False)
                                break
                        models.append(model)
                
                # Set the first embedding model as default if no default exists
                embedding_models = [m for m in models if m["type"] == "embedding"]
                if embedding_models and not any(m.get("default") for m in embedding_models):
                    embedding_models[0]["default"] = True
                    
                # Set the first LLM model as default if no default exists
                llm_models = [m for m in models if m["type"] == "llm"]
                if llm_models and not any(m.get("default") for m in llm_models):
                    llm_models[0]["default"] = True
                
                self.config["providers"]["ollama"]["models"] = models
        except Exception as e:
            logger.warning(f"Could not fetch Ollama models: {e}")
    
    def get_available_providers(self):
        """Get list of available model providers"""
        providers = []
        
        # OpenAI
        if os.getenv('OPENAI_API_KEY'):
            providers.append({
                "id": "openai",
                "name": "OpenAI",
                "available": True
            })
        else:
            providers.append({
                "id": "openai",
                "name": "OpenAI",
                "available": False,
                "reason": "API key not configured. Set OPENAI_API_KEY in environment."
            })
        
        # Claude
        if os.getenv('CLAUDE_API_KEY'):
            providers.append({
                "id": "claude",
                "name": "Anthropic Claude",
                "available": True
            })
        else:
            providers.append({
                "id": "claude",
                "name": "Anthropic Claude",
                "available": False,
                "reason": "API key not configured. Set CLAUDE_API_KEY in environment."
            })
        
        # Ollama
        ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        try:
            response = requests.get(f"{ollama_host}/api/tags", timeout=1)
            if response.status_code == 200:
                providers.append({
                    "id": "ollama",
                    "name": "Ollama",
                    "available": True
                })
            else:
                providers.append({
                    "id": "ollama",
                    "name": "Ollama",
                    "available": False,
                    "reason": f"Ollama server returned status code {response.status_code}"
                })
        except Exception:
            providers.append({
                "id": "ollama",
                "name": "Ollama",
                "available": False,
                "reason": f"Could not connect to Ollama server at {ollama_host}"
            })
        
        # Sentence Transformers (always available for embeddings)
        providers.append({
            "id": "embedding",
            "name": "Sentence Transformers",
            "available": True,
            "note": "Used for embeddings when provider embedding is not available"
        })
        
        return providers
    
    def get_available_models(self, provider=None, model_type=None):
        """
        Get available models, optionally filtered by provider and type
        
        Args:
            provider (str, optional): Filter by provider (openai, claude, ollama, embedding)
            model_type (str, optional): Filter by model type (llm, embedding)
            
        Returns:
            list: List of available models
        """
        models = []
        
        # Get models from all providers or the specified one
        providers = [provider] if provider else self.config["providers"].keys()
        
        for p in providers:
            if p in self.config["providers"]:
                provider_models = self.config["providers"][p]["models"]
                
                # Filter by type if specified
                if model_type:
                    provider_models = [m for m in provider_models if m["type"] == model_type]
                
                # Add provider info to each model
                for model in provider_models:
                    model_copy = model.copy()
                    model_copy["provider"] = p
                    models.append(model_copy)
        
        return models
    
    def get_active_models(self):
        """Get currently active models"""
        return self.config["active"]
    
    def _update_embedding_models(self):
        """Update available embedding models (sentence transformers)"""
        try:
            # Get current embedding models
            current_models = self.config["providers"]["embedding"]["models"]
            current_model_ids = [model["id"] for model in current_models]
            
            # List of known embedding models
            known_models = [
                {"id": "all-MiniLM-L6-v2", "name": "MiniLM L6", "type": "embedding"},
                {"id": "all-mpnet-base-v2", "name": "MPNet Base", "type": "embedding"},
                {"id": "paraphrase-multilingual-MiniLM-L12-v2", "name": "Multilingual MiniLM", "type": "embedding"},
                {"id": "multi-qa-MiniLM-L6-cos-v1", "name": "QA MiniLM", "type": "embedding"}
            ]
            
            # Add any new models not already in the list
            for model in known_models:
                if model["id"] not in current_model_ids:
                    current_models.append(model)
            
            # Set the first model as default if no default exists
            if not any(model.get("default") for model in current_models) and current_models:
                current_models[0]["default"] = True
                
            # Update the config
            self.config["providers"]["embedding"]["models"] = current_models
            
        except Exception as e:
            logger.warning(f"Error updating embedding models: {e}")
    
    def set_active_model(self, provider, model_id, model_type):
        """
        Set active model for a specific provider and type
        
        Args:
            provider (str): The provider (openai, claude, ollama, embedding)
            model_id (str): The model ID
            model_type (str): The model type (llm, embedding)
            
        Returns:
            bool: True if model was set successfully
        """
        if provider not in self.config["providers"]:
            return False
            
        # Check if model exists
        model_exists = False
        for model in self.config["providers"][provider]["models"]:
            if model["id"] == model_id and model["type"] == model_type:
                model_exists = True
                break
                
        if not model_exists:
            return False
        
        # Set active model
        if model_type == "llm":
            if provider == "openai":
                self.config["active"]["models"]["openai_llm"] = model_id
            elif provider == "claude":
                self.config["active"]["models"]["claude_llm"] = model_id
            elif provider == "ollama":
                self.config["active"]["models"]["ollama_llm"] = model_id
        elif model_type == "embedding":
            if provider == "openai":
                self.config["active"]["models"]["openai_embedding"] = model_id
            elif provider == "ollama":
                self.config["active"]["models"]["ollama_embedding"] = model_id
            elif provider == "embedding":
                self.config["active"]["models"]["embedding"] = model_id
        
        # Save the config
        self._save_config()
        
        # Update environment variables to reflect changes
        self._update_environment_variables()
        
        return True
    
    def set_active_provider(self, provider):
        """
        Set active LLM provider
        
        Args:
            provider (str): The provider (openai, claude, ollama)
            
        Returns:
            bool: True if provider was set successfully
        """
        if provider not in ["openai", "claude", "ollama"]:
            return False
        
        # Update active provider
        self.config["active"]["llm_provider"] = provider
        
        # Save the config
        self._save_config()
        
        # Update environment variables
        self._update_environment_variables()
        
        return True
    
    def _update_environment_variables(self):
        """
        This method previously updated environment variables with the configuration.
        
        Now it's a no-op since we're using direct JSON config access instead of environment variables.
        Kept for backward compatibility.
        """
        # We no longer set environment variables here
        # All services should directly access the JSON config instead
        pass
        
    def refresh_models(self, force=False):
        """
        Explicitly refresh available models from providers
        
        Args:
            force (bool): Force refresh even if already initialized
            
        Returns:
            bool: True if refresh was performed
        """
        # Skip if not forcing and already initialized
        if not force and hasattr(self, '_initialized') and self._initialized:
            return False
            
        # Update with available models from providers
        self._update_available_models()
        
        # Save updated config
        self._save_config()
        
        return True