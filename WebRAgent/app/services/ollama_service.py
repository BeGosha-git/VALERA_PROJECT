import os
import requests
from app.services.llm_service import LLMService
from app.services.model_service import ModelService

class OllamaService(LLMService):
    """Service for interacting with Ollama API"""
    
    def __init__(self):
        """Initialize the Ollama service"""
        # Load configuration from the model service
        self.model_service = ModelService()
        self.config = self.model_service.config
        
        # Get Ollama host from environment variable
        self.ollama_host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
        self.base_url = f"{self.ollama_host}/api"
        
        # Get model from JSON config instead of environment variable
        if self.config["active"]["models"]["ollama_llm"]:
            self.model = self.config["active"]["models"]["ollama_llm"]
        else:
            # Fallback to a default model
            self.model = 'llama2'
    
    def _generate_completion(self, system_message, user_message, max_tokens):
        """
        Generate a completion using Ollama API
        
        Args:
            system_message (str): The system message
            user_message (str): The user message
            max_tokens (int): The maximum number of tokens to generate
            
        Returns:
            str: The generated response text
        """
        # Format the prompt for Ollama, which often works better with explicit labels
        full_prompt = f"System: {system_message}\n\nUser: {user_message}\n\nAssistant:"
        
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens
                }
            }
        )
        response.raise_for_status()
        return response.json().get('response', 'No response received')
    
    def _generate_chat_completion(self, messages, max_tokens):
        """
        Generate a chat completion using Ollama API
        
        Args:
            messages (list): The formatted message list
            max_tokens (int): The maximum number of tokens to generate
            
        Returns:
            str: The generated response text
        """
        # Extract system message if present
        system_content = "You are a helpful assistant."
        for message in messages:
            if message['role'] == 'system':
                system_content = message['content']
                break
        
        # Format the conversation as a text prompt since many Ollama models
        # handle this format better than structured chat messages
        conversation_prompt = f"System: {system_content}\n\n"
        
        for msg in messages:
            if msg['role'] != 'system':  # Skip system messages as we handled it above
                role_name = "User" if msg['role'] == 'user' else "Assistant"
                conversation_prompt += f"{role_name}: {msg['content']}\n\n"
        
        conversation_prompt += "Assistant:"
        
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "model": self.model,
                "prompt": conversation_prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens
                }
            }
        )
        response.raise_for_status()
        return response.json().get('response', 'No response received')