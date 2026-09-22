import os
from anthropic import Anthropic
from app.services.llm_service import LLMService
from app.services.model_service import ModelService

class ClaudeService(LLMService):
    """Service for interacting with Anthropic's Claude API"""
    
    def __init__(self):
        """Initialize the Anthropic client"""
        # Get API key from environment variable
        api_key = os.getenv('CLAUDE_API_KEY')
        if not api_key:
            raise ValueError("CLAUDE_API_KEY not found in environment variables")
        self.client = Anthropic(api_key=api_key)
        
        # Load configuration from the model service
        self.model_service = ModelService()
        self.config = self.model_service.config
        
        # Get model from JSON config instead of environment variable
        if self.config["active"]["models"]["claude_llm"]:
            self.model = self.config["active"]["models"]["claude_llm"]
        else:
            # Fallback to a default model
            self.model = 'claude-3-sonnet-20240229'
    
    def _generate_completion(self, system_message, user_message, max_tokens):
        """
        Generate a completion using Claude API
        
        Args:
            system_message (str): The system message
            user_message (str): The user message
            max_tokens (int): The maximum number of tokens to generate
            
        Returns:
            str: The generated response text
        """
        response = self.client.messages.create(
            model=self.model,
            system=system_message,
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        return response.content[0].text
            
    def _generate_chat_completion(self, messages, max_tokens):
        """
        Generate a chat completion using Claude API
        
        Args:
            messages (list): The formatted message list
            max_tokens (int): The maximum number of tokens to generate
            
        Returns:
            str: The generated response text
        """
        # Extract the system message if present as Claude handles it separately
        system_message = "You are a helpful assistant."
        claude_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                # Claude expects role to be either 'user' or 'assistant'
                claude_messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
        
        response = self.client.messages.create(
            model=self.model,
            system=system_message,
            max_tokens=max_tokens,
            messages=claude_messages
        )
        return response.content[0].text