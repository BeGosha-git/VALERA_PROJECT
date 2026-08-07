import os
from abc import ABC, abstractmethod

class LLMService(ABC):
    """Base class for LLM service providers"""
    
    # These standard messages are now managed in the PromptTemplateService
    # but kept here for backward compatibility
    RAG_SYSTEM_MESSAGE = (
        "You are a helpful and knowledgeable assistant. "
        "When answering the user's question, always review the context provided below. "
        "If relevant information is found in the context, prioritize and incorporate it into your response, "
        "citing references to the source documents where applicable with all relevant document_title. "
        "If the context does not contain sufficient or relevant details, don't answer and state there is no relevant context. "
        "Provide clear, concise, and accurate answers."
    )
    
    CHAT_CONTEXT_ENHANCEMENT = (
        "\nWhen answering the user's question, always review the context provided below. "
        "If relevant information is found in the context, prioritize and incorporate it into your response, "
        "citing references to the source documents where applicable. "
        "If the context does not contain sufficient or relevant details, answer based on your knowledge "
        "but acknowledge the limitations."
    )
    
    def format_user_message_with_context(self, prompt, context):
        """Format the user message with context for RAG"""
        return f"Context:\n{context}\n\nQuestion: {prompt}"
    
    def get_provider_name(self):
        """Get the provider name for this LLM service"""
        # This will be implemented in child classes or use class name as fallback
        return self.__class__.__name__.replace('Service', '')
    
    def get_model_name(self):
        """Get the model name being used by this LLM service"""
        # This will be implemented in child classes or use a default
        return getattr(self, 'model', 'Unknown Model')
    
    def generate_response(self, prompt, context=None, max_tokens=1000):
        """
        Generate a response from the LLM - base implementation with common logic
        
        This method is implemented in the base class but uses the provider-specific
        implementation to handle the actual API call.
        
        Args:
            prompt (str): The user's query
            context (str, optional): Retrieved context to augment the prompt
            max_tokens (int, optional): Maximum number of tokens to generate
            
        Returns:
            str: The generated response
        """
        system_message = "You are a helpful assistant."
        user_message = prompt
        
        if context:
            system_message = self.RAG_SYSTEM_MESSAGE
            user_message = self.format_user_message_with_context(prompt, context)
        
        try:
            return self._generate_completion(system_message, user_message, max_tokens)
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_chat_response(self, messages, context=None, max_tokens=1000):
        """
        Generate a response from the LLM using chat format - base implementation
        
        Args:
            messages (list): List of message dictionaries with 'role' and 'content'
            context (str, optional): Retrieved context to augment the prompt
            max_tokens (int, optional): Maximum number of tokens to generate
            
        Returns:
            str: The generated response
        """
        # Make a copy of the messages to avoid modifying the original
        formatted_messages = list(messages)
        
        # Handle context if provided
        if context:
            self._enhance_messages_with_context(formatted_messages, context)
        
        try:
            return self._generate_chat_completion(formatted_messages, max_tokens)
        except Exception as e:
            print(f"Error generating chat response: {e}")
            return f"Error generating response: {str(e)}"
    
    def _enhance_messages_with_context(self, messages, context):
        """
        Enhance messages with context - common implementations
        
        This is a helper method used by generate_chat_response to add
        context to the message list.
        
        Args:
            messages (list): The message list to modify
            context (str): The context to add
        """
        # Find the system message if it exists
        system_message_index = None
        for i, msg in enumerate(messages):
            if msg['role'] == 'system':
                system_message_index = i
                break
        
        if system_message_index is not None:
            # Enhance the existing system message
            messages[system_message_index]['content'] += self.CHAT_CONTEXT_ENHANCEMENT
        
        # Find the most recent user message
        user_message_index = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]['role'] == 'user':
                user_message_index = i
                break
        
        if user_message_index is not None:
            # Add context to the most recent user message
            user_content = messages[user_message_index]['content']
            messages[user_message_index]['content'] = f"Context:\n{context}\n\nQuestion: {user_content}"
    
    @abstractmethod
    def _generate_completion(self, system_message, user_message, max_tokens):
        """
        Provider-specific method to generate a completion
        
        Args:
            system_message (str): The system message
            user_message (str): The user message
            max_tokens (int): The maximum number of tokens to generate
            
        Returns:
            str: The generated response text
        """
        pass
        
    @abstractmethod
    def _generate_chat_completion(self, messages, max_tokens):
        """
        Provider-specific method to generate a chat completion
        
        Args:
            messages (list): The formatted message list
            max_tokens (int): The maximum number of tokens to generate
            
        Returns:
            str: The generated response text
        """
        pass

class LLMFactory:
    """Factory for creating LLM service instances"""
    
    @staticmethod
    def create_llm_service():
        """
        Create an LLM service based on the configured provider
        
        Returns:
            LLMService: An instance of the configured LLM service
        """
        # Load from model service configuration
        from app.services.model_service import ModelService
        model_service = ModelService()
        provider = model_service.config["active"]["llm_provider"].lower()
        
        if not provider:
            # Fallback to default if no provider configured
            provider = "openai"
        
        if provider == 'openai':
            from app.services.openai_service import OpenAIService
            return OpenAIService()
        elif provider == 'ollama':
            from app.services.ollama_service import OllamaService
            return OllamaService()
        elif provider == 'claude':
            from app.services.claude_service import ClaudeService
            return ClaudeService()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")