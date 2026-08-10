import os
from datetime import datetime
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

    @staticmethod
    def _now_context() -> str:
        """Возвращает текущую дату/время для подстановки в промпт."""
        from zoneinfo import ZoneInfo
        try:
            now = datetime.now(ZoneInfo("Europe/Moscow"))
        except Exception:
            now = datetime.now()
        weekdays = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
        months = ["января", "февраля", "марта", "апреля", "мая", "июня",
                  "июля", "августа", "сентября", "октября", "ноября", "декабря"]
        wd = weekdays[now.weekday()]
        return (f"Сегодня {wd}, {now.day} {months[now.month - 1]} {now.year} года. "
                f"Сейчас {now.hour:02d}:{now.minute:02d} по московскому времени. "
                f"Используй эту дату, когда спрашивают о сегодняшнем дне, числе или времени.")
    
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
        # Добавляем текущую дату, чтобы модель могла отвечать на вопросы о «сегодня»
        system_message = f"{system_message}\n\n{self._now_context()}"
        # Format the prompt for Ollama, which often works better with explicit labels
        full_prompt = f"System: {system_message}\n\nUser: {user_message}\n\nAssistant:"
        
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "keep_alive": "30m",  # держим модель в памяти между вызовами
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
        
        # Добавляем текущую дату, чтобы модель могла отвечать на вопросы о «сегодня»
        system_content = f"{system_content}\n\n{self._now_context()}\n\n"
        
        # Add a strong instruction so the model prioritizes the FRESHEST messages.
        # This fixes "stale context wins": the newest user question and the last
        # assistant reply should dominate, older messages are just background.
        system_content = (
            f"{system_content}\n"
            "Conversation dynamics: The messages at the END of the conversation are "
            "the most important. The very last user message is your current question "
            "and must be answered first. The most recent assistant replies and the "
            "last few turns carry more weight than old ones. Use older messages only "
            "as background context; do not let them override the latest topic."
        )
        
        # Format the conversation as a text prompt since many Ollama models
        # handle this format better than structured chat messages.
        # Separate older history from the latest exchange so the model can
        # clearly see what is fresh.
        non_system = [m for m in messages if m['role'] != 'system']
        
        # Keep at most the last N messages as "recent context" (excluding current question)
        RECENT_LIMIT = 6
        history = non_system[:-1] if len(non_system) > 1 else []
        last = non_system[-1] if non_system else None
        
        conversation_prompt = f"System: {system_content}\n\n"
        
        # Older messages (background only)
        if len(history) > RECENT_LIMIT:
            conversation_prompt += "[Earlier conversation omitted — only the recent messages matter]\n\n"
            history = history[-RECENT_LIMIT:]
        
        if history:
            conversation_prompt += "[Conversation history (background):]\n"
            for msg in history:
                role_name = "User" if msg['role'] == 'user' else "Assistant"
                conversation_prompt += f"{role_name}: {msg['content']}\n\n"
            conversation_prompt += "[End of history]\n\n"
        
        # The current question — emphasized as the priority
        if last is not None:
            role_name = "User" if last['role'] == 'user' else "Assistant"
            conversation_prompt += f"[CURRENT {role_name.upper()} — answer this now]:\n{last['content']}\n\n"
        else:
            conversation_prompt += "[CURRENT USER]:\n\n"
        
        conversation_prompt += "Assistant:"
        
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "model": self.model,
                "prompt": conversation_prompt,
                "stream": False,
                "keep_alive": "30m",  # держим модель в памяти между вызовами
                "options": {
                    "num_predict": max_tokens
                }
            }
        )
        response.raise_for_status()
        return response.json().get('response', 'No response received')