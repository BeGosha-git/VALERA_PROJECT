import os
from abc import ABC, abstractmethod

from app.services.valera_facts import facts_card

class LLMService(ABC):
    """Base class for LLM service providers"""
    
    # These standard messages are now managed in the PromptTemplateService
    # but kept here for backward compatibility
    #
    # Промпт для голосового робота: сначала факт, без мета-речи о «контексте» и
    # без цитирования источников (ответ читается вслух).
    RAG_SYSTEM_MESSAGE = (
        "Ты — Валера, робот-экскурсовод РТУ МИРЭА. Ты живой собеседник, а не "
        "справочная служба: говори просто, тепло и по-человечески, обычными "
        "словами, короткими фразами, с живой интонацией. "
        "Дай конкретику (имя, число, адрес) в первом предложении. "
        "Не упоминай источники, контекст, документы и базу знаний, не вставляй "
        "пометки вида [Source 1] — это служебные данные, они вслух не звучат. "
        "Ничего не выдумывай: используй только факты из предоставленной информации. "
        "НЕ АНАЛИЗИРУЙ ГОСТЯ И ЕГО ПОВЕДЕНИЕ: никаких «суть ситуации», «ваша "
        "реплика — коммуникативный сигнал», «вы мягко указываете», «роли "
        "перепутались». Не описывай намерения и эмоции собеседника. "
        "Не делай разбор фраз и их значения: не объясняй смысл реплики гостя, "
        "не анализируй отдельные слова, не пересказывай вопрос — сразу отвечай. "
        "Если смысл реплики не ясен — НЕ говори «не разобрал»: в словах гостя "
        "почти всегда есть смысл. Отвечай на то, что понял, или задай один "
        "короткий уточняющий вопрос. «Не разобрал» — только для набора звуков. "
        "Учитывай предыдущие реплики разговора. "
        "НЕ ОБРАЩАЙСЯ К ГОСТЮ ПО ИМЕНИ: только «ты»/«вы», без имён и титулов. "
        "Если нужного факта нет — скажи «Этого я не знаю». Максимум 40 слов."
    )
    
    CHAT_CONTEXT_ENHANCEMENT = (
        "\nОтвечай с опорой на предоставленную информацию: сначала главный факт "
        "по вопросу. Не упоминай контекст и источники вслух. Ничего не "
        "придумывай — если данных нет, честно скажи «Этого я не знаю». "
        "Не делай разбор фраз и их значения и не анализируй поведение гостя — "
        "сразу отвечай по существу. Если смысл не ясен, отвечай на то, что "
        "понял, или задай короткий уточняющий вопрос, но НЕ говори «не разобрал». "
        "Не обращайся к гостю по имени — только «ты»/«вы», без имён и титулов. "
        "Говори живо, тепло, по-человечески, обычными словами — как в разговоре."
    )
    
    def format_user_message_with_context(self, prompt, context):
        """
        Форматирует сообщение с данными для ответа.

        Меток «Context:»/«Question:» СОЗНАТЕЛЬНО нет: модель копировала эту
        лексику в ответ и вслух говорила «Согласно предоставленному контексту».
        Нейтральные русские метки уменьшают такие вставки.
        """
        return f"Данные для ответа:\n{context}\n\nВопрос гостя: {prompt}"
    
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
        # Даже без контекста системный промт — промт Валеры.
        # Раньше здесь было «You are a helpful assistant.» — и когда поиск не
        # нашёл ни одного фрагмента, модель получала АНГЛИЙСКУЮ безликую
        # инструкцию и отвечала как посторонний чат-бот.
        system_message = self.RAG_SYSTEM_MESSAGE + facts_card()
        user_message = prompt
        
        if context:
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
            messages[system_message_index]['content'] += self.CHAT_CONTEXT_ENHANCEMENT + facts_card()
        else:
            # Системного сообщения нет — создаём, иначе провайдер получит
            # только реплики и будет вести себя как «модель без промта».
            messages.insert(0, {
                'role': 'system',
                'content': self.RAG_SYSTEM_MESSAGE + self.CHAT_CONTEXT_ENHANCEMENT + facts_card(),
            })
        
        # Find the most recent user message
        user_message_index = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]['role'] == 'user':
                user_message_index = i
                break
        
        if user_message_index is not None:
            # Add context to the most recent user message.
            #
            # Метки на РУССКОМ и без двоеточий-ярлыков: английские «Context:» и
            # «Question:» модель копировала в ответ и вслух говорила «Согласно
            # предоставленному контексту». Формат совпадает с
            # format_user_message_with_context, чтобы поведение было одинаковым
            # в обоих путях.
            user_content = messages[user_message_index]['content']
            messages[user_message_index]['content'] = (
                self.format_user_message_with_context(user_content, context)
            )
    
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
        elif provider == 'deepseek':
            # Облачная модель DeepSeek: заметно точнее локальной 3B на фактах
            # из базы знаний, при этом работает без GPU.
            from app.services.deepseek_service import DeepSeekService
            return DeepSeekService()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")