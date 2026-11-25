import logging
import re
from time import monotonic

from app.services.admission_examples import asks_programs
from app.services.llm_service import LLMFactory
from app.services.document_service import DocumentService
from app.services.prompt_template_service import PromptTemplateService

logger = logging.getLogger(__name__)

# Один предлог с возможной точкой — оборванная генерация. «Да», «Нет»,
# числа и другие самостоятельные короткие ответы пропускаем.
_FRAGMENT = re.compile(r"\s*(?:в|на|из|к|для|по|про|о|об|с|и|или|но)?[.!?…\s]*", re.I)


class RAGService:
    """Retrieval-Augmented Generation service"""

    def _search(self, collection_id, query, limit, history=None, original_query=""):
        higher_ed = re.compile(r"высш\w+ образован|бакалавр|специалитет|магистратур|аспирантур", re.I)
        college = re.compile(r"колледж|\bспо\b|средн\w+ профессиональн", re.I)
        scoped = higher_ed.search(query) and not college.search(query)
        started = monotonic()
        logger.info("Поиск по коллекции %s начат", collection_id)
        question = original_query or query
        program_question = query if asks_programs(query) else question
        options = dict(program_question=program_question, history=history) if asks_programs(program_question) else {}
        result = self.document_service.search_documents(
            collection_id=collection_id, query=query, limit=limit * 3 if scoped else limit, **options
        )
        rows = result.get("results", [])
        if scoped:
            # FAQ и заголовки часто дают точное совпадение по цене для другого уровня.
            def matches_level(row):
                heading = (row.get("document_title") or "") + " " + (row.get("content") or "")[:350]
                return not college.search(heading) or higher_ed.search(heading)

            rows = [row for row in rows if matches_level(row)]
        logger.info("Поиск по коллекции занял %.2f с, найдено фрагментов: %d",
                    monotonic() - started, min(limit, len(rows)))
        return rows[:limit]

    @staticmethod
    def _current_question(query, original_query):
        question = original_query or query
        message = question if question == query else f"{question}\nУточнённый смысл текущего вопроса: {query}"
        if asks_programs(query) or asks_programs(question):
            message += ("\nНазови направление и один-два его профиля из таблицы приёма. Различай направление и профиль. "
                        "Если гость спросил про платное обучение, "
                        "в первых двух предложениях дай названия и цену за год. Общую цену назови один раз. "
                        "Если спрашивают о шансах, сравни баллы гостя с проходными по конкретному профилю "
                        "и укажи год. Объясни, что результат нового конкурса заранее неизвестен. "
                        "Если спрашивают, чем занимаются на программе, поясни на примере учебной задачи. "
                        "При выборе на стыке интересов объясни, как программа их сочетает. "
                        "Если среди вопросов есть вопрос о тебе, коротко представься. "
                        "Ответь на каждую часть по данным источников. Если сведений не хватает, укажи, каких именно.")
        return message

    def process_query_stream(self, collection_id, query, max_results=3, max_tokens=1000,
                             conversation_context=None, original_query=""):
        """
        Потоковая версия process_query: генератор событий.

          {"t": "meta", "contexts": N, "model_info": {...}}   — найденные фрагменты
          {"t": "d", "x": "кусок текста"}                      — очередной кусок ответа
          {"t": "done", "response": "весь ответ"}              — конец

        Поиск и промпт те же, что в process_query; отличие только в том, что ответ модели
        отдаётся кусочками, и клиент (голосовой ассистент) может начать озвучку с первого
        предложения.
        """
        contexts = []
        for result in self._search(collection_id, query, max_results, conversation_context, original_query):
            contexts.append({
                'document_id': result.get('document_id', ''),
                'document_title': result.get('document_title', 'Unknown'),
                'content': result.get('content', ''),
                'score': result.get('score', 0.0),
                'file_path': result.get('file_path', ''),
                'page_number': result.get('page_number', 1),
            })
        formatted_context = PromptTemplateService.format_context(contexts) if contexts else ""
        yield {"t": "meta", "contexts": len(contexts),
               "model_info": {'provider': self.llm_service.get_provider_name(),
                              'model': self.llm_service.get_model_name()}}

        exchange = list(conversation_context or [])
        # Текущий вопрос — последним и без дубля (как в process_query).
        if exchange and exchange[-1].get('role') == 'user' and exchange[-1].get('content', '').strip() == query.strip():
            exchange.pop()
        valera_system = PromptTemplateService.get_system_message("rag")
        replaced = False
        for msg in exchange:
            if msg.get('role') == 'system':
                msg['content'] = valera_system
                replaced = True
        if not replaced:
            exchange.insert(0, {'role': 'system', 'content': valera_system})
        exchange.append({'role': 'user', 'content': self._current_question(query, original_query)})

        parts = []
        started = monotonic()
        started_answer = False
        for piece in self.llm_service.stream_chat_response([dict(m) for m in exchange], context=formatted_context, max_tokens=max_tokens):
            if not parts:
                logger.info("Первый фрагмент ответа модели через %.2f с после поиска", monotonic() - started)
            parts.append(piece)
            if not started_answer:
                piece = "".join(parts)
                if _FRAGMENT.fullmatch(piece):
                    continue
                started_answer = True
            yield {"t": "d", "x": piece}
        answer = "".join(parts).strip()
        if not started_answer:
            # Один повтор по тем же источникам, до сохранения ответа в историю чата.
            answer = self.llm_service.generate_chat_response(exchange, context=formatted_context, max_tokens=max_tokens)
            if _FRAGMENT.fullmatch(answer or "") or answer.startswith("Error generating"):
                answer = "Не удалось сформулировать ответ. Повторите, пожалуйста, вопрос."
            yield {"t": "d", "x": answer}
        yield {"t": "done", "response": answer, "contexts": contexts}
    
    def __init__(self):
        """Initialize RAG service"""
        self.document_service = DocumentService()
        self.llm_service = LLMFactory.create_llm_service()
    
    def process_query(self, collection_id, query, max_results=3, max_tokens=1000,
                      conversation_context=None, original_query=""):
        """
        Process a user query using RAG
        
        Args:
            collection_id (str): ID of the collection to search
            query (str): User query
            max_results (int, optional): Maximum number of context chunks to retrieve
            max_tokens (int, optional): Maximum number of tokens for LLM response
            conversation_context (list, optional): Previous conversation messages for context
            
        Returns:
            dict: RAG results with query, context, and LLM response
        """
        # Retrieve relevant context from Qdrant
        
        # Extract context from search results
        contexts = []
        for result in self._search(collection_id, query, max_results, conversation_context, original_query):
            contexts.append({
                'document_id': result.get('document_id', ''),
                'document_title': result.get('document_title', 'Unknown'),
                'content': result.get('content', ''),
                'score': result.get('score', 0.0),
                'file_path': result.get('file_path', ''),
                'page_number': result.get('page_number', 1)
            })
        
        # Format document context for LLM prompt using the improved formatter
        formatted_context = PromptTemplateService.format_context(contexts) if contexts else ""
        
        # Handle conversation context if provided
        if conversation_context and len(conversation_context) > 0:
            # Create a system message if none exists, using the improved system message with citation instructions
            current_exchange = list(conversation_context)  # Copy the conversation context

            # Make sure the CURRENT query is the LAST user message and NOT duplicated.
            # The chat route already added this query via chat.add_message('user', ...)
            # before calling get_context(), so it may already be present in the context.
            # Remove any trailing user message that equals the current query to avoid
            # duplicate "user: <query>" entries (which dilute the weight of the fresh question).
            while current_exchange and current_exchange[-1].get('role') == 'user':
                if current_exchange[-1].get('content', '').strip() == query.strip():
                    current_exchange.pop()
                    break
                # If the last user message differs (older question), keep the history but
                # stop scanning further — we only dedupe an exact trailing duplicate.
                break

            # ГАРАНТИРУЕМ правильный системный промт.
            #
            # Раньше здесь стояло `if not any(role == 'system')`. Но chat.py уже
            # вставляет СВОЙ системный месседж («This is a follow-up question in
            # an ongoing conversation…», на английском), поэтому условие не
            # срабатывало — и промт Валеры НЕ ДОБАВЛЯЛСЯ ВООБЩЕ. DeepSeek
            # получал только английскую отписку и вёл себя как «модель без
            # системного промта»: рассуждал о смысле фраз, путал имена и не
            # знал, что он робот-экскурсовод.
            #
            # Теперь мы НЕ проверяем наличие, а ПРИНУДИТЕЛЬНО заменяем все
            # системные сообщения на канонический промт Валеры.
            valera_system = PromptTemplateService.get_system_message("rag")
            replaced = False
            for msg in current_exchange:
                if msg.get('role') == 'system':
                    msg['content'] = valera_system
                    replaced = True
            # Системного сообщения не было — ставим первым.
            if not replaced:
                current_exchange.insert(0, {
                    'role': 'system',
                    'content': valera_system
                })

            # Ensure the current query is always the single last user message
            current_exchange.append({
                'role': 'user',
                'content': self._current_question(query, original_query)
            })

            # Generate response with LLM using chat format
            llm_response = self.llm_service.generate_chat_response(
                messages=current_exchange,
                context=formatted_context,
                max_tokens=max_tokens
            )
        else:
            # No conversation context, use standard response format
            llm_response = self.llm_service.generate_response(
                prompt=self._current_question(query, original_query),
                context=formatted_context,
                max_tokens=max_tokens
            )
        
        # Get the model information from the LLM service
        model_info = {
            'provider': self.llm_service.get_provider_name(),
            'model': self.llm_service.get_model_name()
        }
        
        # Format final response
        return {
            'query': query,
            'contexts': contexts,
            'response': llm_response,
            'model_info': model_info
        }
