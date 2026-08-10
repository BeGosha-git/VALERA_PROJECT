import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.services.llm_service import LLMFactory
from app.services.searxng_service import SearXNGService
from app.services.base_agent_service import BaseAgentService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSearchAgentService(BaseAgentService):
    """
    Advanced web search system that enables answering complex, multi-faceted questions
    by intelligently decomposing queries, searching across the web, and synthesizing 
    comprehensive answers.
    """
    
    def __init__(self):
        """Initialize Web Search Agent service"""
        self.searxng_service = SearXNGService()
        self.llm_service = LLMFactory.create_llm_service()
    
    def process_query(self, query, max_results=5, max_tokens=1500, strategy="direct", conversation_context=None):
        """
        Process a complex query using Web Search Agent methodology
        
        Args:
            query (str): User query
            max_results (int, optional): Maximum number of search results per subquery
            max_tokens (int, optional): Maximum number of tokens for LLM response
            strategy (str, optional): Strategy to use - "direct" or "informed"
            conversation_context (list, optional): Previous conversation messages for context
            
        Returns:
            dict: Web Search Agent results with original query, decomposed subqueries, 
                 intermediate answers, search results, and synthesized response
        """
        # Format the query based on conversation context
        query = self._format_conversation_context(query, conversation_context)
            
        # Choose strategy based on parameter
        if strategy == "informed":
            return self.process_query_informed(query, max_results, max_tokens)
        else:
            return self.process_query_direct(query, max_results, max_tokens)
    
    def _process_subquery(self, subquery, max_results):
        """
        Обрабатывает один подзапрос (для параллельного выполнения).
        Использует СЫРЫЕ результаты поиска (без LLM-интерпретации каждого
        подзапроса) — это значительно быстрее. Финальный синтез ответа
        делает один LLM-вызов в _synthesize_answer.
        """
        try:
            results = self.searxng_service.search(
                query=subquery,
                num_results=max_results
            )
            # Формируем компактный ответ из сниппетов для синтеза
            answer = self._format_raw_results_for_synthesis(subquery, results)
            return {
                'subquery': subquery,
                'answer': answer,
                'contexts': results
            }
        except Exception as e:
            logger.error(f"Subquery failed ({subquery}): {e}")
            return {
                'subquery': subquery,
                'answer': '',
                'contexts': []
            }

    @staticmethod
    def _format_raw_results_for_synthesis(subquery, results):
        """Собирает сырые результаты в компактный текст для синтеза."""
        if not results:
            return ""
        parts = [f"По запросу '{subquery}' найдено:"]
        for r in results[:5]:
            title = r.get('title', '')
            snippet = r.get('snippet', '') or r.get('content', '')
            if title:
                parts.append(f"- {title}: {snippet}")
        return "\n".join(parts)

    def process_query_direct(self, query, max_results=5, max_tokens=1500):
        """
        Process a query using the direct decomposition strategy.
        Subqueries are searched IN PARALLEL to reduce latency.
        """
        # Step 1: Decompose the query into search subqueries
        subqueries = self._decompose_query(query, is_web_search=True)

        # Step 2: Process each subquery IN PARALLEL (big latency win)
        intermediate_results = []
        all_contexts = []

        with ThreadPoolExecutor(max_workers=min(len(subqueries), 4)) as pool:
            futures = {
                pool.submit(self._process_subquery, sq, max_results): sq
                for sq in subqueries
            }
            for future in as_completed(futures):
                result = future.result()
                if result['answer']:
                    intermediate_results.append(result)
                    for context in result['contexts']:
                        if 'source_type' not in context:
                            context['source_type'] = 'web'
                        all_contexts.append(context)

        # Keep original subquery order if possible
        intermediate_results.sort(key=lambda r: subqueries.index(r['subquery']) if r['subquery'] in subqueries else 0)

        # Step 3: Synthesize a comprehensive answer from intermediate results
        synthesized_response = self._synthesize_answer(query, intermediate_results, is_web_search=True)

        # Get the model information from the LLM service
        model_info = {
            'provider': self.llm_service.get_provider_name(),
            'model': self.llm_service.get_model_name()
        }

        # Format final response with all information
        return {
            'query': query,
            'subqueries': [ir['subquery'] for ir in intermediate_results],
            'intermediate_results': intermediate_results,
            'contexts': all_contexts,
            'response': synthesized_response,
            'strategy': 'direct',
            'search_type': 'web',
            'model_info': model_info
        }
    
    def process_query_informed(self, query, max_results=5, max_tokens=1500):
        """
        Process a query using the informed decomposition strategy, which first gets an initial
        web search result and then uses that context to inform the generation of more targeted subqueries
        """
        # Step 1: Get initial web search result to inform subquery generation
        initial_result = self.searxng_service.process_query(
            query=query,
            max_results=max_results
        )
        
        initial_response = initial_result['response']
        initial_contexts = initial_result['contexts']
        
        # Step 2: Use initial results to generate informed subqueries
        subqueries = self._decompose_query_informed(query, initial_response, initial_contexts, is_web_search=True)
        
        # Step 3: Process each subquery IN PARALLEL
        intermediate_results = []
        all_contexts = []
        
        # Add the initial result as our first intermediate result
        intermediate_results.append({
            'subquery': f"Initial query: {query}",
            'answer': initial_result['response'],
            'contexts': initial_result['contexts']
        })
        
        # Add initial contexts to all contexts
        for context in initial_result['contexts']:
            # Make sure each context has source_type set to 'web'
            if 'source_type' not in context:
                context['source_type'] = 'web'
            all_contexts.append(context)
        
        # Process remaining subqueries in parallel
        with ThreadPoolExecutor(max_workers=min(len(subqueries), 4)) as pool:
            futures = {
                pool.submit(self._process_subquery, sq, max_results): sq
                for sq in subqueries
            }
            for future in as_completed(futures):
                result = future.result()
                if result['answer']:
                    intermediate_results.append(result)
                    for context in result['contexts']:
                        if 'source_type' not in context:
                            context['source_type'] = 'web'
                        all_contexts.append(context)
        
        # Step 4: Synthesize a comprehensive answer from intermediate results
        synthesized_response = self._synthesize_answer(query, intermediate_results, is_web_search=True)
        
        # Get the model information from the LLM service
        model_info = {
            'provider': self.llm_service.get_provider_name(),
            'model': self.llm_service.get_model_name()
        }
        
        # Format final response with all information
        return {
            'query': query,
            'subqueries': [ir['subquery'] for ir in intermediate_results if ir['subquery'] != f"Initial query: {query}"],
            'intermediate_results': intermediate_results,
            'contexts': all_contexts,
            'response': synthesized_response,
            'strategy': 'informed',
            'search_type': 'web',
            'model_info': model_info
        }