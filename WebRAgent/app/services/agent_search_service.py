import logging
from app.services.llm_service import LLMFactory
from app.services.document_service import DocumentService
from app.services.rag_service import RAGService
from app.services.base_agent_service import BaseAgentService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentSearchService(BaseAgentService):
    """
    Advanced knowledge retrieval system that enables answering complex, multi-faceted questions
    by intelligently decomposing queries, searching across multiple contexts, and synthesizing 
    comprehensive answers.
    """
    
    def __init__(self):
        """Initialize Agent Search service"""
        self.document_service = DocumentService()
        self.llm_service = LLMFactory.create_llm_service()
        self.rag_service = RAGService()
    
    def process_query(self, collection_id, query, max_results=3, max_tokens=1500, strategy="direct", conversation_context=None):
        """
        Process a complex query using Agent Search methodology
        
        Args:
            collection_id (str): ID of the collection to search
            query (str): User query
            max_results (int, optional): Maximum number of context chunks to retrieve per subquery
            max_tokens (int, optional): Maximum number of tokens for LLM response
            strategy (str, optional): Strategy to use - "direct" (default) or "informed"
            conversation_context (list, optional): Previous conversation messages for context
            
        Returns:
            dict: Agent Search results with original query, decomposed subqueries, intermediate answers,
                  synthesized response, and supporting contexts
        """
        # Format the query based on conversation context
        query = self._format_conversation_context(query, conversation_context)
        
        # Choose strategy based on parameter
        if strategy == "informed":
            return self.process_query_informed(collection_id, query, max_results, max_tokens)
        else:
            return self.process_query_direct(collection_id, query, max_results, max_tokens)
    
    def process_query_direct(self, collection_id, query, max_results=3, max_tokens=1500):
        """
        Process a query using the direct decomposition strategy (original method)
        """
        # Step 1: Decompose and disambiguate the query into subqueries
        subqueries = self._decompose_query(query, is_web_search=False)
        
        # Step 2: Process each subquery to get intermediate answers
        intermediate_results = []
        all_contexts = []
        
        for subquery in subqueries:
            # Process each subquery with standard RAG
            result = self.rag_service.process_query(
                collection_id=collection_id,
                query=subquery,
                max_results=max_results
            )
            
            # Store intermediate results
            intermediate_results.append({
                'subquery': subquery,
                'answer': result['response'],
                'contexts': result['contexts']
            })
            
            # Add contexts to the overall context collection
            all_contexts.extend(result['contexts'])
        
        # Step 3: Synthesize a comprehensive answer from intermediate results
        synthesized_response = self._synthesize_answer(query, intermediate_results, is_web_search=False)
        
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
            'model_info': model_info
        }
    
    def process_query_informed(self, collection_id, query, max_results=3, max_tokens=1500):
        """
        Process a query using the informed decomposition strategy, which first gets an initial
        RAG result and then uses that context to inform the generation of more targeted subqueries
        """
        # Step 1: Get initial RAG result to inform subquery generation
        initial_result = self.rag_service.process_query(
            collection_id=collection_id,
            query=query,
            max_results=max_results
        )
        
        initial_response = initial_result['response']
        initial_contexts = initial_result['contexts']
        
        # Step 2: Use initial results to generate informed subqueries
        subqueries = self._decompose_query_informed(query, initial_response, initial_contexts, is_web_search=False)
        
        # Step 3: Process each subquery to get intermediate answers
        intermediate_results = []
        all_contexts = []
        
        # Add the initial result as our first intermediate result
        intermediate_results.append({
            'subquery': f"Initial query: {query}",
            'answer': initial_result['response'],
            'contexts': initial_result['contexts']
        })
        
        # Add initial contexts to all contexts
        all_contexts.extend(initial_result['contexts'])
        
        # Process each subquery
        for subquery in subqueries:
            # Process each subquery with standard RAG
            result = self.rag_service.process_query(
                collection_id=collection_id,
                query=subquery,
                max_results=max_results
            )
            
            # Store intermediate results
            intermediate_results.append({
                'subquery': subquery,
                'answer': result['response'],
                'contexts': result['contexts']
            })
            
            # Add contexts to the overall context collection
            all_contexts.extend(result['contexts'])
        
        # Step 4: Synthesize a comprehensive answer from intermediate results
        synthesized_response = self._synthesize_answer(query, intermediate_results, is_web_search=False)
        
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
            'model_info': model_info
        }