from app.services.llm_service import LLMFactory
from app.services.document_service import DocumentService
from app.services.prompt_template_service import PromptTemplateService

class RAGService:
    """Retrieval-Augmented Generation service"""
    
    def __init__(self):
        """Initialize RAG service"""
        self.document_service = DocumentService()
        self.llm_service = LLMFactory.create_llm_service()
    
    def process_query(self, collection_id, query, max_results=3, max_tokens=1000, conversation_context=None):
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
        search_results = self.document_service.search_documents(
            collection_id=collection_id,
            query=query,
            limit=max_results
        )
        
        # Extract context from search results
        contexts = []
        for result in search_results.get('results', []):
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
            if not any(msg['role'] == 'system' for msg in conversation_context):
                conversation_context.insert(0, {
                    'role': 'system',
                    'content': PromptTemplateService.get_system_message("citation_focus")
                })
            
            # Add the current query
            current_exchange = list(conversation_context)  # Copy the conversation context
            current_exchange.append({
                'role': 'user',
                'content': query
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
                prompt=query,
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