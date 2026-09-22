import re
import logging
from app.services.prompt_template_service import PromptTemplateService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseAgentService:
    """Base class for agent search services that implement query decomposition and synthesis"""
    
    def _format_conversation_context(self, query, conversation_context):
        """Format the conversation context to provide to the agent"""
        if conversation_context and len(conversation_context) > 0:
            # Extract the most recent exchanges to inform the agent
            formatted_context = "Previous conversation:\n"
            for message in conversation_context[-3:]:  # Just use the last 3 messages for clarity
                role = message['role']
                if role != 'system':  # Skip system messages in the formatted context
                    formatted_context += f"{role.capitalize()}: {message['content']}\n"
            
            # Add an instruction for the agent
            return f"{formatted_context}\n\nConsidering the conversation above, answer this follow-up question: {query}"
        
        return query
    
    def _parse_subqueries(self, response):
        """Parse LLM response to extract subqueries"""
        subqueries = []
        for line in response.strip().split('\n'):
            # Remove bullet points and leading/trailing whitespace
            cleaned_line = re.sub(r'^\s*[-•*]\s*', '', line).strip()
            
            # Skip empty lines
            if cleaned_line:
                subqueries.append(cleaned_line)
        
        return subqueries
    
    def _decompose_query(self, query, is_web_search=False):
        """
        Decompose a complex query into multiple simpler subqueries
        
        Args:
            query (str): The original complex query
            is_web_search (bool): Whether this is for web search (affects prompt)
            
        Returns:
            list: List of subqueries
        """
        # Get the appropriate prompt using the PromptTemplateService
        decompose_type = "web_search" if is_web_search else "document_search"
        role_type = "web_searcher" if is_web_search else "decomposer"
        
        decompose_prompt = PromptTemplateService.get_decomposition_prompt(
            query=query,
            type=decompose_type,
            role=PromptTemplateService.get_role(role_type),
            format_instructions=PromptTemplateService.get_format_instructions("bullet_list_queries")
        )
        
        # Generate the decomposition
        response = self.llm_service.generate_response(
            prompt=decompose_prompt,
            context=None,
            max_tokens=500
        )
        
        # Parse the subqueries
        subqueries = self._parse_subqueries(response)
        
        # If no subqueries were extracted or the LLM failed, return the original query
        if not subqueries:
            logger.warning("Failed to decompose query, using original query")
            return [query]
        
        return subqueries
    
    def _decompose_query_informed(self, query, initial_response, initial_contexts, is_web_search=False):
        """
        Decompose a query using context from initial results
        
        Args:
            query (str): The original complex query
            initial_response (str): The response from initial search
            initial_contexts (list): The contexts retrieved in the initial search
            is_web_search (bool): Whether this is for web search (affects prompt)
            
        Returns:
            list: List of subqueries
        """
        # Format contexts for the decomposition prompt using efficient context formatter
        formatted_contexts = PromptTemplateService.format_context(initial_contexts[:3], max_length=2000)
        
        # Set up parameters for the informed decomposition prompt
        search_type = "web search" if is_web_search else "search"
        query_type = "search queries" if is_web_search else "questions"
        role_type = "web_searcher" if is_web_search else "decomposer"
        
        # Set up specific instructions based on search type
        specific_instructions = """
        - Be phrased to maximize relevant search engine results
        - Use search engine-friendly syntax (short, precise terms without unnecessary words)
        """ if is_web_search else """
        - Be self-contained and specific
        - Be phrased as a complete question
        """
        
        # Get the prompt from the template service
        decompose_prompt = PromptTemplateService.DECOMPOSITION["informed_decomposition"].format(
            role=PromptTemplateService.get_role(role_type),
            query=query,
            search_type=search_type,
            context=formatted_contexts,
            query_type=query_type,
            specific_instructions=specific_instructions,
            format_instructions=PromptTemplateService.get_format_instructions("bullet_list_queries")
        )
        
        # Generate the decomposition
        response = self.llm_service.generate_response(
            prompt=decompose_prompt,
            context=None,
            max_tokens=500
        )
        
        # Parse the subqueries
        subqueries = self._parse_subqueries(response)
        
        # If no subqueries were extracted or the LLM failed, return default follow-ups
        if not subqueries:
            logger.warning(f"Failed to decompose query using informed strategy, using default follow-up queries")
            if is_web_search:
                subqueries = [
                    f"{query} latest information",
                    f"{query} alternative perspectives"
                ]
            else:
                subqueries = [
                    f"What additional details can be found about {query}?",
                    f"Are there any alternative perspectives on {query}?"
                ]
        
        return subqueries
    
    def _synthesize_answer(self, original_query, intermediate_results, is_web_search=False):
        """
        Synthesize a comprehensive answer from intermediate results
        
        Args:
            original_query (str): The original user query
            intermediate_results (list): List of dictionaries with subquery results
            is_web_search (bool): Whether this is for web search synthesis
            
        Returns:
            str: Synthesized comprehensive answer
        """
        # Format intermediate results for the synthesis prompt
        formatted_results = ""
        for i, result in enumerate(intermediate_results):
            query_type = "Search Query" if is_web_search else "Subquery"
            formatted_results += f"{query_type} {i+1}: {result['subquery']}\n"
            formatted_results += f"Results: {result['answer']}\n\n"
        
        # Use the template service for synthesis prompt
        synthesis_prompt = PromptTemplateService.get_synthesis_prompt(
            query=original_query,
            results=formatted_results,
            is_web_search=is_web_search,
            role=PromptTemplateService.get_role("synthesizer")
        )
        
        # Generate synthesized response
        synthesized_answer = self.llm_service.generate_response(
            prompt=synthesis_prompt,
            context=None,
            max_tokens=1000
        )
        
        return synthesized_answer