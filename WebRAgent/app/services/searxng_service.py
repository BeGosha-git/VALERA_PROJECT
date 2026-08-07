import os
import json
import logging
import requests
from urllib.parse import urlencode
from app.services.llm_service import LLMFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearXNGService:
    """
    Service for performing web searches using SearXNG
    """
    
    def __init__(self):
        """Initialize SearXNG service with configuration from environment variables"""
        self.base_url = os.environ.get('SEARXNG_URL', 'http://searxng:8080')
        self.search_url = f"{self.base_url}/search"
        self.api_url = f"{self.base_url}/search"  # SearXNG API endpoint for JSON results
        self.results_per_page = int(os.environ.get('SEARXNG_RESULTS_PER_PAGE', 10))
        
        logger.info(f"Initialized SearXNGService with URL: {self.base_url}")
    
    def search(self, query, num_results=10, search_type='general'):
        """
        Perform a web search using SearXNG
        
        Args:
            query (str): Search query
            num_results (int): Number of results to return
            search_type (str): Type of search (general, news, images, etc.)
            
        Returns:
            list: List of search results with title, snippet, and URL
        """
        try:
            # Format the request parameters
            params = {
                'q': query,
                'format': 'json',
                'categories': self._map_search_type(search_type),
                'results': min(num_results, 25),  # Prevent overly large requests
                'language': 'en'
            }
            
            # Make the API request
            logger.info(f"Performing SearXNG search for: {query}")
            headers = {
                'Accept': 'application/json',
                'User-Agent': 'RAGSystem/1.0'
            }
            
            response = requests.get(
                self.api_url, 
                params=params,
                headers=headers,
                timeout=10
            )
            
            # Check the response status
            if response.status_code != 200:
                logger.error(f"SearXNG search failed with status code: {response.status_code}")
                return []
            
            # Parse the results
            try:
                search_results = response.json()
            except json.JSONDecodeError:
                logger.error("Failed to parse SearXNG response as JSON")
                return []
            
            # Extract and format the relevant information
            formatted_results = self._format_results(search_results)
            logger.info(f"Retrieved {len(formatted_results)} results from SearXNG")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during SearXNG search: {str(e)}")
            return []
    
    def process_query(self, query, max_results=10, search_type='general', conversation_context=None):
        """
        Process a search query and return results in a format compatible with the RAG service
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of results to return
            search_type (str): Type of search (general, news, images, etc.)
            conversation_context (list, optional): Previous conversation messages for context
            
        Returns:
            dict: Search results in RAG-compatible format
        """
        # Perform the web search
        search_results = self.search(query, max_results, search_type)
        
        # Convert web search results to contextual format expected by RAG interface
        contexts = []
        for result in search_results:
            contexts.append({
                'document_id': f"web_{hash(result['url'])}",
                'document_title': result['title'],
                'content': result['snippet'],
                'score': result.get('score', 0.95),  # Most web results don't have scores
                'file_path': '',
                'url': result['url'],
                'source_type': 'web'
            })
            
        # Use LLM service to provide an interpreted response
        llm_service = LLMFactory.create_llm_service()
        
        # Format web search content for LLM
        formatted_context = self._format_web_results_for_llm(contexts)
        
        if conversation_context and len(conversation_context) > 0:
            # Process with conversation context if available
            # Create a system message if none exists
            current_exchange = list(conversation_context)  # Copy to avoid modifying original
            has_system = any(msg['role'] == 'system' for msg in current_exchange)
            
            if not has_system:
                current_exchange.insert(0, {
                    'role': 'system',
                    'content': "You are a helpful assistant answering questions based on web search results."
                })
            
            # Add the current query
            current_exchange.append({
                'role': 'user',
                'content': query
            })
            
            # Generate response with LLM using chat format
            response = llm_service.generate_chat_response(
                messages=current_exchange,
                context=formatted_context,
                max_tokens=1000
            )
        else:
            # Generate an interpreted response from the web search results
            prompt = f"""
            Based on the following web search results for the query: "{query}", 
            provide a comprehensive and well-structured answer. 
            
            Analyze the information from different sources, identify the most relevant facts,
            and synthesize a coherent response that directly answers the query.
            
            Web search results:
            {formatted_context}
            
            Your answer should:
            1. Directly address the original query
            2. Integrate information from multiple sources when available
            3. Present a logical flow of information
            4. Note any conflicting information found and provide a balanced perspective
            5. Acknowledge if the search results don't fully answer the query
            """
            
            response = llm_service.generate_response(
                prompt=prompt,
                context=None,
                max_tokens=1000
            )
        
        # Get model information
        model_info = {
            'provider': llm_service.get_provider_name(),
            'model': llm_service.get_model_name()
        }
        
        # Return in the expected format
        return {
            'query': query,
            'contexts': contexts,
            'response': response,
            'search_type': 'web',
            'source': 'searxng',
            'model_info': model_info
        }
        
    def _format_web_results_for_llm(self, contexts):
        """Format web search results for LLM consumption"""
        if not contexts:
            return "No web search results found."
            
        formatted_text = "Web search results:\n\n"
        for i, context in enumerate(contexts):
            formatted_text += f"[{i+1}] {context['document_title']}\n"
            formatted_text += f"URL: {context.get('url', 'Unknown URL')}\n"
            formatted_text += f"{context['content']}\n\n"
            
        return formatted_text
        
    def _map_search_type(self, search_type):
        """Map search type to SearXNG category"""
        mapping = {
            'general': 'general',
            'news': 'news',
            'images': 'images',
            'videos': 'videos',
            'files': 'files',
            'science': 'science',
            'it': 'it',
            'social media': 'social media'
        }
        return mapping.get(search_type.lower(), 'general')
    
    def _format_results(self, search_results):
        """Format SearXNG results into a standardized structure"""
        formatted = []
        
        # Check if results exist and extract them
        if 'results' not in search_results:
            return formatted
        
        for result in search_results['results']:
            # Skip results without URLs or titles
            if 'url' not in result or 'title' not in result:
                continue
                
            formatted_result = {
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'snippet': result.get('content', ''),
                'source': result.get('engine', 'web'),
                'score': float(result.get('score', 0.95))
            }
            
            formatted.append(formatted_result)
            
        return formatted
    
    def _generate_search_summary(self, query, contexts):
        """Generate a simple summary of search results"""
        if not contexts:
            return f"No web search results found for '{query}'."
        
        summary = f"Web search results for '{query}':\n\n"
        
        for i, context in enumerate(contexts[:5], 1):
            summary += f"{i}. **{context['document_title']}**\n"
            summary += f"   {context['content']}\n"
            summary += f"   [Source]({context['url']})\n\n"
        
        if len(contexts) > 5:
            summary += f"*...and {len(contexts) - 5} more results.*"
        
        return summary