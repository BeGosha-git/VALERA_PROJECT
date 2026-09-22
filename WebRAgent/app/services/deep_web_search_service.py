import os
import logging
import concurrent.futures
import re
from app.services.searxng_service import SearXNGService
from app.services.web_scraper_service import WebScraperService
from app.services.llm_service import LLMFactory
from app.services.base_agent_service import BaseAgentService
from app.services.prompt_template_service import PromptTemplateService

# Configure logging
logger = logging.getLogger(__name__)

class DeepWebSearchService(BaseAgentService):
    """
    Service for performing deep web searches using SearXNG and web scraping
    This service enhances standard web searches by scraping full content from search results
    Can optionally use agent capabilities for query decomposition.
    """
    
    def __init__(self):
        """Initialize Deep Web Search service"""
        self.searxng_service = SearXNGService()
        self.scraper_service = WebScraperService()
        self.max_results_to_scrape = int(os.environ.get('DEEP_SEARCH_MAX_URLS', 5))
        self.parallel_scraping = os.environ.get('DEEP_SEARCH_PARALLEL', 'True').lower() == 'true'
        self.max_workers = int(os.environ.get('DEEP_SEARCH_MAX_WORKERS', 3))
        self.llm_service = LLMFactory.create_llm_service()
        
        logger.info(f"Initialized DeepWebSearchService with max URLs: {self.max_results_to_scrape}")
    
    def process_query(self, query, max_results=5, search_type='general', conversation_context=None, use_agent=False, agent_strategy='direct'):
        """
        Process a search query with deep web content extraction
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of search results to consider
            search_type (str): Type of search (general, news, etc.)
            conversation_context (list, optional): Previous conversation messages
            use_agent (bool): Whether to use agent-based decomposition
            agent_strategy (str): Strategy for decomposition ('direct' or 'informed')
            
        Returns:
            dict: Deep search results with scraped content
        """
        # Format the query based on conversation context
        query = self._format_conversation_context(query, conversation_context) if conversation_context else query
        
        logger.info(f"Processing deep web search for query: '{query}' (with agent: {use_agent}, strategy: {agent_strategy if use_agent else 'n/a'})")
        
        if use_agent:
            # Use agent-based approach with query decomposition
            return self._process_with_agent(query, max_results, search_type, agent_strategy)
        else:
            # Standard deep search approach
            return self._process_standard(query, max_results, search_type)
            
    def _process_with_agent(self, query, max_results=5, search_type='general', strategy='direct'):
        """
        Process a query using agent-based query decomposition and deep search
        
        Args:
            query (str): User's query
            max_results (int): Maximum results per subquery
            search_type (str): Type of search
            strategy (str): Agent strategy ('direct' or 'informed')
            
        Returns:
            dict: Search results with subqueries and deep content analysis
        """
        logger.info(f"Using agent-based deep search with strategy: {strategy}")
        
        if strategy == 'informed':
            # Step 1: Get initial results to inform decomposition
            initial_search_results = self.searxng_service.search(query, num_results=max_results, search_type=search_type)
            
            if not initial_search_results:
                logger.warning(f"No initial search results found for query: {query}")
                # Get the model information
                model_info = {
                    'provider': self.llm_service.get_provider_name(),
                    'model': self.llm_service.get_model_name()
                }
                
                return {
                    'query': query,
                    'contexts': [],
                    'scraped_contents': [],
                    'subqueries': [],
                    'response': f"No web search results found for '{query}'.",
                    'search_type': 'deep_web_agent',
                    'source': 'deep_search',
                    'strategy': strategy,
                    'model_info': model_info
                }
            
            # Step 2: Scrape and analyze the initial results
            initial_urls = [result['url'] for result in initial_search_results[:self.max_results_to_scrape]]
            
            initial_scraped = self._scrape_urls(initial_urls)
            initial_analyzed = self._analyze_contents(query, initial_scraped)
            
            # Get initial response for context to inform decomposition
            initial_response = "No initial content found"
            initial_contexts = []
            
            if initial_analyzed:
                # Create contexts for decomposition
                initial_contexts = self._create_contexts(initial_analyzed)
                
                # Format the initial analyses for decomposition
                formatted_contexts = ""
                for i, content in enumerate(initial_analyzed):
                    formatted_contexts += f"Source {i+1}: {content['title']}\n"
                    formatted_contexts += f"URL: {content['url']}\n"
                    formatted_contexts += f"{content['analysis']}\n\n"
                
                # Generate informed subqueries based on initial results
                subqueries = self._decompose_query_informed(query, formatted_contexts, initial_contexts, is_web_search=True)
            else:
                # Fall back to direct decomposition if initial scraping failed
                subqueries = self._decompose_query(query, is_web_search=True)
                
        else:  # 'direct' strategy
            # Use direct decomposition
            subqueries = self._decompose_query(query, is_web_search=True)
        
        # Save all information
        all_scraped_contents = []
        all_analyzed_contents = []
        all_contexts = []
        subquery_results = []
        
        # Process each subquery
        for subquery in subqueries:
            logger.info(f"Processing subquery: '{subquery}'")
            
            # Search for the subquery
            search_results = self.searxng_service.search(subquery, num_results=max_results, search_type=search_type)
            
            if not search_results:
                logger.warning(f"No results for subquery: {subquery}")
                continue
                
            # Extract and scrape URLs
            urls_to_scrape = [result['url'] for result in search_results[:self.max_results_to_scrape]]
            scraped_contents = self._scrape_urls(urls_to_scrape)
            analyzed_contents = self._analyze_contents(subquery, scraped_contents)
            
            # Create contexts
            subquery_contexts = self._create_contexts(analyzed_contents)
            
            # Add to our collections
            all_scraped_contents.extend(scraped_contents)
            all_analyzed_contents.extend(analyzed_contents)
            all_contexts.extend(subquery_contexts)
            
            # Generate a response for this subquery
            if analyzed_contents:
                subquery_response = self._analyze_subquery(subquery, analyzed_contents)
            else:
                subquery_response = f"No detailed information found for '{subquery}'"
            
            subquery_results.append({
                'subquery': subquery,
                'answer': subquery_response,
                'contexts': subquery_contexts
            })
        
        # If we have initial results in informed mode, add them
        if strategy == 'informed' and initial_analyzed:
            subquery_results.insert(0, {
                'subquery': f"Initial query: {query}",
                'answer': "Initial context gathering for informed decomposition",
                'contexts': initial_contexts
            })
        
        # Generate comprehensive answer
        final_response = self._synthesize_answer(query, subquery_results, is_web_search=True)
        
        # Get the model information
        model_info = {
            'provider': self.llm_service.get_provider_name(),
            'model': self.llm_service.get_model_name()
        }
        
        # Return complete results
        return {
            'query': query,
            'subqueries': [result['subquery'] for result in subquery_results if not result['subquery'].startswith("Initial query")],
            'intermediate_results': subquery_results,
            'contexts': all_contexts,
            'scraped_contents': all_analyzed_contents,
            'response': final_response,
            'search_type': 'deep_web_agent',
            'source': 'deep_search',
            'strategy': strategy,
            'model_info': model_info
        }
    
    def _analyze_subquery(self, subquery, analyzed_contents):
        """Generate a response for a specific subquery based on analyzed content"""
        # Prepare content for analysis
        sources_content = ""
        for i, content in enumerate(analyzed_contents):
            sources_content += f"Source {i+1}: {content['title']}\n"
            sources_content += f"URL: {content['url']}\n"
            sources_content += f"{content['analysis']}\n\n"
        
        # Use the specialized subquery analysis template
        subquery_analysis_prompt = PromptTemplateService.SPECIALIZED["subquery_analysis"].format(
            query=subquery,
            sources_content=sources_content
        )
        
        # Generate subquery-specific response
        try:
            response = self.llm_service.generate_response(
                prompt=subquery_analysis_prompt,
                max_tokens=600
            )
            return response
        except Exception as e:
            logger.error(f"Error generating subquery analysis: {e}")
            return f"Error analyzing content for '{subquery}'"
    
    def _create_contexts(self, analyzed_contents):
        """Create contexts from analyzed web content"""
        contexts = []
        
        for content in analyzed_contents:
            display_content = content['analysis']
            
            contexts.append({
                'document_id': f"deep_web_{hash(content['url'])}",
                'document_title': content['title'],
                'content': display_content,
                'score': 0.98,  # Deep content generally has higher relevance
                'file_path': '',
                'url': content['url'],
                'source_type': 'deep_web'
            })
        
        return contexts
    
    def _scrape_urls(self, urls):
        """Scrape multiple URLs, handling parallel/sequential modes"""
        scraped_contents = []
        
        if self.parallel_scraping and len(urls) > 1:
            # Parallel scraping
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.max_workers, len(urls))) as executor:
                future_to_url = {executor.submit(self.scraper_service.scrape_url, url): url for url in urls}
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        content = future.result()
                        if content and content.get('success', False):
                            scraped_contents.append(content)
                    except Exception as e:
                        logger.error(f"Error processing {url}: {str(e)}")
        else:
            # Sequential scraping
            scraped_contents = self.scraper_service.scrape_urls(urls, max_urls=self.max_results_to_scrape)
        
        logger.info(f"Successfully scraped {len(scraped_contents)} web pages")
        return scraped_contents
    
    def _analyze_contents(self, query, scraped_contents):
        """Analyze scraped content with LLM for relevant information"""
        analyzed_contents = []
        
        for content in scraped_contents:
            # Generate a concise summary and key points from the full content using improved template
            analysis_prompt = PromptTemplateService.WEB_ANALYSIS["content_analysis"].format(
                role=PromptTemplateService.get_role("web_analyzer"),
                query=query,
                title=content['title'],
                url=content['url'],
                content=content['content']
            )
            
            try:
                analysis = self.llm_service.generate_response(
                    prompt=analysis_prompt,
                    max_tokens=500
                )
                
                analyzed_contents.append({
                    'url': content['url'],
                    'title': content['title'], 
                    'original_content': content['content'][:500] + "..." if len(content['content']) > 500 else content['content'],
                    'analysis': analysis
                })
            except Exception as e:
                logger.error(f"Error analyzing content from {content['url']}: {str(e)}")
        
        return analyzed_contents
    
    def _process_standard(self, query, max_results=5, search_type='general'):
        """Standard deep search processing without agent capabilities"""
        # Step 1: Perform initial search with SearXNG
        search_results = self.searxng_service.search(query, num_results=max_results, search_type=search_type)
        
        if not search_results:
            logger.warning(f"No search results found for query: {query}")
            # Get the model information
            model_info = {
                'provider': self.llm_service.get_provider_name(),
                'model': self.llm_service.get_model_name()
            }
            
            return {
                'query': query,
                'contexts': [],
                'scraped_contents': [],
                'response': f"No web search results found for '{query}'.",
                'search_type': 'deep_web',
                'source': 'deep_search',
                'model_info': model_info
            }
        
        # Step 2: Extract URLs to scrape
        urls_to_scrape = [result['url'] for result in search_results[:self.max_results_to_scrape]]
        logger.info(f"Extracted {len(urls_to_scrape)} URLs to scrape")
        
        # Step 3: Scrape the web pages (using our helper method)
        scraped_contents = self._scrape_urls(urls_to_scrape)
        
        # Step 4: Process each scraped content with LLM for individual analysis
        analyzed_contents = self._analyze_contents(query, scraped_contents)
        
        # Step 5: Convert scraped contents to context format
        contexts = self._create_contexts(analyzed_contents)
        
        # Step 6: Use LLM to synthesize a comprehensive response from all analyzed contents
        if analyzed_contents:
            # Format the results in the structure expected by the synthesis template
            formatted_results = ""
            for i, content in enumerate(analyzed_contents):
                formatted_results += f"Web Source {i+1}: {content['title']} ({content['url']})\n"
                formatted_results += f"Analysis: {content['analysis']}\n\n"
            
            # Use the synthesis template from the prompt service
            synthesis_prompt = PromptTemplateService.get_synthesis_prompt(
                query=query,
                results=formatted_results,
                is_web_search=True,
                role=PromptTemplateService.get_role("synthesizer")
            )
            
            try:
                response = self.llm_service.generate_response(
                    prompt=synthesis_prompt,
                    max_tokens=1500
                )
            except Exception as e:
                logger.error(f"Error synthesizing deep search response: {str(e)}")
                response = f"Error generating deep search response: {str(e)}"
        else:
            # Fallback to standard search if no pages were successfully scraped and analyzed
            formatted_context = self.searxng_service._format_web_results_for_llm([c for c in contexts if c])
            
            response = self.llm_service.generate_response(
                prompt=f"""
                Based on the following search results for the query: "{query}", 
                provide a comprehensive answer. Note that deep content extraction was attempted but unsuccessful.
                
                Search results:
                {formatted_context}
                
                Your answer should address the query directly and note any limitations in the available information.
                """,
                max_tokens=1000
            )
        
        # Get the model information
        model_info = {
            'provider': self.llm_service.get_provider_name(),
            'model': self.llm_service.get_model_name()
        }
        
        # Return results in the expected format
        return {
            'query': query,
            'contexts': contexts,
            'scraped_contents': analyzed_contents,
            'response': response,
            'search_type': 'deep_web',
            'source': 'deep_search',
            'model_info': model_info
        }