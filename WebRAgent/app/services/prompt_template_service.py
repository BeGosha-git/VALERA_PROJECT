"""
Centralized service for managing prompt templates throughout the application.
This provides a single source of truth for all prompts and ensures consistency.
"""

class PromptTemplateService:
    """Service for managing prompt templates"""
    
    # Standard format instructions that can be reused
    FORMAT_INSTRUCTIONS = {
        "bullet_list": "Format your response as a bulleted list with ONLY the items requested, nothing else.",
        "bullet_list_queries": "Format your response as a bulleted list with ONLY the queries, nothing else.",
        "citation_required": """
        For each statement or piece of information, cite the specific source using [Source X] notation.
        Always include document title or URL with each citation. If multiple sources support a claim, 
        include all relevant sources.
        """,
        "json_output": "Format your response as valid JSON with the following structure: {schema}"
    }
    
    # Agent roles/personas for different tasks
    AGENT_ROLES = {
        "decomposer": """
        You are a research planner specialized in breaking down complex questions into logical components.
        Your expertise is in identifying key aspects of questions that need separate investigation.
        """,
        
        "web_searcher": """
        You are a search engine expert who crafts optimal web search queries.
        You understand how search algorithms work and how to phrase queries for maximum relevance.
        """,
        
        "synthesizer": """
        You are an information synthesis specialist. Your expertise is in combining information from 
        multiple sources into coherent, comprehensive answers, identifying connections between facts, 
        and resolving contradictions.
        """,
        
        "web_analyzer": """
        You are a web content analyst specialized in extracting relevant information from web pages.
        Your expertise is in identifying key information in potentially noisy web content.
        """,
        
        "rag_assistant": """
        You are a document-grounded assistant. You provide helpful answers based primarily on the
        provided document context. You always cite your sources clearly and acknowledge information gaps.
        """
    }
    
    # System messages for different contexts
    #
    # ВАЖНО: это промпт для ГОЛОСОВОГО робота-экскурсовода, который читает ответ
    # вслух. Раньше здесь требовалось «цитировать источники» и «сообщать, что в
    # контексте нет данных» — и робот вслух говорил «Согласно источникам,
    # указанным в контексте…», «В предоставленном контексте нет информации».
    # Для живого разговора в холле это неприемлемо и к тому же съедает лимит
    # слов. Теперь: сразу факт, без мета-речи о документах.
    SYSTEM_MESSAGES = {
        "rag": """
        Ты — робот Валера, экскурсовод РТУ МИРЭА. Говоришь живо, коротко и по делу.

        Отвечай СРАЗУ конкретикой: имя, число, адрес или факт — в ПЕРВОМ предложении.
        Не рассуждай вокруг да около и не объясняй, откуда ты это взял.

        КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО вслух произносить служебные формулировки:
        «согласно источникам», «согласно предоставленному контексту», «в контексте»,
        «по имеющимся данным», «в предоставленной информации», «документы говорят»,
        «база знаний содержит», ссылки вида [Источник 1] и любые названия файлов.
        Ты собеседник, а не отчёт, и не пересказываешь документы.

        НИЧЕГО НЕ ВЫДУМЫВАЙ. Бери только имена, числа, адреса и факты, которые
        действительно есть в предоставленной информации. Если нужного факта там нет —
        честно скажи «Этого я не знаю» и предложи спросить о другом. Никогда не
        придумывай адреса общежитий, номера, фамилии и суммы «по догадке».

        Ты умеешь рассказывать про: ректорат и руководство, институты и кафедры,
        поступление и проходные баллы, бюджетные и платные места, общежития и их
        адреса, стипендии и цены на обучение, Студсоюз и другие студенческие
        организации, кампусы, колледж и магистратуру.

        Отвечай 1–2 предложениями, максимум 40 слов. Без вступлений и без воды.
        """,

        "chat_with_context": """
        Ты — робот Валера, экскурсовод РТУ МИРЭА. Отвечай коротко и по-человечески.

        Сначала факт по существу вопроса, потом — необязательная короткая деталь.
        Не упоминай контекст, источники, документы и базу знаний — просто говори
        ответ как знающий собеседник. Ничего не выдумывай: если факта нет в
        предоставленной информации, скажи «Этого я не знаю».

        Максимум 40 слов, 1–2 предложения, без вступлений.
        """,

        "citation_focus": """
        Ты — робот Валера, экскурсовод РТУ МИРЭА, отвечаешь вслух и коротко.

        Дай ПРЯМОЙ ответ на вопрос: нужное имя, число или факт — в первом предложении.
        Не упоминай источники, контекст, документы и не вставляй пометки вида
        [Source 1] — это служебные данные, которые вслух не звучат.
        Бери только те имена, числа и адреса, которые реально есть в предоставленной
        информации; ничего не достраивай по смыслу и не придумывай.
        Если факта нет — коротко скажи «Этого я не знаю».

        Максимум 40 слов, 1–2 предложения, без вступлений.
        """
    }
    
    # Query intent analysis templates
    INTENT_ANALYSIS = {
        "standard": """
        Before addressing this query, analyze its structure and intent:

        Query: {query}

        1. What type of information is being requested? (factual, explanation, comparison, etc.)
        2. What are the key entities or concepts mentioned?
        3. What assumptions are implied in this query?
        4. What is the likely goal of the person asking this question?

        Use this analysis to inform how you approach answering the query.
        """
    }
    
    # Query decomposition templates
    DECOMPOSITION = {
        "web_search": """
        {role}
        
        Original Query: {query}
        
        Break this query down into 2-4 specific search queries that together will help answer the original question completely.
        Each search query should:
        1. Be phrased to maximize relevant search engine results
        2. Focus on a particular aspect of the original query
        3. Use search engine-friendly syntax (short, precise terms without unnecessary words)
        4. Avoid complex language that would reduce search effectiveness
        
        {format_instructions}
        """,
        
        "document_search": """
        {role}
        
        Original Query: {query}
        
        Break this query down into 2-4 specific, focused subqueries that together will help answer the original question completely.
        Each subquery should:
        1. Be self-contained and specific
        2. Focus on a particular aspect of the original query
        3. Be phrased as a complete question
        
        {format_instructions}
        """,
        
        "informed_decomposition": """
        {role}
        
        Original Query: {query}
        
        I've already done an initial {search_type} and found some information, but we need to explore further:
        
        Initial Results: 
        {context}
        
        Based on what we've found so far, identify 2-3 specific follow-up {query_type} that would help us:
        1. Fill in important missing information not covered in the initial search
        2. Explore specific aspects of the query that weren't fully addressed
        3. Resolve any ambiguities or contradictions in the initial results
        
        Each query should:
        - Be focused on gathering new information not already covered
        - NOT duplicate information we already have from the initial search
        {specific_instructions}
        
        {format_instructions}
        """
    }
    
    # Answer synthesis templates
    SYNTHESIS = {
        "standard": """
        {role}
        
        Original Query: {query}
        
        I've broken this query down and found results for each part:
        
        {results}
        
        Synthesize a comprehensive, cohesive answer to the original query that:
        1. Directly addresses the original question
        2. Integrates information from all results
        3. Presents a logical flow of information
        4. Avoids unnecessary repetition
        5. Maintains factual accuracy from the source information
        6. {citation_instruction}
        7. Notes any conflicting information found and provides a balanced perspective
        
        Your answer should be thorough but concise, well-structured, and directly useful to the person who asked the original query.
        {limitation_instruction}
        """
    }
    
    # Web content analysis templates
    WEB_ANALYSIS = {
        "content_analysis": """
        {role}
        
        Please analyze the following web page content and extract key information relevant to: "{query}"
        
        Web Page: {title}
        URL: {url}
        
        Content:
        {content}
        
        Provide:
        1. A concise summary of the content (2-3 sentences)
        2. 3-5 key facts or points that are most relevant to the query
        3. An assessment of how well this content answers the query (high/medium/low)
        
        Format your response with clear sections. Only include information present in the content.
        """
    }
    
    # Title generation templates
    TITLE_GENERATION = {
        "chat_title": """
        Generate a very short title (3-5 words max) for a conversation that starts with this message:
        "{message}"
        
        Your response should ONLY include the short title, nothing else.
        """
    }
    
    # Specialized templates
    SPECIALIZED = {
        "subquery_analysis": """
        Based on the following web content analyses for the query: "{query}",
        please provide a concise, focused answer that specifically addresses this query.
        
        Analyzed web content:
        {sources_content}
        
        Your response should:
        1. Focus specifically on answering "{query}"
        2. Integrate information from all relevant sources
        3. Be concise but complete
        4. Cite sources with URLs where appropriate
        5. Indicate confidence level for your answer (high/medium/low)
        """
    }
    
    # Helper functions to format and retrieve prompts
    @staticmethod
    def get_role(role_type):
        """Get a role/persona definition"""
        return PromptTemplateService.AGENT_ROLES.get(role_type, PromptTemplateService.AGENT_ROLES["rag_assistant"])
    
    @staticmethod
    def get_format_instructions(format_type):
        """Get format instructions by type"""
        return PromptTemplateService.FORMAT_INSTRUCTIONS.get(format_type, "")
    
    @staticmethod
    def get_system_message(message_type):
        """Get a system message by type"""
        return PromptTemplateService.SYSTEM_MESSAGES.get(message_type, PromptTemplateService.SYSTEM_MESSAGES["rag"])
    
    @staticmethod
    def get_decomposition_prompt(query, type="document_search", **kwargs):
        """Get a decomposition prompt formatted with parameters"""
        template = PromptTemplateService.DECOMPOSITION.get(type)
        
        # Set default role if not provided
        if "role" not in kwargs:
            kwargs["role"] = PromptTemplateService.get_role("decomposer")
            
        # Set default format instructions if not provided
        if "format_instructions" not in kwargs:
            kwargs["format_instructions"] = PromptTemplateService.get_format_instructions("bullet_list_queries")
            
        return template.format(query=query, **kwargs)
    
    @staticmethod
    def get_synthesis_prompt(query, results, is_web_search=False, **kwargs):
        """Get a synthesis prompt formatted with parameters"""
        template = PromptTemplateService.SYNTHESIS.get("standard")
        
        # Set default role if not provided
        if "role" not in kwargs:
            kwargs["role"] = PromptTemplateService.get_role("synthesizer")
            
        # Set citation instruction based on search type
        if is_web_search:
            kwargs["citation_instruction"] = "Includes proper citations to web sources including URLs in parentheses"
            kwargs["limitation_instruction"] = "Be careful to only include factual information from the search results, and acknowledge any significant information gaps."
        else:
            kwargs["citation_instruction"] = "Отвечай сразу фактом (имя, число, адрес) — без ссылок на документы и без пометок вида [Источник 1]"
            kwargs["limitation_instruction"] = "Если нужного факта нет в данных, честно скажи «Этого я не знаю». Ничего не придумывай — особенно имена, числа и адреса."
            
        return template.format(query=query, results=results, **kwargs)
    
    @staticmethod
    def format_context(contexts, max_length=2200):
        """
        Собирает контекст для промпта, укладываясь в ``max_length`` символов.

        Ограничение небольшое СОЗНАТЕЛЬНО: в этом окружении VPN обрывает
        крупные HTTP-тела, поэтому промпт с контекстом должен быть коротким
        (клиент дополнительно страхует размер — см. OllamaService._fit_body).
        Для голосового робота, который отвечает в 1–2 предложения, верхнего
        самого релевантного фрагмента достаточно.

        Важное отличие от прошлой версии: если самый релевантный фрагмент один
        и он больше лимита, он РАНЬШЕ ПРОСТО ВЫБРАСЫВАЛСЯ (в контекст попадала
        одна шапка), и модель отвечала «нет данных». Теперь он обрезается.
        """
        formatted = "Context information:\n\n"
        sorted_contexts = sorted(contexts, key=lambda x: x.get('score', 0), reverse=True)

        current_length = len(formatted)
        for i, ctx in enumerate(sorted_contexts):
            title = ctx.get('document_title', 'Unknown')
            score = ctx.get('score', 0.0)
            content = ctx.get('content', '') or ''
            url = ctx.get('url', '')

            source_info = f"URL: {url}\n" if url else ""
            header = f"[Source {i+1}] From '{title}' (relevance: {score:.2f}):\n{source_info}"

            room = max_length - current_length
            if room <= len(header) + 40:
                # Больше не влезает ничего осмысленного.
                if i > 0:
                    formatted += "...(additional context truncated due to length)..."
                break

            # Обрезаем содержимое так, чтобы блок целиком вошёл в остаток.
            if len(content) > room - len(header):
                content = content[: max(0, room - len(header))].rstrip()
                content += " …"

            formatted += f"{header}{content}\n\n"
            current_length += len(header) + len(content) + 2

        return formatted