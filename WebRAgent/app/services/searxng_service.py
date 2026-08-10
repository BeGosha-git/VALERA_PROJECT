import os
import json
import logging
import time
import requests
from urllib.parse import urlencode
from app.services.llm_service import LLMFactory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Все веб-движки, которые могут работать. При запуске мы пингуем каждый
# и оставляем только отвечающие (см. _ping_engines). Блокированные (google,
# brave, duckduckgo, qwant, startpage...) не исключаем навсегда — если они
# снова станут доступны, они автоматически включатся.
ALL_WEB_ENGINES = [
    "google", "brave", "duckduckgo", "qwant", "startpage",
    "seznam", "mojeek", "mullvad", "wikipedia", "github",
    "yandex", "presearch", "stract",
    "mwmbl", "bing", "marginalia", "gigablast",
]

# Движки, которые дают нерелевантный мусор для русского поиска (чешские форумы
# у seznam, YouTube/несортированный мусор у presearch/mullvad). Их НЕ включаем
# в активные — они только замедляют поиск и портят качество ответов.
JUNK_ENGINES = {"seznam", "mojeek", "mullvad", "presearch", "stract", "marginalia", "gigablast", "yandex"}


class SearXNGService:
    """
    Service for performing web searches using SearXNG
    """

    # Кэш рабочих движков на уровне КЛАССА: SearXNGService создаётся заново
    # на каждый HTTP-запрос, поэтому инстанс-кэш не помогает. Классовый кэш
    # переживает запросы и делает пинг только один раз на процесс.
    _active_engines_cache = None
    _active_engines_timestamp = 0.0
    _ENGINES_CACHE_TTL = 600  # 10 минут

    def __init__(self):
        """Initialize SearXNG service with configuration from environment variables"""
        self.base_url = os.environ.get('SEARXNG_URL', 'http://searxng:8080')
        self.search_url = f"{self.base_url}/search"
        self.api_url = f"{self.base_url}/search"  # SearXNG API endpoint for JSON results
        self.results_per_page = int(os.environ.get('SEARXNG_RESULTS_PER_PAGE', 10))
        
        logger.info(f"Initialized SearXNGService with URL: {self.base_url}")

    # ------------------------------------------------------------------
    # Динамический пинг движков
    # ------------------------------------------------------------------
    def _ping_engines(self, test_query="тест", ping_timeout=3, max_engines=8):
        """
        Пингует каждый веб-движок коротким запросом и возвращает список
        тех, что реально ответили результатами. Блокированные/недоступные
        движки пропускаются, чтобы не ждать их таймауты в каждом запросе.

        Пинг выполняется ПАРАЛЛЕЛЬНО (по пулу потоков), чтобы не ждать
        последовательно таймауты заблокированных движков.
        """
        # Приоритет: сначала релевантные для русского поиска движки (mwmbl, bing),
        # чтобы они гарантированно попали в топ-max_engines. Остальные — после.
        # Пингуем НЕ все движки: заблокированные (google, brave, duckduckgo...)
        # известны заранее и только зря грузят SearXNG параллельными запросами.
        # Приоритетные всегда проверяем; из остальных берём небольшой набор.
        # Мусорные движки (JUNK_ENGINES) не пингуем и не включаем — они дают
        # нерелевантные результаты и замедляют поиск.
        priority = [e for e in ("mwmbl", "bing", "wikipedia", "github", "seznam", "mojeek")
                    if e not in JUNK_ENGINES]
        secondary = [e for e in ALL_WEB_ENGINES if e not in priority and e not in JUNK_ENGINES]
        # Проверяем приоритетные + максимум 3 из остальных (которые могли стать
        # доступными), чтобы не перегружать SearXNG фоновым пингом.
        ordered = priority + secondary[:3]

        def _ping_one(engine):
            try:
                resp = requests.get(
                    self.api_url,
                    params={
                        'q': test_query,
                        'format': 'json',
                        'engines': engine,
                        'results': 1,
                    },
                    headers={'Accept': 'application/json',
                             'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0'},
                    timeout=ping_timeout,
                )
                if resp.status_code != 200:
                    logger.info(f"  [PING] {engine}: статус {resp.status_code} — пропуск")
                    return engine, False
                data = resp.json()
                n = len(data.get('results', []))
                logger.info(f"  [PING] {engine}: {n} результатов — {'✅ рабочий' if n > 0 else '⏭️ пропуск'}")
                return engine, bool(data.get('results'))
            except Exception as e:
                logger.info(f"  [PING] {engine}: ошибка ({e}) — пропуск")
                return engine, False

        active = []
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=min(len(ordered), 6)) as pool:
            futures = {pool.submit(_ping_one, e): e for e in ordered}
            for fut in as_completed(futures):
                engine, ok = fut.result()
                if ok:
                    active.append(engine)

        # Восстанавливаем приоритетный порядок рабочих движков
        active.sort(key=lambda e: ordered.index(e) if e in ordered else len(ordered))

        if not active:
            # Ничего не ответило — фолбэк на проверенные релевантные движки
            active = ["mwmbl", "bing", "wikipedia", "github"]
        logger.info(f"✅ [PING] Рабочие движки после пинга: {active}")
        return active[:max_engines]

    def _get_active_engines(self):
        """
        Возвращает рабочие движки (кэш на уровне класса, TTL 10 минут).
        Если кэш пуст (например, новый процесс при Flask reloader) — сразу
        используем проверенный фолбэк без долгого пинга, а полный пинг
        запускаем в фоне, чтобы не блокировать первый запрос.
        """
        now = time.time()
        cached = SearXNGService._active_engines_cache
        cache_fresh = cached is not None and (now - SearXNGService._active_engines_timestamp <= self._ENGINES_CACHE_TTL)

        if cache_fresh:
            # Отфильтровываем мусорные движки на случай, если кэш старый
            return [e for e in cached if e not in JUNK_ENGINES] or ["mwmbl", "bing", "wikipedia", "github"]

        # Кэш пуст/устарел: для быстрого ответа используем надёжный фолбэк,
        # а полный пинг выполняем в фоне (результат применится со следующего запроса).
        # Фолбэк = релевантные движки для русского поиска (НЕ seznam/mullvad,
        # которые возвращают мусор: чешские форумы, YouTube и т.п.).
        fallback = ["mwmbl", "bing", "wikipedia", "github"]

        def _background_ping():
            try:
                SearXNGService._active_engines_cache = self._ping_engines()
                SearXNGService._active_engines_timestamp = time.time()
            except Exception:
                pass

        import threading
        t = threading.Thread(target=_background_ping, daemon=True)
        t.start()

        # Кэшируем фолбэк, чтобы последующие запросы не запускали пинг повторно
        SearXNGService._active_engines_cache = fallback
        SearXNGService._active_engines_timestamp = now
        return fallback
    
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
            # Определяем рабочие движки (пинг с кэшированием)
            active_engines = self._get_active_engines()
            if not active_engines:
                logger.warning("No active SearXNG engines, returning empty")
                return []

            # Делим движки на две группы для ПАРАЛЛЕЛЬНОГО поиска.
            # Внутри SearXNG движки выполняются последовательно, поэтому один
            # запрос со всеми движками = сумма их таймаутов (bing ~2.1с + mwmbl
            # 0.6с + ...). Параллельные запросы по группам сокращают время до
            # максимума по группам (fast ~1с, slow ~2.1с) вместо суммы.
            fast_engines = [e for e in active_engines if e in ("mwmbl", "wikipedia", "github")]
            slow_engines = [e for e in active_engines if e not in fast_engines]

            headers = {
                'Accept': 'application/json',
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'
            }

            def _query_engines(engines):
                if not engines:
                    return []
                params = {
                    'q': query,
                    'format': 'json',
                    # НЕ передаём 'categories': он перекрывает 'engines' и заставляет
                    # SearXNG вызывать ВСЕ движки категории (включая заблокированные),
                    # из-за чего запрос ждёт их таймауты. 'engines' уже отфильтрован
                    # пингом до рабочих, поэтому категорию не задаём.
                    'results': min(num_results, 25),  # Prevent overly large requests
                    'language': 'all',
                    'engines': ','.join(engines),
                    'timeout': '8',  # hard cap so we don't wait for slow engines
                }
                import time as _t
                _t0 = _t.time()
                try:
                    resp = requests.get(self.api_url, params=params, headers=headers, timeout=20)
                    _dt = _t.time() - _t0
                    logger.info(f"⏱️ [SEARXNG] '{query[:40]}' | {engines} → {_dt:.2f}с (статус {resp.status_code})")
                    if resp.status_code != 200:
                        return []
                    try:
                        data = resp.json()
                    except json.JSONDecodeError:
                        logger.error("❌ [SEARXNG] Не удалось распарсить JSON ответ")
                        return []
                    unresponsive = data.get('unresponsive_engines', [])
                    if unresponsive:
                        logger.warning(f"⚠️ [SEARXNG] Неответившие движки: {unresponsive}")
                    return self._format_results(data)
                except Exception as e:
                    logger.error(f"❌ [SEARXNG] Ошибка группы {engines}: {e}")
                    return []

            logger.info(f"🔍 [SEARXNG] Запрос: '{query}' | движки: {active_engines} | лимит: {min(num_results, 25)}")

            # Параллельные запросы по группам (максимум 2 группы — быстрые + остальные)
            import time as _t
            from concurrent.futures import ThreadPoolExecutor, as_completed
            _t0 = _t.time()
            groups = [fast_engines, slow_engines]
            results = []
            with ThreadPoolExecutor(max_workers=min(len(groups), 2)) as pool:
                futures = [pool.submit(_query_engines, g) for g in groups if g]
                for fut in as_completed(futures):
                    results.extend(fut.result())

            _dt = _t.time() - _t0
            logger.info(f"⏱️ [SEARXNG] Суммарно за {_dt:.2f}с")

            # Дедупликация по URL (один сайт может прийти от нескольких движков)
            seen = set()
            deduped = []
            for r in results:
                url = r.get('url', '')
                if url and url in seen:
                    continue
                if url:
                    seen.add(url)
                deduped.append(r)

            logger.info(f"✅ [SEARXNG] Найдено {len(deduped)} результатов по запросу '{query}'")
            for i, r in enumerate(deduped[:5], 1):
                logger.info(f"   [{i}] {r.get('title', '')[:60]} | {r.get('url', '')[:80]}")

            return deduped

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
        logger.info(f"[SEARXNG] process_query: '{query}' → {len(search_results)} результатов")
        
        # Convert web search results to contextual format expected by RAG interface.
        # LIMIT to top N relevant contexts for the LLM — if we pass 40+ results
        # (including junk from weak engines), the model drowns in noise and says
        # "cannot find". Top 5 relevant (already sorted by score) is enough.
        MAX_CONTEXTS_FOR_LLM = 5
        contexts = []
        for result in search_results[:MAX_CONTEXTS_FOR_LLM]:
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
        logger.info(f"[SEARXNG] Контекст для LLM ({len(contexts)} источников), первые 300 символов:\n{formatted_context[:300]}")
        
        if conversation_context and len(conversation_context) > 0:
            # Process with conversation context if available
            current_exchange = list(conversation_context)  # Copy to avoid modifying original
            has_system = any(msg['role'] == 'system' for msg in current_exchange)

            # Dedupe a trailing user message equal to the current query (the chat route
            # already added it before building context), so the fresh question appears
            # exactly once and keeps its weight as the newest message.
            while current_exchange and current_exchange[-1].get('role') == 'user':
                if current_exchange[-1].get('content', '').strip() == query.strip():
                    current_exchange.pop()
                    break
                break

            if not has_system:
                current_exchange.insert(0, {
                    'role': 'system',
                    'content': "You are a helpful assistant answering questions based on web search results."
                })

            # Add the current query (single, as the last message)
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
        
        logger.info(f"[SEARXNG] LLM-ответ ({len(response)} символов): {response[:200]}")
        
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
        
        # Движки, которые дают РЕЛЕВАНТНЫЕ русские результаты (mwmbl, bing).
        # Их результаты получают повышенный score, чтобы гарантированно попасть
        # в топ-5 контекста для LLM. Остальные (wikipedia, seznam, github...)
        # часто дают мусор, который вытесняет полезное.
        PRIORITY_ENGINES = {"mwmbl", "bing", "presearch"}

        # Check if results exist and extract them
        if 'results' not in search_results:
            return formatted
        
        for result in search_results['results']:
            # Skip results without URLs or titles
            if 'url' not in result or 'title' not in result:
                continue
                
            engine = result.get('engine', 'web')
            score = float(result.get('score', 0.95))
            # Приоритетные движки — сильный буст score (0.95 → 1.0)
            if engine in PRIORITY_ENGINES:
                score = max(score, 1.0)

            formatted_result = {
                'title': result.get('title', ''),
                'url': result.get('url', ''),
                'snippet': result.get('content', ''),
                'source': engine,
                'score': score
            }
            
            formatted.append(formatted_result)

        # Deduplicate by URL (keep the highest-scoring entry per URL).
        # SearXNG can return the same page from multiple engines.
        by_url = {}
        for item in formatted:
            url = item['url']
            if url not in by_url or item['score'] > by_url[url]['score']:
                by_url[url] = item
        formatted = list(by_url.values())

        # Сортировка: сначала ВСЕ результаты приоритетных движков (mwmbl, bing,
        # presearch), затем остальные. Внутри каждой группы — по score убыванию.
        # Это гарантирует, что релевантные русские результаты всегда попадают
        # в топ-5 контекста для LLM, а мусор (wikipedia, boursorama...) — в конец.
        def _sort_key(item):
            return (1 if item.get('source') not in PRIORITY_ENGINES else 0,
                    -item.get('score', 0.0))
        formatted.sort(key=_sort_key)
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