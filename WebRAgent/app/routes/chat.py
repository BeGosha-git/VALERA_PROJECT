from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from flask_login import login_required, current_user
import logging

from app.models.chat import Chat
from app.models.collection import Collection
from app.services.rag_service import RAGService
from app.services.agent_search_service import AgentSearchService
from app.services.searxng_service import SearXNGService
from app.services.web_search_agent_service import WebSearchAgentService
from app.services.llm_service import LLMFactory
from app.utils import slim_result, estimate_size

# Логгер на уровне модуля. Объявлять его внутри функции нельзя: ниже по коду
# он же присваивался в блоке except, из-за чего Python считал имя локальным —
# обращение к нему до присваивания падало с UnboundLocalError.
logger = logging.getLogger(__name__)

chat_bp = Blueprint('chat', __name__, url_prefix='/chat')

@chat_bp.route('/')
@login_required
def index():
    """Chat interface listing recent chats"""
    # Get all collections for dropdown
    collections = Collection.get_all()
    
    # Get recent chats for current user
    recent_chats = Chat.get_recent(limit=10, user_id=current_user.id)
    
    return render_template(
        'chat/index.html', 
        collections=collections, 
        recent_chats=recent_chats
    )

@chat_bp.route('/new', methods=['POST'])
@login_required
def new_chat():
    """Create a new chat"""
    # Create a new chat with the current user's ID
    chat = Chat.create(user_id=current_user.id)
    if not chat:
        return jsonify({'error': 'Failed to create chat'}), 500
    
    # Return the chat ID
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'chat_id': chat.chat_id})
    else:
        return redirect(url_for('chat.view_chat', chat_id=chat.chat_id))

@chat_bp.route('/<chat_id>')
@login_required
def view_chat(chat_id):
    """View a specific chat"""
    chat = Chat.get(chat_id)
    # Verify the chat exists and belongs to the current user
    if not chat or chat.user_id != current_user.id:
        flash('Chat not found or you do not have permission to view it', 'error')
        return redirect(url_for('chat.index'))
    
    # Get all collections for dropdown
    collections = Collection.get_all()
    
    # Get chat messages
    messages = chat.get_messages()
    
    return render_template(
        'chat/view.html', 
        chat=chat, 
        messages=messages,
        collections=collections
    )

@chat_bp.route('/<chat_id>/query', methods=['POST'])
@login_required
def chat_query(chat_id):
    """Process a query in the context of a chat"""
    chat = Chat.get(chat_id)
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
    
    # Verify the chat belongs to the current user
    if chat.user_id != current_user.id:
        return jsonify({'error': 'You do not have permission to access this chat'}), 403
    
    # Get query parameters
    collection_id = request.form.get('collection_id')
    query_text = request.form.get('query')
    
    # Check which search options are enabled
    use_agent_search = request.form.get('use_agent_search') == 'on'
    use_web_search = request.form.get('use_web_search') == 'on'
    agent_strategy = request.form.get('agent_strategy', 'direct')
    
    # Get max_results parameter
    try:
        max_results = int(request.form.get('max_results', 4))
        max_results = min(max(max_results, 1), 10)
    except ValueError:
        max_results = 4
    
    # Validate inputs based on search type
    if not query_text:
        return jsonify({'error': 'Missing query'}), 400
    
    # For non-web search, we need a collection ID
    if not use_web_search and not collection_id:
        return jsonify({'error': 'Missing collection ID (required for document search)'}), 400
    
    # Add the user message to the chat
    chat.add_message('user', query_text)
    
    # Get previous chat context
    chat_context = chat.get_context()

    # Системный промт здесь НЕ добавляем.
    #
    # Раньше тут вставлялось английское «This is a follow-up question in an
    # ongoing conversation…», и RAGService видел «системное сообщение уже есть»
    # → промт Валеры не подставлялся ВООБЩЕ. DeepSeek работал без системного
    # промта: рассуждал о смысле фраз, путал имена, не знал, что он робот.
    # Теперь единственный источник системного промта — RAGService.
    formatted_query = query_text
    
    # Process query with appropriate service based on options
    try:
        if use_web_search:
            if use_agent_search:
                # Web search with agent capabilities
                web_agent_service = WebSearchAgentService()
                result = web_agent_service.process_query(
                    query=query_text,
                    max_results=max_results,
                    strategy=agent_strategy,
                    conversation_context=chat_context
                )
            else:
                # Standard web search
                web_search_service = SearXNGService()
                result = web_search_service.process_query(
                    query=query_text,
                    max_results=max_results,
                    conversation_context=chat_context
                )
        else:
            # Document-based searches
            if use_agent_search:
                # Process query with Agent Search
                agent_service = AgentSearchService()
                result = agent_service.process_query(
                    collection_id=collection_id,
                    query=query_text,
                    max_results=max_results,
                    strategy=agent_strategy,
                    conversation_context=chat_context
                )
            else:
                # Process query with standard RAG
                rag_service = RAGService()
                result = rag_service.process_query(
                    collection_id=collection_id,
                    query=query_text,
                    max_results=max_results,
                    conversation_context=chat_context
                )
                
        # Add the assistant's response to the chat with metadata about the search
        assistant_response = result['response']
        metadata = {
            'search_type': 'web' if use_web_search else 'document',
            'agent_search': use_agent_search,
            'agent_strategy': agent_strategy if use_agent_search else None,
            'collection_id': collection_id if not use_web_search else None
        }
        
        chat.add_message('assistant', assistant_response, metadata)
        
        # Первый обмен — генерируем название чата. Раньше это был ЕЩЁ ОДИН синхронный вызов LLM
        # перед ответом (гость ждал лишние секунды); теперь — в фоне.
        if chat.message_count <= 2:
            import threading
            threading.Thread(
                target=lambda: chat.generate_title(LLMFactory.create_llm_service()), daemon=True
            ).start()
            
        # Process the response
        import markdown
        if 'response' in result:
            result['formatted_response'] = markdown.markdown(result['response'])

        # Return the result with additional chat info
        result['chat'] = chat.to_dict()

        # Уменьшаем ответ: в этом окружении VPN обрывает HTTP-ответы крупнее
        # ~16 КБ, и клиент получал 0 байт вместо данных ("Ошибка соединения").
        # Полные тексты чанков клиенту не нужны — достаточно превью.
        payload = slim_result(result)
        logger.info(
            "[CHAT] ответ %d байт, контекстов %d",
            estimate_size(payload), len(payload.get('contexts') or []),
        )

        return jsonify(payload)
        
    except Exception as e:
        # Log the error
        logger.error(f"Error processing chat query: {str(e)}")
        
        # Return an error response
        return jsonify({
            'error': 'Failed to process query',
            'message': str(e)
        }), 500

@chat_bp.route('/<chat_id>/query_stream', methods=['POST'])
@login_required
def query_chat_stream(chat_id):
    """
    Потоковый ответ по базе знаний (NDJSON): голосовой робот начинает говорить с первого
    предложения, пока модель ещё пишет остальное.

    Строки потока: {"t":"meta"} → {"t":"d","x":"…"}… → {"t":"done","response":"…"} (или {"t":"err"}).
    Работает только для поиска по коллекции; веб-поиск и агентный поиск — через обычный /query.
    """
    import json as _json
    import threading
    from flask import Response, stream_with_context

    chat = Chat.get(chat_id)
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
    if chat.user_id != current_user.id:
        return jsonify({'error': 'You do not have permission to access this chat'}), 403
    collection_id = request.form.get('collection_id')
    query_text = request.form.get('query')
    if not query_text or not collection_id:
        return jsonify({'error': 'Нужны query и collection_id'}), 400
    try:
        max_results = min(max(int(request.form.get('max_results', 4)), 1), 10)
    except ValueError:
        max_results = 4

    chat.add_message('user', query_text)
    chat_context = chat.get_context()

    def generate():
        full = ""
        try:
            rag_service = RAGService()
            for ev in rag_service.process_query_stream(
                collection_id=collection_id, query=query_text,
                max_results=max_results, conversation_context=chat_context,
            ):
                if ev["t"] == "done":
                    full = ev["response"]
                    ev = {"t": "done", "response": full}
                yield _json.dumps(ev, ensure_ascii=False) + "\n"
        except Exception as e:  # noqa: BLE001
            logger.error(f"Ошибка потокового запроса: {e}")
            yield _json.dumps({"t": "err", "message": str(e)}, ensure_ascii=False) + "\n"
            return
        if full:
            chat.add_message('assistant', full, {'search_type': 'document', 'collection_id': collection_id, 'stream': True})
            if chat.message_count <= 2:
                # Название чата генерирует LLM — отдельным потоком, чтобы не задерживать ответ.
                threading.Thread(
                    target=lambda: chat.generate_title(LLMFactory.create_llm_service()), daemon=True
                ).start()

    resp = Response(stream_with_context(generate()), mimetype='application/x-ndjson')
    resp.headers['Cache-Control'] = 'no-cache'
    resp.headers['X-Accel-Buffering'] = 'no'
    return resp


@chat_bp.route('/<chat_id>/messages')
@login_required
def get_messages(chat_id):
    """Get messages for a chat"""
    chat = Chat.get(chat_id)
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
        
    # Verify the chat belongs to the current user
    if chat.user_id != current_user.id:
        return jsonify({'error': 'You do not have permission to access this chat'}), 403
    
    # Get parameters
    limit = request.args.get('limit', 50, type=int)
    before_id = request.args.get('before_id')
    
    # Get messages
    messages = chat.get_messages(limit=limit, before_id=before_id)
    
    # Format message content with markdown
    import markdown
    for message in messages:
        if 'content' in message:
            message['formatted_content'] = markdown.markdown(message['content'])
    
    return jsonify({
        'chat': chat.to_dict(),
        'messages': messages
    })

@chat_bp.route('/<chat_id>/append', methods=['POST'])
@login_required
def append_messages(chat_id):
    """
    Append an already-produced question/answer pair to the chat.

    Used by external clients (e.g. the voice assistant) that answer some
    phrases WITHOUT going through the search services (fast conversational
    replies via a direct LLM call). Persisting those turns keeps the chat
    history complete, so the conversation context stays coherent for
    follow-up questions.

    Form fields:
        question (str): user message to store
        answer (str): assistant message to store
    """
    chat = Chat.get(chat_id)
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404

    if chat.user_id != current_user.id:
        return jsonify({'error': 'You do not have permission to access this chat'}), 403

    question = (request.form.get('question') or '').strip()
    answer = (request.form.get('answer') or '').strip()
    if not question or not answer:
        return jsonify({'error': 'Missing question or answer'}), 400

    chat.add_message('user', question)
    chat.add_message('assistant', answer, {'search_type': 'direct'})

    return jsonify({'success': True, 'chat': chat.to_dict()})

@chat_bp.route('/<chat_id>/title', methods=['POST'])
@login_required
def update_title(chat_id):
    """Update the title of a chat"""
    chat = Chat.get(chat_id)
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
        
    # Verify the chat belongs to the current user
    if chat.user_id != current_user.id:
        return jsonify({'error': 'You do not have permission to modify this chat'}), 403
    
    # Get new title
    new_title = request.form.get('title')
    if not new_title:
        return jsonify({'error': 'Missing title'}), 400
    
    # Update the title
    success = chat.update_title(new_title)
    if not success:
        return jsonify({'error': 'Failed to update title'}), 500
    
    return jsonify({'success': True, 'chat': chat.to_dict()})

@chat_bp.route('/<chat_id>/delete', methods=['POST'])
@login_required
def delete_chat(chat_id):
    """Delete a chat"""
    chat = Chat.get(chat_id)
    if not chat:
        return jsonify({'error': 'Chat not found'}), 404
        
    # Verify the chat belongs to the current user
    if chat.user_id != current_user.id:
        return jsonify({'error': 'You do not have permission to delete this chat'}), 403
    
    # Delete the chat
    success = chat.delete()
    if not success:
        return jsonify({'error': 'Failed to delete chat'}), 500
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'success': True})
    else:
        return redirect(url_for('chat.index'))