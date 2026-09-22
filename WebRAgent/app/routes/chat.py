from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from flask_login import login_required, current_user
from app.models.chat import Chat
from app.models.collection import Collection
from app.services.rag_service import RAGService
from app.services.agent_search_service import AgentSearchService
from app.services.searxng_service import SearXNGService
from app.services.web_search_agent_service import WebSearchAgentService
from app.services.llm_service import LLMFactory

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
    
    # Format the query with context
    formatted_query = query_text
    if chat_context and len(chat_context) > 0:
        # Create a system message to inform the LLM about previous context
        system_context = """This is a follow-up question in an ongoing conversation. 
        Consider the following conversation history when answering the new query."""
        
        # Add system instruction to the context for better results
        if not any(msg['role'] == 'system' for msg in chat_context):
            chat_context.insert(0, {
                'role': 'system',
                'content': system_context
            })
    
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
        
        # If this is the first exchange, generate a title for the chat
        if chat.message_count <= 2:
            llm_service = LLMFactory.create_llm_service()
            chat.generate_title(llm_service)
            
        # Process the response
        import markdown
        if 'response' in result:
            result['formatted_response'] = markdown.markdown(result['response'])
        
        # Return the result with additional chat info
        result['chat'] = chat.to_dict()
        
        return jsonify(result)
        
    except Exception as e:
        # Log the error
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error processing chat query: {str(e)}")
        
        # Return an error response
        return jsonify({
            'error': 'Failed to process query',
            'message': str(e)
        }), 500

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