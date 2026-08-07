import os
import mimetypes
from flask import Blueprint, render_template, request, jsonify, send_file, abort, url_for, Response, redirect
from flask_login import login_required, current_user
from app.models.collection import Collection
from app.models.document import Document
from app.services.rag_service import RAGService
from app.services.agent_search_service import AgentSearchService
from app.services.mindmap_service import MindmapService

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """Main page with query interface - redirects to login if not authenticated"""
    # If user is not logged in, redirect to login page
    if not current_user.is_authenticated:
        return redirect(url_for('auth.login'))
        
    # Get all collections for dropdown
    collections = Collection.get_all()
    return render_template('main/index.html', collections=collections)

@main_bp.route('/query', methods=['POST'])
@login_required
def query():
    """Process user query and return RAG results"""
    collection_id = request.form.get('collection_id')
    query_text = request.form.get('query')
    
    # Check which search options are enabled
    use_agent_search = request.form.get('use_agent_search') == 'on'
    use_web_search = request.form.get('use_web_search') == 'on'
    use_deep_search = request.form.get('use_deep_search') == 'on'
    agent_strategy = request.form.get('agent_strategy', 'direct')
    
    # Get max_results parameter, default to 4 if not provided
    try:
        max_results = int(request.form.get('max_results', 4))
        # Limit to reasonable range
        max_results = min(max(max_results, 1), 10)
    except ValueError:
        max_results = 4
    
    # Validate inputs based on search type
    if not query_text:
        return jsonify({
            'error': 'Missing query'
        }), 400
    
    # For non-web search, we need a collection ID
    if not use_web_search and not collection_id:
        return jsonify({
            'error': 'Missing collection ID (required for document search)'
        }), 400
    
    # Handle different search combinations
    if use_web_search:
        # Import web search related services
        from app.services.searxng_service import SearXNGService
        from app.services.web_search_agent_service import WebSearchAgentService
        
        if use_deep_search:
            # Deep web search (scrape results)
            from app.services.deep_web_search_service import DeepWebSearchService
            deep_search_service = DeepWebSearchService()
            
            if use_agent_search:
                # Deep web search with agent capabilities
                result = deep_search_service.process_query(
                    query=query_text,
                    max_results=max_results,
                    search_type='general',
                    conversation_context=None,  # Could be enhanced with conversation history
                    use_agent=True,
                    agent_strategy=agent_strategy
                )
            else:
                # Standard deep web search
                result = deep_search_service.process_query(
                    query=query_text,
                    max_results=max_results,
                    search_type='general',
                    conversation_context=None  # Could be enhanced with conversation history
                )
        elif use_agent_search:
            # Web search with agent capabilities
            web_agent_service = WebSearchAgentService()
            result = web_agent_service.process_query(
                query=query_text,
                max_results=max_results,
                strategy=agent_strategy
            )
        else:
            # Standard web search
            web_search_service = SearXNGService()
            result = web_search_service.process_query(
                query=query_text,
                max_results=max_results
            )
    else:
        # Document-based searches
        if use_agent_search:
            # Process query with Agent Search and specified strategy
            agent_service = AgentSearchService()
            result = agent_service.process_query(
                collection_id=collection_id,
                query=query_text,
                max_results=max_results,
                strategy=agent_strategy
            )
        else:
            # Process query with standard RAG
            rag_service = RAGService()
            result = rag_service.process_query(
                collection_id=collection_id,
                query=query_text,
                max_results=max_results
            )
    
    # Return JSON if AJAX request, otherwise render template
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # Process the response with markdown for JSON
        import markdown
        # Create a copy to avoid modifying the original
        json_result = dict(result)
        if 'response' in json_result:
            json_result['response'] = markdown.markdown(json_result['response'])
        return jsonify(json_result)
    else:
        collections = Collection.get_all()
        return render_template(
            'main/index.html',
            collections=collections,
            result=result,
            selected_collection=collection_id
        )

@main_bp.route('/collections')
@login_required
def list_collections():
    """List available collections"""
    collections = Collection.get_all()
    return jsonify([c.to_dict() for c in collections])

@main_bp.route('/document/<document_id>')
@login_required
def view_document(document_id):
    """View document preview"""
    document = Document.get(document_id)
    if not document or not document.file_path or not os.path.exists(document.file_path):
        abort(404)
    
    return render_template('main/document_preview.html', document=document)

@main_bp.route('/document/<document_id>/raw')
@login_required
def document_raw(document_id):
    """Serve raw document file"""
    document = Document.get(document_id)
    if not document or not document.file_path or not os.path.exists(document.file_path):
        abort(404)
    
    # Get the MIME type to serve the file correctly
    mime_type, _ = mimetypes.guess_type(document.file_path)
    if not mime_type:
        mime_type = 'application/octet-stream'
    
    # Return the file as an attachment
    return send_file(document.file_path, 
                    mimetype=mime_type,
                    as_attachment=True,
                    download_name=os.path.basename(document.file_path))

@main_bp.route('/document/<document_id>/preview')
@login_required
def document_preview(document_id):
    """Serve document preview"""
    document = Document.get(document_id)
    if not document or not document.file_path or not os.path.exists(document.file_path):
        abort(404)
    
    # Get the MIME type and file extension
    mime_type, _ = mimetypes.guess_type(document.file_path)
    file_extension = os.path.splitext(document.file_path)[1].lower()
    
    # PDF documents - direct preview
    if mime_type == 'application/pdf' or file_extension == '.pdf':
        return send_file(document.file_path, mimetype='application/pdf')
    
    # Images - direct preview
    if mime_type and mime_type.startswith('image/'):
        return send_file(document.file_path, mimetype=mime_type)
    
    # HTML documents - direct preview
    if mime_type == 'text/html' or file_extension in ['.html', '.htm']:
        with open(document.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return Response(content, mimetype='text/html')
    
    # Text-based documents - simple preview
    if mime_type and (mime_type.startswith('text/') or file_extension in ['.txt', '.md', '.csv']):
        with open(document.file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Format markdown for better display
        if file_extension == '.md':
            import markdown
            content = markdown.markdown(content)
            return Response(f'<div style="padding: 20px;">{content}</div>', mimetype='text/html')
        
        # Format CSV for better display
        if file_extension == '.csv':
            try:
                import pandas as pd
                import io
                df = pd.read_csv(document.file_path)
                html_table = df.to_html(classes='table table-striped table-bordered', index=False)
                return Response(f'<div style="padding: 20px;"><style>body {{ font-family: Arial; }}</style>{html_table}</div>', 
                               mimetype='text/html')
            except Exception:
                # Fall back to plain text if pandas fails
                return Response(f'<pre>{content}</pre>', mimetype='text/html')
                
        # Plain text with pre tag for proper formatting
        return Response(f'<pre style="padding: 20px; white-space: pre-wrap;">{content}</pre>', mimetype='text/html')
    
    # Office documents (docx, xlsx, pptx) - extract preview using docling
    if file_extension in ['.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt']:
        try:
            from app.services.document_service import DocumentService
            document_service = DocumentService()
            extracted_text = document_service.extract_text(document.file_path)
            
            # Convert to HTML for better rendering
            formatted_html = f'''
            <div style="padding: 20px; font-family: Arial, sans-serif;">
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 15px;">
                    <p><strong>Document Preview:</strong> {document.title}</p>
                    <p><small>This is an extracted text preview. <a href="{url_for('main.document_raw', document_id=document.id)}">Download</a> the original file for proper formatting.</small></p>
                </div>
                <div style="white-space: pre-wrap;">{extracted_text}</div>
            </div>
            '''
            return Response(formatted_html, mimetype='text/html')
        except Exception as e:
            # If text extraction fails, fall back to the download page
            return render_template('main/document_preview_fallback.html', document=document, error=str(e))
    
    # Default: just offer download
    return render_template('main/document_preview_fallback.html', document=document)

@main_bp.route('/mindmap', methods=['POST'])
@login_required
def generate_mindmap():
    """Generate a mindmap for the query and context"""
    # Get the query and context data
    query = request.form.get('query')
    context = request.form.get('context')
    
    if not query:
        return jsonify({'error': 'Missing query'}), 400
    
    # Generate mindmap
    mindmap_service = MindmapService()
    html_content = mindmap_service.generate_mindmap(query, context)
    
    # Return the HTML content
    return Response(html_content, mimetype='text/html')

@main_bp.route('/about')
def about():
    """About page"""
    return render_template('main/about.html')