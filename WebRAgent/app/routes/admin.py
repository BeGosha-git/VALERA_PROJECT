import os
from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app.models.collection import Collection
from app.models.document import Document
from app.models.user import User
from app.services.document_service import DocumentService
from app.services.qdrant_service import QdrantService
from app.services.model_service import ModelService

admin_bp = Blueprint('admin', __name__)

def admin_required(f):
    """Decorator to require admin access"""
    @login_required
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin:
            flash('Admin access required', 'error')
            return redirect(url_for('main.index'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

@admin_bp.route('/')
@admin_required
def index():
    """Admin dashboard"""
    collections = Collection.get_all()
    users = User.get_all()
    return render_template('admin/index.html', collections=collections, users=users)

@admin_bp.route('/users', methods=['GET', 'POST'])
@admin_required
def users():
    """User management"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        is_admin = 'is_admin' in request.form
        
        if not username or not password:
            flash('Username and password are required', 'error')
            return redirect(url_for('admin.users'))
            
        # Create user
        user, message = User.create_user(
            username=username, 
            password=password,
            is_admin=is_admin
        )
        
        if user:
            flash(f'User "{username}" created successfully', 'success')
        else:
            flash(message, 'error')
            
        return redirect(url_for('admin.users'))
    
    users = User.get_all()
    return render_template('admin/users.html', users=users)

@admin_bp.route('/users/<user_id>/delete', methods=['POST'])
@admin_required
def delete_user(user_id):
    """Delete a user"""
    # Prevent deleting your own account
    if user_id == current_user.id:
        flash('You cannot delete your own account', 'error')
        return redirect(url_for('admin.users'))
        
    user = User.get(user_id)
    if not user:
        flash('User not found', 'error')
        return redirect(url_for('admin.users'))
        
    username = user.username
    success = User.delete(user_id)
    
    if success:
        flash(f'User "{username}" deleted successfully', 'success')
    else:
        flash(f'Failed to delete user "{username}"', 'error')
        
    return redirect(url_for('admin.users'))

@admin_bp.route('/collections', methods=['GET', 'POST'])
@admin_required
def collections():
    """Collection management"""
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        embedding_provider = request.form.get('embedding_provider')
        embedding_model = request.form.get('embedding_model')
        
        if not name:
            flash('Collection name is required', 'error')
            return redirect(url_for('admin.collections'))
        
        # We'll dynamically determine embedding dimensions in the QdrantService
        # using the actual API for the selected embedding model
        
        # Clean up the model name for sentence transformers to avoid issues with tags
        if embedding_provider == 'embedding' and embedding_model and ':' in embedding_model:
            embedding_model = embedding_model.split(':')[0]
        
        # Create collection with embedding information
        # Let's determine the actual dimensions by asking QdrantService
        qdrant = QdrantService()
        
        try:
            # Determine vector dimensions through API test
            actual_dimensions = qdrant.get_vector_size(embedding_provider, embedding_model)
            
            collection = Collection(
                name=name, 
                description=description,
                embedding_provider=embedding_provider,
                embedding_model=embedding_model,
                embedding_dimensions=actual_dimensions
            )
            collection.save()
            
            # Create collection in Qdrant with the specific embedding model
            success = qdrant.create_collection(
                collection_name=collection.id, 
                vector_size=actual_dimensions,
                provider=embedding_provider,
                model=embedding_model
            )
            
            if not success:
                # Collection creation failed
                flash(f'Failed to create vector database for collection "{name}"', 'error')
                # Remove the saved collection
                collection.delete()
                return redirect(url_for('admin.collections'))
                
        except ValueError as e:
            # Dimension detection failed
            error_message = str(e).split('\n')[0]
            flash(f'Could not determine embedding dimensions: {error_message}', 'error')
            return redirect(url_for('admin.collections'))
        except Exception as e:
            # Other errors
            flash(f'Error creating collection: {str(e)}', 'error')
            return redirect(url_for('admin.collections'))
        
        flash(f'Collection "{name}" created successfully', 'success')
        return redirect(url_for('admin.collections'))
    
    # Get available providers for embedding models
    model_service = ModelService()
    providers = model_service.get_available_providers()
    
    collections = Collection.get_all()
    return render_template('admin/collections.html', collections=collections, providers=providers)

@admin_bp.route('/collections/<collection_id>')
@admin_required
def collection_detail(collection_id):
    """View collection details"""
    collection = Collection.get(collection_id)
    if not collection:
        flash('Collection not found', 'error')
        return redirect(url_for('admin.collections'))
    
    documents = Document.get_all(collection_id=collection_id)
    return render_template('admin/collection_detail.html', collection=collection, documents=documents)

@admin_bp.route('/collections/<collection_id>/delete', methods=['POST'])
@admin_required
def delete_collection(collection_id):
    """Delete a collection"""
    collection = Collection.get(collection_id)
    if not collection:
        flash('Collection not found', 'error')
        return redirect(url_for('admin.collections'))
    
    # Delete collection from Qdrant
    qdrant = QdrantService()
    qdrant.delete_collection(collection_id)
    
    # Delete collection from storage
    collection.delete()
    
    # Delete associated documents
    documents = Document.get_all(collection_id=collection_id)
    for doc in documents:
        doc.delete()
    
    flash(f'Collection "{collection.name}" deleted successfully', 'success')
    return redirect(url_for('admin.collections'))

@admin_bp.route('/collections/<collection_id>/upload', methods=['GET', 'POST'])
@admin_required
def upload_document(collection_id):
    """Upload one or more documents to collection"""
    collection = Collection.get(collection_id)
    if not collection:
        flash('Collection not found', 'error')
        return redirect(url_for('admin.collections'))
    
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'documents' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)
        
        files = request.files.getlist('documents')
        base_title = request.form.get('title', '')
        
        # Check if at least one file was selected
        if not files or files[0].filename == '':
            flash('No selected files', 'error')
            return redirect(request.url)
        
        # Process each document
        document_service = DocumentService()
        processed_count = 0
        files_to_process = []
        
        # Get form parameters
        use_docling = 'use_docling' in request.form and request.form.get('use_docling') == 'on'
        
        # Get chunking parameters with defaults
        chunk_strategy = request.form.get('chunk_strategy', 'sentence')
        
        try:
            chunk_size = int(request.form.get('chunk_size', 1000))
            # Constrain to reasonable values
            chunk_size = max(100, min(chunk_size, 8000))
        except (ValueError, TypeError):
            chunk_size = 1000
            
        try:
            chunk_overlap = int(request.form.get('chunk_overlap', 200))
            # Constrain to reasonable values
            chunk_overlap = max(0, min(chunk_overlap, 1000))
        except (ValueError, TypeError):
            chunk_overlap = 200
            
        # Get metadata extraction level
        extract_metadata = request.form.get('extract_metadata', 'basic')
        
        # Preprocessing step: handle ZIP files and prepare a list of all files to process
        extracted_files = []
        for file in files:
            if file.filename == '':
                continue
                
            # Check if it's a ZIP file
            if file.filename.lower().endswith('.zip'):
                # Save the zip file temporarily
                zip_path = document_service.save_uploaded_file(file)
                
                # Extract and process each file in the ZIP
                zip_contents = document_service.process_zip_file(zip_path)
                extracted_files.extend(zip_contents)
                
                # Remove the original ZIP file once extracted (it's no longer needed)
                try:
                    os.remove(zip_path)
                except Exception as e:
                    logger.error(f"Error removing temporary ZIP file: {e}")
            else:
                # Regular file, add to the list to process
                files_to_process.append((None, file))  # None as path because it's not saved yet
        
        # Add extracted files to the processing list
        files_to_process.extend(extracted_files)
        
        # Now process all files (both directly uploaded and extracted from ZIP)
        for file_path, file in files_to_process:
            # Set title based on input or filename
            if base_title:
                # If we have multiple files and a base title, append the filename
                if len(files_to_process) > 1:
                    title = f"{base_title} - {file.filename}"
                else:
                    title = base_title
            else:
                title = file.filename
            
            try:
                # If file_path is provided (from ZIP extraction), use it directly
                # Otherwise, let process_document save the file
                document = document_service.process_document(
                    collection_id=collection_id,
                    file=file,
                    title=title,
                    use_docling=use_docling,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    chunk_strategy=chunk_strategy,
                    extract_metadata=extract_metadata,
                    file_path=file_path  # This will be None for regular uploads
                )
                processed_count += 1
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                flash(f"Error processing {file.filename}: {str(e)}", "error")
        
        # Clean up extracted files if needed
        for file_path, file in extracted_files:
            try:
                file.stream.close()  # Close the file handle
            except:
                pass
        
        if processed_count == 1:
            flash('1 document uploaded and processed successfully', 'success')
        else:
            flash(f'{processed_count} documents uploaded and processed successfully', 'success')
            
        return redirect(url_for('admin.collection_detail', collection_id=collection_id))
    
    return render_template('admin/upload_document.html', collection=collection)

@admin_bp.route('/documents/<document_id>/delete', methods=['POST'])
@admin_required
def delete_document(document_id):
    """Delete a document"""
    document = Document.get(document_id)
    if not document:
        flash('Document not found', 'error')
        return redirect(url_for('admin.collections'))
    
    collection_id = document.collection_id
    title = document.title
    
    # First, delete from Qdrant to ensure vector points are removed
    qdrant = QdrantService()
    deletion_success = qdrant.delete_document_points(collection_id, document_id)
    
    # Then delete the document from database
    document.delete()
    
    # Also remove the file from filesystem if it exists
    if document.file_path and os.path.exists(document.file_path):
        try:
            os.remove(document.file_path)
        except Exception as e:
            print(f"Error deleting file: {e}")
    
    if not deletion_success:
        flash(f'Document "{title}" deleted, but there was an issue removing vectors from search index', 'warning')
    else:
        flash(f'Document "{title}" deleted successfully', 'success')
    
    return redirect(url_for('admin.collection_detail', collection_id=collection_id))

# Model Management Routes
@admin_bp.route('/models')
@admin_required
def models():
    """Model management dashboard"""
    model_service = ModelService()
    providers = model_service.get_available_providers()
    active_config = model_service.get_active_models()
    return render_template(
        'admin/models.html', 
        providers=providers, 
        active_config=active_config
    )
    
@admin_bp.route('/api/models/refresh', methods=['POST'])
@admin_required
def api_refresh_models():
    """API endpoint to refresh models from providers"""
    model_service = ModelService()
    try:
        # Force refresh of all models
        model_service.refresh_models(force=True)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@admin_bp.route('/api/models/providers')
@admin_required
def api_model_providers():
    """API endpoint to get available model providers"""
    model_service = ModelService()
    providers = model_service.get_available_providers()
    return jsonify({"providers": providers})

@admin_bp.route('/api/models/list')
@admin_required
def api_models_list():
    """API endpoint to get available models"""
    model_service = ModelService()
    provider = request.args.get('provider')
    model_type = request.args.get('type')
    models = model_service.get_available_models(provider, model_type)
    return jsonify({"models": models})

@admin_bp.route('/api/models/active')
@admin_required
def api_active_models():
    """API endpoint to get active models"""
    model_service = ModelService()
    active = model_service.get_active_models()
    return jsonify(active)

@admin_bp.route('/api/models/set-model', methods=['POST'])
@admin_required
def api_set_model():
    """API endpoint to set active model"""
    model_service = ModelService()
    data = request.json
    
    provider = data.get('provider')
    model_id = data.get('model_id')
    model_type = data.get('type')
    
    if not provider or not model_id or not model_type:
        return jsonify({"success": False, "error": "Missing required parameters"}), 400
        
    success = model_service.set_active_model(provider, model_id, model_type)
    
    if success:
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "error": "Failed to set model"}), 400

@admin_bp.route('/api/models/set-provider', methods=['POST'])
@admin_required
def api_set_provider():
    """API endpoint to set active provider"""
    model_service = ModelService()
    data = request.json
    
    provider = data.get('provider')
    
    if not provider:
        return jsonify({"success": False, "error": "Missing provider parameter"}), 400
        
    success = model_service.set_active_provider(provider)
    
    if success:
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "error": "Failed to set provider"}), 400