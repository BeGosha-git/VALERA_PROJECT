from app import create_app

app = create_app()

if __name__ == '__main__':
    # Create necessary directories
    import os
    from app.models.collection import Collection
    from app.models.document import Document
    
    # Create data directories
    Collection.create_storage()
    Document.create_storage()
    
    # Check if running in Docker
    in_docker = os.environ.get('DOCKER_ENV', False)
    
    # Run the application
    # When in docker, listen on all interfaces (0.0.0.0)
    app.run(debug=True, host='0.0.0.0' if in_docker else '127.0.0.1')