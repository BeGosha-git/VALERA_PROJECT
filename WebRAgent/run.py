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
    
    # Run the application.
    #   threaded=True  — обрабатываем запросы ПАРАЛЛЕЛЬНО. Иначе один долгий
    #                    запрос (например, заливка документа в базу знаний)
    #                    блокирует весь сервер, и остальные соединения
    #                    «висят» в очереди, не читаясь (наблюдали зависание
    #                    заливки на большом файле).
    #   use_reloader=False — НЕ перезапускать сервер при изменении файлов кода.
    #                    Перезапуск посреди долгой операции рвал запросы
    #                    (заливка базы обрывалась на середине).
    debug = os.environ.get('FLASK_DEBUG', '0') in ('1', 'true', 'True')
    app.run(
        debug=debug,
        threaded=True,
        use_reloader=False,
        host='0.0.0.0' if in_docker else '127.0.0.1',
    )