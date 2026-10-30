import os
import threading
from flask import Flask
from flask_login import LoginManager
from dotenv import load_dotenv
import jinja2
import markdown

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default-dev-key')
    
    # Login page background configuration
    app.config['LOGIN_USE_BACKGROUND_IMAGES'] = os.getenv('LOGIN_USE_BACKGROUND_IMAGES', 'true').lower() == 'true'
    
    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    
    # Import and register blueprints
    from app.routes.main import main_bp
    from app.routes.admin import admin_bp
    from app.routes.auth import auth_bp
    from app.routes.chat import chat_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(admin_bp, url_prefix='/admin')
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(chat_bp)
    
    # Setup login manager user loader
    from app.models.user import User
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.get(user_id)
    
    # Add custom Jinja2 filters
    @app.template_filter('nl2br')
    def nl2br(value):
        if value:
            return jinja2.utils.markupsafe.Markup(value.replace('\n', '<br>'))
        return ''
    
    @app.template_filter('markdown')
    def render_markdown(text):
        if text:
            return jinja2.utils.markupsafe.Markup(markdown.markdown(text))
        return ''
    
    # Initialize admin user from environment variables
    with app.app_context():
        User.initialize_admin()

    # Предзагрузка активных движков SearXNG в фоне, чтобы первый запрос
    # не ждал пинг (~20с). Пинг идёт один раз и кэшируется на уровне класса.
    def _prewarm_searxng():
        try:
            from app.services.searxng_service import SearXNGService
            svc = SearXNGService()
            svc._get_active_engines()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Pre-warm SearXNG engines failed: {e}")

    threading.Thread(target=_prewarm_searxng, daemon=True).start()

    # Предзагрузка модели эмбеддингов для поиска по базе знаний.
    # Без неё первый поиск по коллекции ждёт загрузку SentenceTransformer
    # с диска (до ~17с на холодную). Грузим в фоне при старте — дальше все
    # запросы берут модель из общего кэша процесса (~0.01с).
    def _prewarm_embeddings():
        try:
            from app.services.qdrant_service import get_sentence_transformer
            model_name = None
            try:
                # Сначала — модель, на которой построена коллекция «МИРЭА»:
                # именно ей кодируются вопросы гостей.
                from app.models.collection import Collection
                for col in Collection.get_all():
                    if (col.name or '').strip().lower() == 'мирэа' and col.embedding_model:
                        model_name = col.embedding_model
                        break
            except Exception:
                pass
            if not model_name:
                model_name = 'all-MiniLM-L6-v2'
                try:
                    from app.services.model_service import ModelService
                    cfg = ModelService().config
                    model_name = cfg["active"]["models"].get("embedding") or model_name
                except Exception:
                    pass
            get_sentence_transformer(model_name)
            import logging
            logging.getLogger(__name__).info(
                f"Embedding model preloaded: {model_name}"
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Pre-warm embeddings failed: {e}"
            )

    threading.Thread(target=_prewarm_embeddings, daemon=True).start()

    # Предсборка снимка коллекции (чанки + вектора + IDF) в памяти.
    # Поиск работает по снимку; без прогрева первый вопрос ждёт ~2.5с (модель
    # эмбеддингов + выгрузка коллекции). С прогрева первый ответ сразу быстрый.
    def _prewarm_snapshot():
        try:
            import logging
            log = logging.getLogger(__name__)
            from app.services.document_service import (
                DocumentService, _get_snapshot, _build_snapshot,
            )
            ds = DocumentService()
            collections = ds.qdrant_service.list_collections() or []
            for col in collections[:1]:  # активная база — обычно одна
                cid = col if isinstance(col, str) else (col.get("id") or "")
                if not cid:
                    continue
                snap = _build_snapshot(ds.qdrant_service, cid) or {}
                # Кладём в кэш под тем же ключом, что использует поиск.
                from app.services import document_service as _d
                if snap:
                    import time as _t
                    with _d._SNAPSHOT_LOCK:
                        _d._SNAPSHOT_CACHE[str(cid)] = snap
                    log.info(
                        f"KB snapshot preloaded: {cid} "
                        f"({len(snap.get('items', []))} chunks)"
                    )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Pre-warm snapshot failed: {e}")

    threading.Thread(target=_prewarm_snapshot, daemon=True).start()

    # ------------------------------------------------------------------
    # Сжатие ответов gzip
    # ------------------------------------------------------------------
    # Зачем: страница главной — ~33 КБ. В некоторых окружениях (например,
    # при активном VPN-клиенте с перехватом loopback-трафика) несжатые
    # ответы больше ~16 КБ «зависают»: заголовки приходят, а тело — нет.
    # Браузер в итоге показывает пустую страницу. gzip уменьшает объём
    # в 3-4 раза и снимает проблему. Мелкие ответы не трогаем.
    import gzip as _gzip
    from flask import request as _request

    @app.after_request
    def _compress_response(response):
        try:
            accept = _request.headers.get("Accept-Encoding", "").lower()
            if "gzip" not in accept:
                return response
            if response.direct_passthrough or response.is_streamed or response.status_code >= 300:
                return response
            if response.headers.get("Content-Encoding"):
                return response
            body = response.get_data()
            if len(body) < 1024:          # мелочь сжимать смысла нет
                return response
            compressed = _gzip.compress(body, 6)
            if len(compressed) >= len(body):
                return response
            response.set_data(compressed)
            response.headers["Content-Encoding"] = "gzip"
            response.headers["Content-Length"] = str(len(compressed))
            response.headers["Vary"] = "Accept-Encoding"
        except Exception:  # noqa: BLE001
            # Сжатие — оптимизация; при любой ошибке отдаём ответ как есть
            pass
        return response

    return app