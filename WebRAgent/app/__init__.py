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
    
    return app