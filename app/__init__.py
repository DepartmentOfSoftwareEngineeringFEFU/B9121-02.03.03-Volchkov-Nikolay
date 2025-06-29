from flask import Flask
from .config import setup_logging
from .routes import routes_bp

def create_app():
    setup_logging()
    app = Flask(__name__)
    app.register_blueprint(routes_bp)
    app.logger.info("Flask application initialized")
    return app