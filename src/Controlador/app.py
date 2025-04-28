from flask import Flask
from flask import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)

    # Registrar blueprints aquí
    from src.Controlador.Rutas.pred_routes import pred_bp
    from src.Controlador.Rutas.data_routes import data_bp
    
    app.register_blueprint(pred_bp, url_prefix='/api/prediccion')
    app.register_blueprint(data_bp, url_prefix='/api/datos')

    return app