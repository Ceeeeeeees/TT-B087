from flask import Flask, Blueprint, request, jsonify
from src.Modelo.Prediccion import RealizarPrediccion
from src.Controlador.Utils.validacion import validarDatosEstudiante

pred_bp = Blueprint('prediccion', __name__)

@pred_bp.route('/individual', methods=['POST'])
def prediccionIndividual():
    """
    Endpoint para realizar la predicción del rendimiento académico de un estudiante.
    ---
    Recibe un JSON con datos del estudiante y devuelve la predicción.
    """
    datos = request.get_json()

    error = validarDatosEstudiante(datos)
    if error:
        return jsonify({"error": error}), 400
    try:
        resultado = RealizarPrediccion(datos)
        return jsonify({"resultado": resultado}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
