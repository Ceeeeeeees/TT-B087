def validarDatosEstudiante(datos):
    """
    Validar datos ingresados por el usuario para la predicción.
    ---
    Se asegura que todos los campos requeridos estén presentes y sean válidos.
    """

    required_fields = ['nombre', 'edad', 'calificaciones']
    for field in required_fields:
        if field not in datos:
            return f"Falta el campo requerido: {field}"
    return None