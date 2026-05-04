# scripts/core/errors.py

class DatasetError(Exception):
    """Error relacionado con la carga o lectura de datasets."""
    pass

class DownloadError(Exception):
    """Error relacionado con la descarga de datos."""
    pass

class ModelError(Exception):
    """Error relacionado con el entrenamiento o predicción de modelos."""
    pass
