# ============================================================================
# SRC/__INIT__.PY — Punto de entrada del paquete src
# ============================================================================
# TFM: Predicción de Abandono Universitario
#
# Importa todo desde config.py para que los notebooks puedan hacer:
#   from src.config import RUTA_RAW, AUTORA, info_entorno
#
# También permite:
#   from src import config  (acceso al módulo completo)
# ============================================================================

from .config import *
