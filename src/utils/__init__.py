# ============================================================================
# SRC/UTILS/__INIT__.PY — Exports del módulo de utilidades
# ============================================================================
# TFM: Predicción de Abandono Universitario
#
# Este archivo define qué funciones y constantes están disponibles
# cuando haces: from src.utils import formato_numero_es, crear_directorios
#
# Los módulos de utilidades son:
#   - formatters.py → Formato numérico español (1.234,56)
#   - files.py → Guardar/cargar archivos (parquet, CSV)
#   - text.py → Transformación de texto (snake_case, etc.)
#   - log.py → Sistema de logging (consola + archivo)
#   - progress.py → Barras de progreso (tqdm)
#   - graficos.py → Gráficos matplotlib reutilizables
#
# NOTA: Las constantes de proyecto (AUTORA, GITHUB_REPO, COLORES)
# se importan desde src.config_proyecto, NO desde aquí.
# Usar: from src.config import AUTORA, COLORES, GITHUB_REPO
# ============================================================================

# --- Formatters ---
from .formatters import (
    formato_numero_es,
    formato_porcentaje_es,
    formato_fecha_es,
    formato_moneda_es,
    formato_miles_es,
    DECIMAL_SEP,
    MILES_SEP,
    FORMATO_FECHA,
    FORMATO_DATETIME,
)

# --- Files ---
from .files import (
    crear_directorios,
    guardar_parquet_con_metadata,
    guardar_csv_espanol,
    cargar_parquet,
    verificar_archivo,
    listar_archivos,
    verificar_paquetes,
    CSV_SEP,
    ENCODING_CSV,
)

# --- Text ---
from .text import (
    convertir_a_snake_case,
    estandarizar_columnas,
    limpiar_texto,
    capitalizar_nombre,
    truncar_texto,
)

# --- Log ---
from .log import (
    configurar_logging,
    log_info,
    log_warning,
    log_error,
    log_debug,
    log_success,
    log_step,
)

# --- Progress ---
from .progress import (
    progreso,
    progreso_manual,
)

__all__ = [
    # Formatters
    'formato_numero_es', 'formato_porcentaje_es', 'formato_fecha_es',
    'formato_moneda_es', 'formato_miles_es',
    'DECIMAL_SEP', 'MILES_SEP', 'FORMATO_FECHA', 'FORMATO_DATETIME',

    # Files
    'crear_directorios', 'guardar_parquet_con_metadata', 'guardar_csv_espanol',
    'cargar_parquet', 'verificar_archivo', 'listar_archivos', 'verificar_paquetes',
    'CSV_SEP', 'ENCODING_CSV',

    # Text
    'convertir_a_snake_case', 'estandarizar_columnas', 'limpiar_texto',
    'capitalizar_nombre', 'truncar_texto',

    # Log
    'configurar_logging', 'log_info', 'log_warning', 'log_error',
    'log_debug', 'log_success', 'log_step',

    # Progress
    'progreso', 'progreso_manual',
]
