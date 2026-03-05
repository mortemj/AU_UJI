# ============================================================================
# CONFIG.PY — Hub central de configuración
# ============================================================================
# TFM: Predicción de Abandono Universitario
#
# Este archivo es el PUNTO DE ENTRADA de toda la configuración.
# No define nada propio: solo importa y reexporta desde los módulos
# especializados. Así los notebooks pueden hacer:
#
#   from src.config import RUTA_RAW, AUTORA, TABLAS_INFO, info_entorno
#
# sin tener que saber en qué archivo está cada cosa.
#
# ¿Por qué está organizado así?
#   - config_entorno.py → Rutas de archivos y detección de entorno
#   - config_proyecto.py → Autora, GitHub, versión, colores, fases
#   - config_datos.py → Diccionarios y constantes de los datos
#   - config_utils.py → Funciones de diagnóstico (info_entorno, etc.)
# ============================================================================

# --- Entorno y rutas ---
from .config_entorno import (
    BASE_PATH, ENTORNO,
    RUTA_RAW, RUTA_INTERIM, RUTA_PROCESSED,
    RUTA_FEATURES, RUTA_AUTOML,
    RUTA_HTML, RUTA_EDA, RUTA_REPORTES, RUTA_ASSETS, RUTA_DOCS,
    RUTA_NOTEBOOKS, RUTA_TESTS,
    EXCEL_PRINCIPAL, EXCEL_PREINSCRIPCION,
    DATASET_FINAL_PARQUET, DATASET_FINAL_CSV,
    ARCHIVO_LOG,
    MAPEO_HOJAS, HOJAS_EXCEL_PRINCIPAL, NOMBRES_PARQUET
)

# --- Identidad del proyecto ---
from .config_proyecto import (
    AUTORA, EMAIL_UOC, EMAIL_UJI,
    GITHUB_REPO, GITHUB_NOTEBOOKS,
    VERSION_DATOS,
    COLORES, ColoresTFM,
    FASES_CONFIG
)

# --- Diccionarios y constantes de datos ---
from .config_datos import (
    TABLAS_INFO,
    DICCIONARIO_COLUMNAS,
    DICCIONARIO_RAMAS,
    DICCIONARIO_FORMA_PAGO,
    VALORES_NULOS,
    CURSO_REFERENCIA,
    FECHA_REFERENCIA,
    VALORES_VACIOS,
    DICCIONARIO_CP_PROVINCIA,
    ETIQUETAS_VARIABLES
)

# --- Funciones de diagnóstico ---
from .config_utils import (
    info_entorno,
    verificar_directorios,
    resumen_tablas,
    resumen_columnas,
    diagnostico_proyecto
)
