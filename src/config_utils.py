# ============================================================================
# CONFIG_UTILS.PY — Funciones auxiliares y diagnósticos del proyecto
# ============================================================================
# TFM: Predicción de Abandono Universitario
#
# Funciones para comprobar que el proyecto está bien configurado:
#   - info_entorno() → Muestra rutas y entorno detectado
#   - verificar_directorios() → Crea carpetas si no existen
#   - resumen_tablas() → Lista las tablas del proyecto
#   - resumen_columnas() → Lista las columnas definidas
#   - diagnostico_proyecto() → Ejecuta todo lo anterior de golpe
#
# ¿Cuándo usar esto?
#   Al principio de un notebook nuevo, para confirmar que todo está OK:
#     from src.config import info_entorno
#     info_entorno()
# ============================================================================

from pathlib import Path

from src.utils.log import (
    log_info, log_warning, log_error, log_debug, log_success
)

from .config_entorno import (
    ENTORNO, BASE_PATH,
    RUTA_RAW, RUTA_INTERIM, RUTA_PROCESSED, RUTA_FEATURES, RUTA_AUTOML,
    RUTA_HTML, RUTA_EDA, RUTA_REPORTES, RUTA_ASSETS, RUTA_DOCS,
    RUTA_NOTEBOOKS, RUTA_TESTS,
    EXCEL_PRINCIPAL
)

from .config_datos import (
    TABLAS_INFO,
    DICCIONARIO_COLUMNAS
)


# ============================================================================
# 1. INFO ENTORNO
# ============================================================================

def info_entorno() -> None:
    """
    Muestra en consola las rutas y el entorno detectado.

    Útil para verificar que el proyecto se está ejecutando
    desde la ubicación correcta.

    Example
    -------
    >>> from src.config import info_entorno
    >>> info_entorno()
    📌 INFORMACIÓN DEL ENTORNO DEL PROYECTO
    🖥️  Entorno detectado: Local
    📂 Ruta base: /home/usuario/AU_UJI
    ...
    """
    log_info("=" * 75)
    log_info("📌 INFORMACIÓN DEL ENTORNO DEL PROYECTO")
    log_info("=" * 75)
    log_info(f"🖥️  Entorno detectado: {ENTORNO}")
    log_info(f"📂 Ruta base:     {BASE_PATH}")
    log_info(f"📁 RAW:           {RUTA_RAW}")
    log_info(f"📁 INTERIM:       {RUTA_INTERIM}")
    log_info(f"📁 PROCESSED:     {RUTA_PROCESSED}")
    log_info(f"📁 FEATURES:      {RUTA_FEATURES}")
    log_info(f"📁 AUTOML:        {RUTA_AUTOML}")
    log_info(f"📁 NOTEBOOKS:     {RUTA_NOTEBOOKS}")
    log_info(f"📄 Excel principal: {EXCEL_PRINCIPAL}")
    log_info("=" * 75)


# ============================================================================
# 2. VERIFICACIÓN DE DIRECTORIOS
# ============================================================================

def verificar_directorios() -> None:
    """
    Crea todas las carpetas del proyecto si no existen.

    Seguro ejecutar múltiples veces — no borra contenido.

    Example
    -------
    >>> from src.config import verificar_directorios
    >>> verificar_directorios()
    ✅ Directorios verificados.
    """
    directorios = [
        RUTA_RAW, RUTA_INTERIM, RUTA_PROCESSED, RUTA_FEATURES, RUTA_AUTOML,
        RUTA_HTML, RUTA_EDA, RUTA_REPORTES, RUTA_ASSETS,
        RUTA_DOCS, RUTA_NOTEBOOKS, RUTA_TESTS,
        (BASE_PATH / 'logs'),
        (BASE_PATH / 'results'),
        (BASE_PATH / 'results' / 'metricas'),
    ]

    for ruta in directorios:
        ruta.mkdir(parents=True, exist_ok=True)

    log_success("Directorios verificados.")


# ============================================================================
# 3. RESÚMENES DE TABLAS Y COLUMNAS
# ============================================================================

def resumen_tablas() -> None:
    """Muestra en consola la lista de tablas definidas en TABLAS_INFO."""
    log_info("📊 RESUMEN DE TABLAS")
    for nombre, info in TABLAS_INFO.items():
        descripcion = info.get('descripcion', 'Sin descripción')
        log_info(f"  - {nombre}: {descripcion}")


def resumen_columnas() -> None:
    """Muestra en consola la lista de columnas definidas en DICCIONARIO_COLUMNAS."""
    log_info("📄 RESUMEN DE COLUMNAS")
    for col, info in DICCIONARIO_COLUMNAS.items():
        descripcion = info.get('descripcion', 'Sin descripción')
        log_info(f"  - {col}: {descripcion}")


# ============================================================================
# 4. DIAGNÓSTICO COMPLETO
# ============================================================================

def diagnostico_proyecto() -> None:
    """
    Ejecuta un diagnóstico completo del proyecto.

    Llama a info_entorno + verificar_directorios + resumen_tablas + resumen_columnas.

    Example
    -------
    >>> from src.config import diagnostico_proyecto
    >>> diagnostico_proyecto()
    """
    info_entorno()
    verificar_directorios()
    resumen_tablas()
    resumen_columnas()
    log_success("Diagnóstico completado.")
