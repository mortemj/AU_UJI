# ============================================================================
# CONFIG_ENTORNO.PY — Entorno y rutas del proyecto
# ============================================================================
# TFM: Predicción de Abandono Universitario
#
# Este archivo es la FUENTE ÚNICA de verdad para:
#   1. Detectar desde dónde se ejecuta el proyecto (Colab, Kaggle, Local)
#   2. Definir TODAS las rutas de archivos y carpetas
#   3. Definir el mapeo de hojas del Excel original
#
# ¿Cómo funciona la detección de entorno?
#   - Google Colab: busca en Google Drive la carpeta AU_UJI
#   - Kaggle: busca en /kaggle/working/
#   - Local: sube niveles desde la carpeta actual hasta encontrar src/
#
# IMPORTANTE: Si renombras carpetas de datos, SOLO hay que cambiar aquí.
# Todos los notebooks importan estas rutas vía src/config.py.
# ============================================================================

from pathlib import Path
from typing import Tuple
import os
import sys


# ============================================================================
# 1. DETECCIÓN DE ENTORNO
# ============================================================================

def detectar_entorno() -> Tuple[Path, str]:
    """
    Detecta el entorno de ejecución y devuelve la ruta base del proyecto.

    Returns
    -------
    Tuple[Path, str]
        (ruta_base, nombre_entorno)
        Ejemplo: (Path('/home/user/AU_UJI'), 'Local')

    Entornos soportados
    -------------------
    - Google Colab: /content/drive/MyDrive/AU_UJI
    - Kaggle: /kaggle/working/AU_UJI
    - Azure Notebooks: /home/azureuser/AU_UJI
    - Local (Windows/Mac/Linux): busca hacia arriba hasta encontrar src/
    """
    # --- Google Colab ---
    if 'google.colab' in sys.modules:
        return Path('/content/drive/MyDrive/AU_UJI'), 'Colab'

    # --- Kaggle ---
    if os.path.exists('/kaggle'):
        return Path('/kaggle/working/AU_UJI'), 'Kaggle'

    # --- Azure Notebooks ---
    if 'AZURE_NOTEBOOKS' in os.environ:
        return Path('/home/azureuser/AU_UJI'), 'Azure'

    # --- Local ---
    # Sube niveles desde la carpeta actual hasta encontrar la que tiene src/
    current = Path.cwd()
    for parent in current.parents:
        if (parent / 'src').exists():
            return parent, 'Local'

    # Fallback: si no encuentra src/, usa la carpeta actual
    return current, 'Local'


BASE_PATH, ENTORNO = detectar_entorno()


# ============================================================================
# 2. RUTAS DEL PROYECTO
# ============================================================================
# Estructura de carpetas de datos:
#
#   data/
#   ├── 00_raw/          ← Excel originales (datos de la universidad)
#   ├── 01_interim/      ← Parquets individuales (1 por hoja, Fase 1)
#   ├── 02_processed/    ← df_alumno.parquet (dataset unificado, Fase 1)
#   ├── 03_features/     ← df_expediente_features.parquet (Fase 3)
#   └── automl/          ← Dataset para AutoML (Fase 3.5)

# --- Datos ---
RUTA_RAW = BASE_PATH / 'data' / '00_raw'
RUTA_INTERIM = BASE_PATH / 'data' / '01_interim'
RUTA_PROCESSED = BASE_PATH / 'data' / '02_processed'
RUTA_FEATURES = BASE_PATH / 'data' / '03_features'
RUTA_AUTOML = BASE_PATH / 'data' / 'automl'

# --- Documentación y HTML ---
RUTA_HTML = BASE_PATH / 'docs' / 'html'
RUTA_EDA = BASE_PATH / 'docs' / 'eda'
RUTA_REPORTES = BASE_PATH / 'docs' / 'reportes'
RUTA_ASSETS = BASE_PATH / 'docs' / 'assets'
RUTA_DOCS = BASE_PATH / 'docs'

# --- Código y notebooks ---
RUTA_SRC = BASE_PATH / 'src'
RUTA_TESTS = BASE_PATH / 'tests'
RUTA_NOTEBOOKS = BASE_PATH / 'notebooks'

# --- Archivos específicos ---
EXCEL_PRINCIPAL = RUTA_RAW / 'datos_proyecto_sin_preinscrip.xlsx'
EXCEL_PREINSCRIPCION = RUTA_RAW / 'preinscripcion_si.xlsx'

DATASET_FINAL_PARQUET = RUTA_PROCESSED / 'df_alumno.parquet'
DATASET_FINAL_CSV = RUTA_PROCESSED / 'df_alumno.csv'

ARCHIVO_LOG = BASE_PATH / 'logs' / 'tfm_abandono.log'


# ============================================================================
# 3. MAPEO DE HOJAS DEL EXCEL
# ============================================================================
# El Excel principal tiene 8 hojas con nombres "oficiales" (de la universidad).
# Internamente usamos nombres más cortos y descriptivos.
#
# Ejemplo: La hoja "Nac-Sexo_Nacionalidad" se convierte en el parquet
#          "demograficos.parquet" en data/01_interim/

MAPEO_HOJAS = {
    'Titulaciones': 'titulaciones',
    'Recibos': 'recibos',
    'Domicilios': 'domicilios',
    'Expedientes': 'expedientes',
    'Nac-Sexo_Nacionalidad': 'demograficos',
    'Circunstancias': 'becas',
    'Trabajo': 'trabajo',
    'Notas': 'notas',
}

# Listas derivadas (para iterar cómodamente en los notebooks)
HOJAS_EXCEL_PRINCIPAL = list(MAPEO_HOJAS.keys())
NOMBRES_PARQUET = list(MAPEO_HOJAS.values())
