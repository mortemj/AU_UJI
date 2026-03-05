# ============================================================================
# METRICAS.PY — Sistema de métricas compartidas entre notebooks
# ============================================================================
# TFM: Predicción de Abandono Universitario
#
# Guarda y carga métricas en archivos JSON para que los notebooks
# puedan compartir información entre sí sin depender de variables globales.
#
# Ejemplo: Fase 1 guarda las métricas de limpieza, y Fase 2 las lee
# para mostrar un resumen en el EDA.
#
# Uso básico:
#   from src.metricas import guardar_metricas, cargar_metricas
#
#   # Guardar
#   guardar_metricas('fase1_limpieza', {'n_tablas': 9, 'total_filas': 839000})
#
#   # Cargar (desde otro notebook)
#   datos = cargar_metricas('fase1_limpieza')
#   print(datos['n_tablas'])  # → 9
#
# Los archivos se guardan en: results/metricas/*.json
# ============================================================================

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
from src.utils.log import log_info, log_error, log_success, log_warning


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

def _obtener_ruta_metricas() -> Path:
    """
    Busca la carpeta results/metricas/ subiendo niveles.

    Returns
    -------
    Path
        Ruta a la carpeta de métricas
    """
    posibles_raices = [
        Path.cwd() / 'results' / 'metricas',
        Path.cwd().parent / 'results' / 'metricas',
        Path.cwd().parent.parent / 'results' / 'metricas',
    ]

    for ruta in posibles_raices:
        if ruta.parent.exists():
            ruta.mkdir(parents=True, exist_ok=True)
            return ruta

    ruta = Path.cwd() / 'results' / 'metricas'
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


# ============================================================================
# FUNCIONES PRINCIPALES
# ============================================================================

def guardar_metricas(
    nombre_archivo: str,
    datos: Dict[str, Any],
    fase: Optional[str] = None,
    notebook: Optional[str] = None,
    descripcion: Optional[str] = None
) -> Path:
    """
    Guarda métricas en un archivo JSON.

    Parameters
    ----------
    nombre_archivo : str
        Nombre del archivo (sin .json). Ej: 'fase1_limpieza'
    datos : Dict
        Diccionario con las métricas a guardar
    fase : str, optional
        Fase del proyecto (ej: 'fase1')
    notebook : str, optional
        Notebook que genera estas métricas
    descripcion : str, optional
        Descripción de qué contienen estas métricas

    Returns
    -------
    Path
        Ruta del archivo guardado
    """
    ruta_base = _obtener_ruta_metricas()
    ruta_archivo = ruta_base / f'{nombre_archivo}.json'

    metricas = {
        '_meta': {
            'version': '1.0.0',
            'fecha_generacion': datetime.now().isoformat(),
            'fecha_formato': datetime.now().strftime('%d/%m/%Y %H:%M'),
            'fase': fase,
            'notebook': notebook,
            'descripcion': descripcion
        },
        **datos
    }

    with open(ruta_archivo, 'w', encoding='utf-8') as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)

    log_success(f"Métricas guardadas: {ruta_archivo}")
    return ruta_archivo


def cargar_metricas(nombre_archivo: str) -> Dict[str, Any]:
    """
    Carga métricas desde un archivo JSON.

    Parameters
    ----------
    nombre_archivo : str
        Nombre del archivo (sin .json)

    Returns
    -------
    Dict
        Contenido del archivo de métricas

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe
    """
    ruta_base = _obtener_ruta_metricas()
    ruta_archivo = ruta_base / f'{nombre_archivo}.json'

    if not ruta_archivo.exists():
        raise FileNotFoundError(f"No se encontró: {ruta_archivo}")

    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        return json.load(f)


def existe_metricas(nombre_archivo: str) -> bool:
    """Comprueba si un archivo de métricas existe."""
    ruta_base = _obtener_ruta_metricas()
    return (ruta_base / f'{nombre_archivo}.json').exists()


def listar_metricas() -> List[str]:
    """Lista todos los archivos de métricas disponibles."""
    ruta_base = _obtener_ruta_metricas()
    return [f.stem for f in ruta_base.glob('*.json')]


# ============================================================================
# FUNCIONES ESPECÍFICAS POR FASE
# ============================================================================

def guardar_metricas_limpieza(tablas_procesadas: Dict[str, pd.DataFrame]) -> Path:
    """
    Guarda métricas de la Fase 1 (limpieza de tablas).

    Parameters
    ----------
    tablas_procesadas : Dict[str, pd.DataFrame]
        Diccionario {nombre_tabla: DataFrame_limpio}

    Returns
    -------
    Path
        Ruta del archivo guardado
    """
    tablas_info = {}
    total_filas = 0
    total_columnas = 0

    for nombre, df in tablas_procesadas.items():
        tablas_info[nombre] = {
            'filas': len(df),
            'columnas': len(df.columns),
            'columnas_lista': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'memoria_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
        total_filas += len(df)
        total_columnas += len(df.columns)

    datos = {
        'tablas': tablas_info,
        'totales': {
            'n_tablas': len(tablas_procesadas),
            'total_filas': total_filas,
            'total_columnas': total_columnas
        }
    }

    return guardar_metricas(
        'fase1_limpieza', datos,
        fase='fase1',
        notebook='f1_m02_limpieza.ipynb',
        descripcion='Métricas de las tablas limpias en data/01_interim/'
    )


def guardar_metricas_union(df_alumno: pd.DataFrame) -> Path:
    """
    Guarda métricas del dataset final unificado (Fase 1).

    Parameters
    ----------
    df_alumno : pd.DataFrame
        Dataset final unificado

    Returns
    -------
    Path
        Ruta del archivo guardado
    """
    datos = {
        'dataset_final': {
            'nombre': 'df_alumno',
            'filas': len(df_alumno),
            'columnas': len(df_alumno.columns),
            'columnas_lista': list(df_alumno.columns),
            'dtypes': {col: str(dtype) for col, dtype in df_alumno.dtypes.items()},
            'memoria_mb': round(df_alumno.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
    }

    nulos = df_alumno.isnull().sum()
    datos['nulos'] = {
        'por_columna': {col: int(n) for col, n in nulos.items() if n > 0},
        'total': int(nulos.sum()),
        'columnas_con_nulos': int((nulos > 0).sum())
    }

    return guardar_metricas(
        'fase1_union', datos,
        fase='fase1',
        notebook='f1_m04_dataset_final.ipynb',
        descripcion='Métricas del dataset unificado df_alumno'
    )


def guardar_metricas_eda(nombre: str, metricas: Dict[str, Any], notebook: str) -> Path:
    """Guarda métricas de un análisis EDA (Fase 2)."""
    return guardar_metricas(
        f'fase2_{nombre}', metricas,
        fase='fase2',
        notebook=notebook,
        descripcion=f'Métricas del análisis de {nombre}'
    )


# ============================================================================
# FUNCIONES DE RESUMEN
# ============================================================================

def obtener_resumen_proyecto() -> Dict[str, Any]:
    """
    Genera un resumen de todas las métricas disponibles del proyecto.

    Returns
    -------
    Dict
        Resumen con métricas de cada fase
    """
    resumen = {
        'fases': {},
        'fecha_resumen': datetime.now().isoformat()
    }

    if existe_metricas('fase1_limpieza'):
        metricas = cargar_metricas('fase1_limpieza')
        resumen['fases']['fase1'] = {
            'limpieza': metricas.get('totales', {}),
            'fecha': metricas.get('_meta', {}).get('fecha_formato')
        }

    if existe_metricas('fase1_union'):
        metricas = cargar_metricas('fase1_union')
        if 'fase1' not in resumen['fases']:
            resumen['fases']['fase1'] = {}
        resumen['fases']['fase1']['union'] = metricas.get('dataset_final', {})

    metricas_fase2 = [f for f in listar_metricas() if f.startswith('fase2_')]
    if metricas_fase2:
        resumen['fases']['fase2'] = {
            'analisis_completados': len(metricas_fase2),
            'archivos': metricas_fase2
        }

    return resumen


def imprimir_resumen_metricas(nombre_archivo: str) -> None:
    """Imprime un resumen legible de un archivo de métricas."""
    try:
        metricas = cargar_metricas(nombre_archivo)

        log_info(f"{'=' * 60}")
        log_info(f"MÉTRICAS: {nombre_archivo}")
        log_info(f"{'=' * 60}")

        meta = metricas.get('_meta', {})
        if meta:
            log_info(f"  📅 Generado: {meta.get('fecha_formato', 'N/A')}")
            log_info(f"  📓 Notebook: {meta.get('notebook', 'N/A')}")
            log_info(f"  📋 Descripción: {meta.get('descripcion', 'N/A')}")

        if 'totales' in metricas:
            log_info("  📊 TOTALES:")
            for k, v in metricas['totales'].items():
                log_info(f"    {k}: {v:,}" if isinstance(v, int) else f"    {k}: {v}")

        if 'dataset_final' in metricas:
            df_info = metricas['dataset_final']
            log_info("  🎯 DATASET FINAL:")
            log_info(f"    Filas: {df_info.get('filas', 0):,}")
            log_info(f"    Columnas: {df_info.get('columnas', 0)}")

        log_info(f"{'=' * 60}")

    except FileNotFoundError:
        log_error(f"No se encontró: {nombre_archivo}")
