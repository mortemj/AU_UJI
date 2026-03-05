# ============================================================================
# FILES.PY — FUNCIONES DE I/O DE ARCHIVOS
# ============================================================================
# TFM: Predicción de Abandono Universitario
# Autora: María José Morte
# ============================================================================
# Funciones para guardar y cargar archivos (parquet, CSV, etc.)
#
# NOTA: Los datos de autora/versión se importan de config_proyecto.
# NO están hardcodeados en las funciones.
# ============================================================================

from pathlib import Path
from typing import Optional, List
from datetime import datetime
import pandas as pd

from .log import log_info, log_error, log_success


# ============================================================================
# IMPORTAR DATOS DE IDENTIDAD (FUENTE ÚNICA)
# ============================================================================

try:
    from ..config_proyecto import AUTORA, VERSION_DATOS
    _CONFIG_DISPONIBLE = True
except ImportError:
    try:
        from src.config_proyecto import AUTORA, VERSION_DATOS
        _CONFIG_DISPONIBLE = True
    except ImportError:
        AUTORA = "María José Morte"
        VERSION_DATOS = "1.0.0"
        _CONFIG_DISPONIBLE = False


# Constantes para CSV español
CSV_SEP: str = ';'
DECIMAL_SEP: str = ','
ENCODING_CSV: str = 'utf-8-sig'


def crear_directorios(rutas: List[Path]) -> None:
    """
    Crea múltiples directorios si no existen.

    Seguro ejecutar múltiples veces (no falla si ya existen).

    Parameters
    ----------
    rutas : List[Path]
        Lista de rutas de directorios a crear.

    Examples
    --------
    >>> from pathlib import Path
    >>> crear_directorios([Path('data/raw'), Path('data/processed')])
    ✓ Directorios verificados: 2
    """
    creados = 0
    for ruta in rutas:
        if not ruta.exists():
            ruta.mkdir(parents=True, exist_ok=True)
            creados += 1

    if creados > 0:
        log_info(f"Directorios creados: {creados}")
    else:
        log_info(f"Directorios verificados: {len(rutas)}")


def guardar_parquet_con_metadata(
    df: pd.DataFrame,
    ruta: Path,
    version: str = None,
    autora: str = None,
    compresion: str = 'snappy'
) -> Path:
    """
    Guarda DataFrame en parquet incluyendo metadatos de versión.

    Los valores por defecto se toman de config_proyecto.py (fuente única).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a guardar.
    ruta : Path
        Ruta del archivo de salida.
    version : str, optional
        Versión de los datos. Default: config_proyecto.VERSION_DATOS
    autora : str, optional
        Nombre de la autora. Default: config_proyecto.AUTORA
    compresion : str, optional
        Tipo de compresión. Por defecto 'snappy'.

    Returns
    -------
    Path
        Ruta del archivo guardado.
    """
    import pyarrow as pya
    import pyarrow.parquet as pq

    # Usar valores de config_proyecto si no se pasan explícitamente
    version = version or VERSION_DATOS
    autora = autora or AUTORA

    # Crear directorio si no existe
    ruta.parent.mkdir(parents=True, exist_ok=True)

    # Convertir a tabla de PyArrow
    table = pya.Table.from_pandas(df)

    # Añadir metadatos
    fecha_generacion = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    metadata = {
        b'version': version.encode(),
        b'fecha_generacion': fecha_generacion.encode(),
        b'autora': autora.encode(),
        b'filas': str(len(df)).encode(),
        b'columnas': str(len(df.columns)).encode()
    }

    # Combinar con metadatos existentes
    existing_metadata = table.schema.metadata or {}
    combined_metadata = {**existing_metadata, **metadata}
    table = table.replace_schema_metadata(combined_metadata)

    # Guardar
    pq.write_table(table, ruta, compression=compresion)

    log_success(f"Guardado: {ruta.name} ({len(df):,} filas, {len(df.columns)} columnas)")

    return ruta


def guardar_csv_espanol(
    df: pd.DataFrame,
    ruta: Path,
    incluir_indice: bool = False
) -> Path:
    """
    Guarda DataFrame en CSV con formato español.

    Usa punto y coma como separador, coma como decimal,
    y encoding utf-8-sig para que Excel español lo lea bien.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a guardar.
    ruta : Path
        Ruta del archivo de salida.
    incluir_indice : bool, optional
        Si incluir el índice. Por defecto False.

    Returns
    -------
    Path
        Ruta del archivo guardado.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        ruta,
        index=incluir_indice,
        sep=CSV_SEP,
        decimal=DECIMAL_SEP,
        encoding=ENCODING_CSV
    )

    log_success(f"Guardado: {ruta.name} ({len(df):,} filas)")

    return ruta


def cargar_parquet(ruta: Path) -> Optional[pd.DataFrame]:
    """
    Carga un archivo parquet y muestra información.

    Parameters
    ----------
    ruta : Path
        Ruta del archivo parquet.

    Returns
    -------
    pd.DataFrame or None
        DataFrame cargado, o None si hay error.
    """
    if not ruta.exists():
        log_error(f"No existe: {ruta}")
        return None

    try:
        df = pd.read_parquet(ruta)
        log_info(f"Cargado: {ruta.name} ({len(df):,} filas, {len(df.columns)} columnas)")
        return df
    except Exception as e:
        log_error(f"Error cargando {ruta.name}: {e}")
        return None


def verificar_archivo(ruta: Path) -> bool:
    """
    Verifica si un archivo existe y muestra su tamaño.

    Parameters
    ----------
    ruta : Path
        Ruta del archivo a verificar.

    Returns
    -------
    bool
        True si existe, False si no.
    """
    if ruta.exists():
        size_bytes = ruta.stat().st_size
        if size_bytes >= 1_000_000:
            size_str = f"{size_bytes / 1_000_000:.1f} MB"
        elif size_bytes >= 1_000:
            size_str = f"{size_bytes / 1_000:.1f} KB"
        else:
            size_str = f"{size_bytes} bytes"

        log_info(f"Existe: {ruta.name} ({size_str})")
        return True
    else:
        log_error(f"No existe: {ruta}")
        return False


def listar_archivos(
    directorio: Path,
    extension: str = '*',
    recursivo: bool = False
) -> List[Path]:
    """
    Lista archivos en un directorio.

    Parameters
    ----------
    directorio : Path
        Directorio a listar.
    extension : str, optional
        Extensión a filtrar (sin punto). Por defecto '*' (todos).
    recursivo : bool, optional
        Si buscar en subdirectorios. Por defecto False.

    Returns
    -------
    List[Path]
        Lista de rutas de archivos.
    """
    if not directorio.exists():
        log_error(f"No existe el directorio: {directorio}")
        return []

    patron = f"**/*.{extension}" if recursivo else f"*.{extension}"
    archivos = list(directorio.glob(patron))

    log_info(f"Encontrados: {len(archivos)} archivos .{extension} en {directorio.name}")

    return sorted(archivos)


def verificar_paquetes() -> bool:
    """
    Verifica que todas las dependencias necesarias están instaladas.

    Returns
    -------
    bool
        True si todas las dependencias están disponibles.
    """
    dependencias = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('pandera', 'pa'),
        ('matplotlib', 'plt'),
        ('seaborn', 'sns'),
        ('openpyxl', None),
        ('pyarrow', None)
    ]

    todas_ok = True
    faltantes = []

    for paquete, alias in dependencias:
        try:
            __import__(paquete)
        except ImportError:
            faltantes.append(paquete)
            todas_ok = False

    if todas_ok:
        log_success("Todas las dependencias OK")
    else:
        log_error(f"Faltan: {', '.join(faltantes)}")

    return todas_ok
