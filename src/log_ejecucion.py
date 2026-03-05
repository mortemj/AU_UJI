# ============================================================================
# LOG_EJECUCION.PY — Registro de ejecución de notebooks
# ============================================================================
# TFM: Predicción de Abandono Universitario
#
# Sistema para registrar qué notebooks se ejecutaron, cuándo y cuánto
# tardaron. Guarda un historial en results/log_ejecucion.json.
#
# Uso básico con context manager:
#   from src.log_ejecucion import LogEjecucion
#   with LogEjecucion('f1_m02_limpieza') as log:
#       # código del notebook
#       log.agregar_info('tablas_procesadas', 9)
#       log.agregar_archivo_generado('data/01_interim/expedientes.parquet')
#
# Uso rápido (sin context manager):
#   from src.log_ejecucion import registrar_inicio, registrar_fin
#   info = registrar_inicio('mi_notebook', fase='fase1')
#   # ... código ...
#   registrar_fin(info, archivos_generados=['archivo.parquet'])
# ============================================================================

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
from src.utils.log import log_info, log_error, log_success, log_warning


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

def _obtener_ruta_log() -> Path:
    """
    Busca la carpeta results/ subiendo niveles desde el directorio actual.

    Returns
    -------
    Path
        Ruta al archivo log_ejecucion.json
    """
    posibles_rutas = [
        Path.cwd() / 'results' / 'log_ejecucion.json',
        Path.cwd().parent / 'results' / 'log_ejecucion.json',
        Path.cwd().parent.parent / 'results' / 'log_ejecucion.json',
    ]

    for ruta in posibles_rutas:
        if ruta.parent.exists():
            return ruta

    # Fallback: crear en carpeta actual
    ruta = Path.cwd() / 'results' / 'log_ejecucion.json'
    ruta.parent.mkdir(parents=True, exist_ok=True)
    return ruta


# ============================================================================
# CLASE PRINCIPAL
# ============================================================================

class LogEjecucion:
    """
    Context manager para registrar la ejecución de un notebook.

    Registra automáticamente: inicio, fin, duración, estado (completado/error),
    archivos generados e información adicional.

    Parameters
    ----------
    notebook : str
        Nombre del notebook (ej: 'f1_m02_limpieza')
    fase : str, optional
        Fase del proyecto (ej: 'fase1')

    Examples
    --------
    >>> with LogEjecucion('f1_m02_limpieza', fase='fase1') as log:
    ...     # código del notebook
    ...     log.agregar_info('tablas_procesadas', 9)
    ...     log.agregar_archivo_generado('data/01_interim/expedientes.parquet')
    """

    def __init__(self, notebook: str, fase: str = None):
        self.notebook = notebook
        self.fase = fase
        self.inicio = None
        self.info = {}
        self.archivos_generados = []
        self.errores = []

    def __enter__(self):
        self.inicio = datetime.now()
        log_info(f"{'=' * 60}")
        log_info(f"🚀 INICIANDO: {self.notebook}")
        log_info(f"   Fecha: {self.inicio.strftime('%d/%m/%Y %H:%M:%S')}")
        log_info(f"{'=' * 60}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        fin = datetime.now()
        duracion = (fin - self.inicio).total_seconds()

        if exc_type is not None:
            estado = 'error'
            self.errores.append(str(exc_val))
        else:
            estado = 'completado'

        _registrar_ejecucion(
            notebook=self.notebook,
            fase=self.fase,
            inicio=self.inicio.isoformat(),
            fin=fin.isoformat(),
            duracion_segundos=round(duracion, 2),
            estado=estado,
            info=self.info,
            archivos_generados=self.archivos_generados,
            errores=self.errores
        )

        log_info(f"{'=' * 60}")
        if estado == 'completado':
            log_success(f"COMPLETADO: {self.notebook}")
        else:
            log_error(f"ERROR: {self.notebook}")
        log_info(f"   Duración: {_formatear_duracion(duracion)}")
        if self.archivos_generados:
            log_info(f"   Archivos generados: {len(self.archivos_generados)}")
        log_info(f"{'=' * 60}")

    def agregar_info(self, clave: str, valor: Any):
        """Añade información adicional al log (ej: nº tablas procesadas)."""
        self.info[clave] = valor

    def agregar_archivo_generado(self, ruta: str):
        """Registra un archivo generado durante la ejecución."""
        self.archivos_generados.append(ruta)

    def agregar_error(self, mensaje: str):
        """Registra un error no crítico (el notebook sigue ejecutándose)."""
        self.errores.append(mensaje)


# ============================================================================
# FUNCIONES DE LOG
# ============================================================================

def _registrar_ejecucion(**kwargs) -> None:
    """Registra una ejecución en el archivo JSON."""
    ruta_log = _obtener_ruta_log()

    if ruta_log.exists():
        with open(ruta_log, 'r', encoding='utf-8') as f:
            log = json.load(f)
    else:
        log = {'ejecuciones': []}

    log['ejecuciones'].append(kwargs)
    log['ultima_actualizacion'] = datetime.now().isoformat()

    with open(ruta_log, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def registrar_inicio(notebook: str, fase: str = None) -> Dict:
    """
    Registra el inicio de ejecución de un notebook.

    Parameters
    ----------
    notebook : str
        Nombre del notebook
    fase : str, optional
        Fase del proyecto

    Returns
    -------
    Dict
        Info del inicio (pasar a registrar_fin)
    """
    return {
        'notebook': notebook,
        'fase': fase,
        'inicio': datetime.now()
    }


def registrar_fin(
    info_inicio: Dict,
    archivos_generados: List[str] = None,
    metricas: Dict = None
) -> None:
    """
    Registra el fin de ejecución de un notebook.

    Parameters
    ----------
    info_inicio : Dict
        Diccionario devuelto por registrar_inicio()
    archivos_generados : List[str], optional
        Lista de rutas de archivos generados
    metricas : Dict, optional
        Métricas adicionales a guardar
    """
    fin = datetime.now()
    duracion = (fin - info_inicio['inicio']).total_seconds()

    _registrar_ejecucion(
        notebook=info_inicio['notebook'],
        fase=info_inicio.get('fase'),
        inicio=info_inicio['inicio'].isoformat(),
        fin=fin.isoformat(),
        duracion_segundos=round(duracion, 2),
        estado='completado',
        archivos_generados=archivos_generados or [],
        metricas=metricas or {}
    )

    log_success(f"{info_inicio['notebook']} completado en {_formatear_duracion(duracion)}")


# ============================================================================
# FUNCIONES DE CONSULTA
# ============================================================================

def obtener_historial(notebook: str = None, ultimos: int = 10) -> List[Dict]:
    """
    Obtiene el historial de ejecuciones.

    Parameters
    ----------
    notebook : str, optional
        Filtrar por notebook específico
    ultimos : int
        Número máximo de ejecuciones a devolver

    Returns
    -------
    List[Dict]
        Lista de ejecuciones (más recientes al final)
    """
    ruta_log = _obtener_ruta_log()

    if not ruta_log.exists():
        return []

    with open(ruta_log, 'r', encoding='utf-8') as f:
        log = json.load(f)

    ejecuciones = log.get('ejecuciones', [])

    if notebook:
        ejecuciones = [e for e in ejecuciones if e.get('notebook') == notebook]

    return ejecuciones[-ultimos:]


def imprimir_historial(notebook: str = None, ultimos: int = 10) -> None:
    """Imprime el historial de ejecuciones de forma legible."""
    historial = obtener_historial(notebook, ultimos)

    log_info(f"{'=' * 70}")
    log_info(f"📋 HISTORIAL DE EJECUCIONES" + (f" — {notebook}" if notebook else ""))
    log_info(f"{'=' * 70}")

    if not historial:
        log_info("  No hay ejecuciones registradas")
    else:
        for e in historial:
            estado_emoji = "✅" if e.get('estado') == 'completado' else "❌"
            fecha = datetime.fromisoformat(e.get('inicio', '')).strftime('%d/%m/%Y %H:%M')
            duracion = _formatear_duracion(e.get('duracion_segundos', 0))
            log_info(f"  {estado_emoji} {e.get('notebook', 'N/A')} | {fecha} | {duracion}")

    log_info(f"{'=' * 70}")


def ultima_ejecucion(notebook: str) -> Optional[Dict]:
    """Devuelve la última ejecución de un notebook, o None."""
    historial = obtener_historial(notebook, ultimos=1)
    return historial[0] if historial else None


def fue_ejecutado_hoy(notebook: str) -> bool:
    """Comprueba si un notebook fue ejecutado hoy."""
    ultima = ultima_ejecucion(notebook)
    if not ultima:
        return False
    fecha_ejecucion = datetime.fromisoformat(ultima.get('inicio', ''))
    return fecha_ejecucion.date() == datetime.now().date()


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def _formatear_duracion(segundos: float) -> str:
    """Formatea duración en formato legible (ej: '2m 15s')."""
    if segundos < 60:
        return f"{segundos:.1f}s"
    elif segundos < 3600:
        minutos = int(segundos // 60)
        segs = int(segundos % 60)
        return f"{minutos}m {segs}s"
    else:
        horas = int(segundos // 3600)
        minutos = int((segundos % 3600) // 60)
        return f"{horas}h {minutos}m"
