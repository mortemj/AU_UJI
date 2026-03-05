# ============================================================================
# LOG.PY - SISTEMA DE LOGGING
# ============================================================================
# TFM: Predicción de Abandono Universitario
# Autora: María José Morte
# ============================================================================
# Sistema profesional de registro de eventos.
# Guarda en archivo + muestra en consola con emojis.
# ============================================================================

import logging
from pathlib import Path
from typing import Optional


# Variable global para el logger
_logger: Optional[logging.Logger] = None


def configurar_logging(ruta_log: Optional[Path] = None) -> logging.Logger:
    """
    Configura el sistema de logging del proyecto.
    
    Crea un logger que escribe tanto a archivo como a consola.
    Si no se especifica ruta, solo escribe a consola.
    
    Parameters
    ----------
    ruta_log : Path, optional
        Ruta del archivo de log. Si es None, solo consola.
        
    Returns
    -------
    logging.Logger
        Logger configurado listo para usar.
        
    Examples
    --------
    >>> logger = configurar_logging(Path('logs/tfm.log'))
    >>> logger.info("Mensaje de prueba")
    """
    global _logger
    
    # Si ya está configurado, devolverlo
    if _logger is not None:
        return _logger
    
    # Crear logger con nombre del proyecto
    _logger = logging.getLogger('tfm_abandono')
    _logger.setLevel(logging.DEBUG)
    
    # Evitar duplicados si ya tiene handlers
    if _logger.handlers:
        return _logger
    
    # Formato para archivo (más detallado)
    formato_archivo = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para archivo (si se especifica ruta)
    if ruta_log is not None:
        ruta_log.parent.mkdir(parents=True, exist_ok=True)
        handler_archivo = logging.FileHandler(
            ruta_log, 
            encoding='utf-8',
            mode='a'  # Append, no sobrescribir
        )
        handler_archivo.setLevel(logging.DEBUG)
        handler_archivo.setFormatter(formato_archivo)
        _logger.addHandler(handler_archivo)
    
    return _logger


def _get_logger() -> logging.Logger:
    """Obtiene el logger, creándolo si no existe."""
    global _logger
    if _logger is None:
        _logger = configurar_logging()
    return _logger


def log_info(mensaje: str, emoji: str = '✓') -> None:
    """
    Muestra mensaje informativo en consola y lo guarda en log.
    
    Parameters
    ----------
    mensaje : str
        Mensaje a mostrar.
    emoji : str, optional
        Emoji a mostrar antes del mensaje. Por defecto '✓'.
        
    Examples
    --------
    >>> log_info("Archivo cargado correctamente")
    ✓ Archivo cargado correctamente
    
    >>> log_info("Proceso iniciado", "🚀")
    🚀 Proceso iniciado
    """
    print(f"{emoji} {mensaje}")
    _get_logger().info(mensaje)


def log_warning(mensaje: str, emoji: str = '⚠') -> None:
    """
    Muestra advertencia en consola y la guarda en log.
    
    Parameters
    ----------
    mensaje : str
        Mensaje de advertencia.
    emoji : str, optional
        Emoji a mostrar. Por defecto '⚠'.
        
    Examples
    --------
    >>> log_warning("Valor fuera de rango")
    ⚠ Valor fuera de rango
    """
    print(f"{emoji} {mensaje}")
    _get_logger().warning(mensaje)


def log_error(mensaje: str, emoji: str = '❌') -> None:
    """
    Muestra error en consola y lo guarda en log.
    
    Parameters
    ----------
    mensaje : str
        Mensaje de error.
    emoji : str, optional
        Emoji a mostrar. Por defecto '❌'.
        
    Examples
    --------
    >>> log_error("No se pudo abrir el archivo")
    ❌ No se pudo abrir el archivo
    """
    print(f"{emoji} {mensaje}")
    _get_logger().error(mensaje)


def log_debug(mensaje: str) -> None:
    """
    Guarda mensaje de debug en log (no muestra en consola).
    
    Útil para información detallada que no queremos mostrar
    al usuario pero sí guardar para depuración.
    
    Parameters
    ----------
    mensaje : str
        Mensaje de debug.
        
    Examples
    --------
    >>> log_debug("Variable x = 42")
    # No muestra nada en consola, solo guarda en archivo log
    """
    _get_logger().debug(mensaje)


def log_success(mensaje: str) -> None:
    """
    Muestra mensaje de éxito en consola (verde).
    
    Parameters
    ----------
    mensaje : str
        Mensaje de éxito.
        
    Examples
    --------
    >>> log_success("Proceso completado correctamente")
    ✅ Proceso completado correctamente
    """
    log_info(mensaje, emoji='✅')


def log_step(paso: int, total: int, mensaje: str) -> None:
    """
    Muestra progreso de un proceso por pasos.
    
    Parameters
    ----------
    paso : int
        Número del paso actual.
    total : int
        Número total de pasos.
    mensaje : str
        Descripción del paso.
        
    Examples
    --------
    >>> log_step(1, 5, "Cargando datos")
    [1/5] Cargando datos
    """
    print(f"[{paso}/{total}] {mensaje}")
    _get_logger().info(f"Paso {paso}/{total}: {mensaje}")
