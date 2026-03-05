# ============================================================================
# FORMATTERS.PY - FUNCIONES DE FORMATO ESPAÑOL
# ============================================================================
# TFM: Predicción de Abandono Universitario
# Autora: María José Morte
# ============================================================================
# Funciones para formatear números, porcentajes y fechas al estilo español.
# España usa: punto para miles (1.234), coma para decimales (3,14).
# ============================================================================

from typing import Optional, Union
from datetime import datetime
import numpy as np
import pandas as pd


# Constantes de formato español
DECIMAL_SEP: str = ','
MILES_SEP: str = '.'
FORMATO_FECHA: str = '%d/%m/%Y'
FORMATO_DATETIME: str = '%d/%m/%Y %H:%M:%S'


def formato_numero_es(
    numero: Optional[Union[int, float]], 
    decimales: int = 0
) -> str:
    """
    Formatea un número al estilo español (punto miles, coma decimal).
    
    Parameters
    ----------
    numero : int, float o None
        Número a formatear. Si es None o NaN, devuelve '-'.
    decimales : int, optional
        Cantidad de decimales a mostrar. Por defecto 0.
        
    Returns
    -------
    str
        Número formateado como string.
        
    Examples
    --------
    >>> formato_numero_es(1234567)
    '1.234.567'
    
    >>> formato_numero_es(1234567.89, decimales=2)
    '1.234.567,89'
    
    >>> formato_numero_es(1234.5, decimales=1)
    '1.234,5'
    
    >>> formato_numero_es(None)
    '-'
    
    >>> formato_numero_es(float('nan'))
    '-'
    """
    # Si es nulo o NaN, devolver guion
    if numero is None:
        return '-'
    if isinstance(numero, float) and (np.isnan(numero) or np.isinf(numero)):
        return '-'
    
    try:
        # Si no queremos decimales o es entero exacto
        if decimales == 0 or (isinstance(numero, float) and numero == int(numero)):
            # Formatear como entero con separador de miles
            resultado = f'{int(numero):,}'.replace(',', MILES_SEP)
        else:
            # Formatear con decimales
            resultado = f'{numero:,.{decimales}f}'
            # Cambiar separadores: coma→@ luego punto→coma luego @→punto
            resultado = resultado.replace(',', '@').replace('.', DECIMAL_SEP).replace('@', MILES_SEP)
        return resultado
    except (ValueError, TypeError):
        # Si hay error, devolver el valor como string
        return str(numero)


def formato_porcentaje_es(
    numero: Optional[Union[int, float]], 
    decimales: int = 1
) -> str:
    """
    Formatea un número como porcentaje al estilo español.
    
    Parameters
    ----------
    numero : int, float o None
        Número a formatear (ya debe estar en escala 0-100).
        Si es None o NaN, devuelve '-'.
    decimales : int, optional
        Cantidad de decimales a mostrar. Por defecto 1.
        
    Returns
    -------
    str
        Porcentaje formateado como string con símbolo %.
        
    Examples
    --------
    >>> formato_porcentaje_es(85.5)
    '85,5%'
    
    >>> formato_porcentaje_es(100)
    '100,0%'
    
    >>> formato_porcentaje_es(33.333, decimales=2)
    '33,33%'
    
    >>> formato_porcentaje_es(None)
    '-'
    """
    # Si es nulo o NaN, devolver guion
    if numero is None:
        return '-'
    if isinstance(numero, float) and (np.isnan(numero) or np.isinf(numero)):
        return '-'
    
    try:
        # Formatear con decimales y cambiar punto por coma
        resultado = f'{numero:.{decimales}f}'.replace('.', DECIMAL_SEP)
        return f'{resultado}%'
    except (ValueError, TypeError):
        return str(numero)


def formato_fecha_es(
    fecha: Optional[Union[str, datetime, pd.Timestamp]],
    incluir_hora: bool = False
) -> str:
    """
    Formatea una fecha al estilo español (dd/mm/yyyy).
    
    Parameters
    ----------
    fecha : str, datetime, Timestamp o None
        Fecha a formatear.
    incluir_hora : bool, optional
        Si True, incluye hora (dd/mm/yyyy HH:MM:SS). Por defecto False.
        
    Returns
    -------
    str
        Fecha formateada como string.
        
    Examples
    --------
    >>> from datetime import datetime
    >>> formato_fecha_es(datetime(2024, 12, 7))
    '07/12/2024'
    
    >>> formato_fecha_es(datetime(2024, 12, 7, 15, 30), incluir_hora=True)
    '07/12/2024 15:30:00'
    
    >>> formato_fecha_es(None)
    '-'
    
    >>> formato_fecha_es('2024-12-07')
    '07/12/2024'
    """
    if fecha is None or pd.isna(fecha):
        return '-'
    
    try:
        # Convertir a datetime si es string
        if isinstance(fecha, str):
            fecha = pd.to_datetime(fecha)
        
        # Elegir formato según si incluye hora
        formato = FORMATO_DATETIME if incluir_hora else FORMATO_FECHA
        return fecha.strftime(formato)
    except Exception:
        return str(fecha)


def formato_moneda_es(
    numero: Optional[Union[int, float]],
    decimales: int = 2,
    simbolo: str = '€'
) -> str:
    """
    Formatea un número como moneda al estilo español.
    
    Parameters
    ----------
    numero : int, float o None
        Cantidad a formatear.
    decimales : int, optional
        Cantidad de decimales. Por defecto 2.
    simbolo : str, optional
        Símbolo de moneda. Por defecto '€'.
        
    Returns
    -------
    str
        Moneda formateada como string.
        
    Examples
    --------
    >>> formato_moneda_es(1234.56)
    '1.234,56 €'
    
    >>> formato_moneda_es(1000000)
    '1.000.000,00 €'
    
    >>> formato_moneda_es(99.9, simbolo='$')
    '99,90 $'
    """
    if numero is None:
        return '-'
    if isinstance(numero, float) and (np.isnan(numero) or np.isinf(numero)):
        return '-'
    
    try:
        numero_formateado = formato_numero_es(numero, decimales=decimales)
        return f'{numero_formateado} {simbolo}'
    except (ValueError, TypeError):
        return str(numero)


def formato_miles_es(numero: Optional[Union[int, float]]) -> str:
    """
    Formatea un número grande de forma compacta (K, M, B).
    
    Parameters
    ----------
    numero : int, float o None
        Número a formatear.
        
    Returns
    -------
    str
        Número formateado de forma compacta.
        
    Examples
    --------
    >>> formato_miles_es(1500)
    '1,5K'
    
    >>> formato_miles_es(2500000)
    '2,5M'
    
    >>> formato_miles_es(500)
    '500'
    """
    if numero is None:
        return '-'
    if isinstance(numero, float) and (np.isnan(numero) or np.isinf(numero)):
        return '-'
    
    try:
        abs_numero = abs(numero)
        signo = '-' if numero < 0 else ''
        
        if abs_numero >= 1_000_000_000:
            valor = abs_numero / 1_000_000_000
            return f'{signo}{valor:.1f}B'.replace('.', DECIMAL_SEP)
        elif abs_numero >= 1_000_000:
            valor = abs_numero / 1_000_000
            return f'{signo}{valor:.1f}M'.replace('.', DECIMAL_SEP)
        elif abs_numero >= 1_000:
            valor = abs_numero / 1_000
            return f'{signo}{valor:.1f}K'.replace('.', DECIMAL_SEP)
        else:
            return formato_numero_es(numero, decimales=0)
    except (ValueError, TypeError):
        return str(numero)
