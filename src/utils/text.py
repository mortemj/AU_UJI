# ============================================================================
# TEXT.PY - FUNCIONES DE TRANSFORMACIÓN DE TEXTO
# ============================================================================
# TFM: Predicción de Abandono Universitario
# Autora: María José Morte
# ============================================================================
# Funciones para transformar y estandarizar texto en columnas de DataFrames.
# ============================================================================

import re
import unicodedata
import pandas as pd


def convertir_a_snake_case(nombre: str) -> str:
    """
    Convierte un nombre de columna a snake_case.
    
    Transforma cualquier texto a formato snake_case:
    - Minúsculas
    - Sin tildes ni caracteres especiales
    - Espacios y símbolos reemplazados por guión bajo
    - Sin guiones bajos múltiples ni al inicio/final
    
    Parameters
    ----------
    nombre : str
        Nombre original de la columna.
        
    Returns
    -------
    str
        Nombre en formato snake_case.
        
    Examples
    --------
    >>> convertir_a_snake_case('Per_id_Ficticio')
    'per_id_ficticio'
    
    >>> convertir_a_snake_case('Curso Aca')
    'curso_aca'
    
    >>> convertir_a_snake_case('Créd_Titulación')
    'cred_titulacion'
    
    >>> convertir_a_snake_case('Año Nacimiento')
    'anio_nacimiento'
    
    >>> convertir_a_snake_case('  Espacios   Múltiples  ')
    'espacios_multiples'
    """
    # Normalizar caracteres unicode (quitar tildes)
    nombre = unicodedata.normalize('NFKD', nombre)
    nombre = nombre.encode('ASCII', 'ignore').decode('ASCII')
    
    # Convertir a minúsculas
    nombre = nombre.lower()
    
    # Reemplazar espacios y caracteres especiales por guión bajo
    nombre = re.sub(r'[^a-z0-9]', '_', nombre)
    
    # Eliminar guiones bajos múltiples
    nombre = re.sub(r'_+', '_', nombre)
    
    # Eliminar guiones bajos al inicio y final
    nombre = nombre.strip('_')
    
    return nombre


def estandarizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza los nombres de columnas de un DataFrame a snake_case.
    
    Aplica convertir_a_snake_case a todos los nombres de columnas.
    Útil para homogeneizar columnas de diferentes fuentes.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas a estandarizar.
        
    Returns
    -------
    pd.DataFrame
        DataFrame con columnas en snake_case.
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'Nombre Alumno': [1], 'Año Nac': [2000]})
    >>> df_std = estandarizar_columnas(df)
    >>> list(df_std.columns)
    ['nombre_alumno', 'anio_nac']
    """
    # Crear diccionario de mapeo
    mapeo = {col: convertir_a_snake_case(col) for col in df.columns}
    
    # Renombrar columnas
    return df.rename(columns=mapeo)


def limpiar_texto(texto: str) -> str:
    """
    Limpia un texto eliminando espacios extra y normalizando.
    
    Parameters
    ----------
    texto : str
        Texto a limpiar.
        
    Returns
    -------
    str
        Texto limpio.
        
    Examples
    --------
    >>> limpiar_texto('  Hola   mundo  ')
    'Hola mundo'
    
    >>> limpiar_texto('MAYÚSCULAS')
    'MAYÚSCULAS'
    """
    if not isinstance(texto, str):
        return str(texto) if texto is not None else ''
    
    # Eliminar espacios al inicio y final
    texto = texto.strip()
    
    # Reemplazar múltiples espacios por uno solo
    texto = re.sub(r'\s+', ' ', texto)
    
    return texto


def capitalizar_nombre(nombre: str) -> str:
    """
    Capitaliza un nombre propio correctamente.
    
    Maneja preposiciones y artículos españoles (de, del, la, las, etc.)
    
    Parameters
    ----------
    nombre : str
        Nombre a capitalizar.
        
    Returns
    -------
    str
        Nombre capitalizado.
        
    Examples
    --------
    >>> capitalizar_nombre('GARCÍA DE LA FUENTE')
    'García de la Fuente'
    
    >>> capitalizar_nombre('maría josé')
    'María José'
    """
    if not isinstance(nombre, str):
        return str(nombre) if nombre is not None else ''
    
    # Palabras que van en minúscula
    minusculas = {'de', 'del', 'la', 'las', 'los', 'el', 'y', 'e', 'i', 'o', 'u'}
    
    palabras = nombre.lower().split()
    resultado = []
    
    for i, palabra in enumerate(palabras):
        # Primera palabra siempre en mayúscula
        if i == 0:
            resultado.append(palabra.capitalize())
        elif palabra in minusculas:
            resultado.append(palabra)
        else:
            resultado.append(palabra.capitalize())
    
    return ' '.join(resultado)


def truncar_texto(texto: str, max_chars: int = 50, sufijo: str = '...') -> str:
    """
    Trunca un texto a un máximo de caracteres.
    
    Parameters
    ----------
    texto : str
        Texto a truncar.
    max_chars : int, optional
        Máximo de caracteres. Por defecto 50.
    sufijo : str, optional
        Sufijo a añadir si se trunca. Por defecto '...'.
        
    Returns
    -------
    str
        Texto truncado.
        
    Examples
    --------
    >>> truncar_texto('Este es un texto muy largo', max_chars=15)
    'Este es un t...'
    
    >>> truncar_texto('Corto', max_chars=50)
    'Corto'
    """
    if not isinstance(texto, str):
        return str(texto) if texto is not None else ''
    
    if len(texto) <= max_chars:
        return texto
    
    return texto[:max_chars - len(sufijo)] + sufijo
