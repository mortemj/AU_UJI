# ============================================================================
# SCHEMAS.PY — Esquemas Pandera para validación de DataFrames
# ============================================================================
# TFM: Predicción de Abandono Universitario
#
# Los esquemas se generan DINÁMICAMENTE a partir de DICCIONARIO_COLUMNAS
# y TABLAS_INFO definidos en config_datos.py.
#
# ¿Para qué sirve esto?
#   Pandera valida que los DataFrames tengan las columnas correctas,
#   con los tipos de datos esperados y dentro de los rangos válidos.
#   Si algo falla, da un error claro en vez de un fallo silencioso.
#
# Uso:
#   from src.schemas import ESQUEMAS, ESQUEMA_EXPEDIENTES
#   ESQUEMA_EXPEDIENTES.validate(df_expedientes)  # lanza error si falla
#
# NOTA: Antes importaba de src.constants (borrado).
#       Ahora importa de src.config_datos (fuente única).
# ============================================================================

import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema

# CORREGIDO: importar de config_datos (antes era constants.py, ya borrado)
from src.config_datos import (
    DICCIONARIO_COLUMNAS,
    TABLAS_INFO,
)


# ============================================================================
# 1. FUNCIÓN AUXILIAR: CONSTRUIR ESQUEMA PARA UNA TABLA
# ============================================================================

def construir_esquema(nombre_tabla: str) -> DataFrameSchema:
    """
    Construye un esquema Pandera dinámicamente a partir de DICCIONARIO_COLUMNAS.

    Lee las columnas definidas para la tabla en TABLAS_INFO, y para cada una
    genera las restricciones (tipo, nullable, rango, valores permitidos)
    a partir de DICCIONARIO_COLUMNAS.

    Parameters
    ----------
    nombre_tabla : str
        Nombre de la tabla (clave en TABLAS_INFO).
        Ej: 'expedientes', 'demograficos', 'notas'

    Returns
    -------
    DataFrameSchema
        Esquema Pandera listo para validar con .validate(df)

    Raises
    ------
    ValueError
        Si la tabla o alguna columna no está definida en los diccionarios

    Example
    -------
    >>> esquema = construir_esquema('expedientes')
    >>> esquema.validate(df_expedientes)  # OK si pasa, error si falla
    """
    if nombre_tabla not in TABLAS_INFO:
        raise ValueError(f"Tabla desconocida: {nombre_tabla}")

    columnas_info = TABLAS_INFO[nombre_tabla]["columnas"]
    columnas_schema = {}

    for col in columnas_info:
        if col not in DICCIONARIO_COLUMNAS:
            raise ValueError(f"Columna '{col}' no está definida en DICCIONARIO_COLUMNAS")

        info = DICCIONARIO_COLUMNAS[col]

        tipo = info.get("tipo", object)
        nullable = info.get("nullable", True)
        descripcion = info.get("descripcion", "")

        # Checks dinámicos según lo definido en el diccionario
        checks = []

        if "rango" in info:
            min_val, max_val = info["rango"]
            checks.append(Check.between(min_val, max_val))

        if "valores" in info:
            checks.append(Check.isin(info["valores"]))

        columnas_schema[col] = Column(
            tipo,
            nullable=nullable,
            checks=checks if checks else None,
            description=descripcion
        )

    return DataFrameSchema(
        columns=columnas_schema,
        strict=False,  # Permite columnas extra no definidas
        name=nombre_tabla,
        description=TABLAS_INFO[nombre_tabla].get("descripcion", "")
    )


# ============================================================================
# 2. ESQUEMAS GENERADOS AUTOMÁTICAMENTE
# ============================================================================
# Se generan al importar este módulo. Si cambias TABLAS_INFO o
# DICCIONARIO_COLUMNAS en config_datos.py, los esquemas se actualizan solos.

ESQUEMAS = {
    nombre: construir_esquema(nombre)
    for nombre in TABLAS_INFO.keys()
}

# Accesos directos para las tablas más usadas
ESQUEMA_EXPEDIENTES = ESQUEMAS.get("expedientes")
ESQUEMA_TITULACIONES = ESQUEMAS.get("titulaciones")
ESQUEMA_DEMOGRAFICOS = ESQUEMAS.get("demograficos")
ESQUEMA_NOTAS = ESQUEMAS.get("notas")
