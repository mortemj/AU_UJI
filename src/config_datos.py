# ============================================================================
# CONFIG_DATOS.PY — Diccionarios y constantes de los datos
# ============================================================================
# TFM: Predicción de Abandono Universitario
#
# Contiene toda la información sobre la ESTRUCTURA de los datos:
#   - TABLAS_INFO: metadatos de cada tabla (columnas, claves, filas esperadas)
#   - DICCIONARIO_COLUMNAS: descripción y tipo de cada variable
#   - Diccionarios de códigos (ramas, formas de pago, provincias)
#   - Valores nulos y especiales
#   - Constantes de referencia temporal
#
# ¿Quién lo usa?
#   - Los notebooks de limpieza (Fase 1) para saber qué columnas esperar
#   - Los esquemas Pandera (schemas.py) para validar DataFrames
#   - La documentación HTML para mostrar metadatos
#
# NOTA: Antes estos datos estaban repartidos entre constants.py,
# config_datos.py y validation.py. Ahora está todo aquí.
# ============================================================================

from typing import Dict, Any, List


# ============================================================================
# 1. INFORMACIÓN DE TABLAS
# ============================================================================
# Cada tabla del proyecto con sus metadatos.
# 'filas_esperadas' es orientativo (puede variar ligeramente entre ejecuciones).

TABLAS_INFO: Dict[str, Dict[str, Any]] = {
    'titulaciones': {
        'descripcion': 'Catálogo de titulaciones ofertadas',
        'clave_primaria': ['exp_tit_id'],
        'filas_esperadas': 108,
        'hoja_excel': 'Titulaciones',
        'columnas': ['exp_tit_id', 'titulacion', 'rama', 'cred_titulacion'],
    },
    'expedientes': {
        'descripcion': 'Información académica de cada alumno por titulación',
        'clave_primaria': ['per_id_ficticio', 'exp_tit_id'],
        'filas_esperadas': 109575,
        'hoja_excel': 'Expedientes',
        'columnas': [
            'per_id_ficticio', 'exp_tit_id', 'curso_aca_ini',
            'nota', 'cred_matriculados', 'cred_superados',
            'egresado', 'nuevo',
        ],
    },
    'demograficos': {
        'descripcion': 'Datos demográficos del alumno (sexo, nacionalidad, nacimiento)',
        'clave_primaria': ['per_id_ficticio'],
        'filas_esperadas': 30873,
        'hoja_excel': 'Nac-Sexo_Nacionalidad',
        'columnas': [
            'per_id_ficticio', 'sexo', 'anio_nacimiento',
            'pais_nombre', 'provincia', 'poblacion',
        ],
    },
    'domicilios': {
        'descripcion': 'Dirección del alumno durante el curso',
        'clave_primaria': ['per_id_ficticio', 'curso_aca'],
        'filas_esperadas': 210911,
        'hoja_excel': 'Domicilios',
        'columnas': [
            'per_id_ficticio', 'curso_aca',
            'poblacion', 'provincia', 'pais',
        ],
    },
    'becas': {
        'descripcion': 'Situación de becas del alumno',
        'clave_primaria': ['per_id_ficticio', 'mat_curso_aca'],
        'filas_esperadas': 70524,
        'hoja_excel': 'Circunstancias',
        'columnas': [
            'per_id_ficticio', 'mat_curso_aca',
            'nombre_beca',
        ],
    },
    'trabajo': {
        'descripcion': 'Situación laboral del alumno',
        'clave_primaria': ['per_id_ficticio', 'mat_curso_aca'],
        'filas_esperadas': 195524,
        'hoja_excel': 'Trabajo',
        'columnas': [
            'per_id_ficticio', 'mat_curso_aca',
            'nombre_trabajo',
        ],
    },
    'notas': {
        'descripcion': 'Calificaciones medias por curso',
        'clave_primaria': ['per_id_ficticio', 'curso_aca', 'exp_tit_id'],
        'filas_esperadas': 107908,
        'hoja_excel': 'Notas',
        'columnas': [
            'per_id_ficticio', 'curso_aca', 'exp_tit_id',
            'media_curso',
        ],
    },
    'recibos': {
        'descripcion': 'Información de pagos y matrículas',
        'clave_primaria': ['per_id_ficticio', 'curso_aca'],
        'filas_esperadas': 114447,
        'hoja_excel': 'Recibos',
        'columnas': [
            'per_id_ficticio', 'curso_aca',
            'forma_de_pago', 'numero_pagos',
        ],
    },
    'preinscripcion': {
        'descripcion': 'Datos de preinscripción universitaria',
        'clave_primaria': ['per_id_ficticio'],
        'filas_esperadas': 210996,
        'hoja_excel': 'Hoja1',
        'opcional': True,
        'columnas': [
            'per_id_ficticio', 'provincia', 'pais',
        ],
    },
}


# ============================================================================
# 2. DICCIONARIO DE COLUMNAS
# ============================================================================
# Descripción detallada de cada variable del proyecto.
# Se usa para documentación, validación y generación de HTML.

DICCIONARIO_COLUMNAS: Dict[str, Dict[str, Any]] = {
    # --- Identificadores ---
    'per_id_ficticio': {
        'descripcion': 'Identificador único del alumno (anonimizado)',
        'tipo': 'int', 'nullable': False, 'ejemplo': 12345
    },
    'exp_tit_id': {
        'descripcion': 'Código identificador de la titulación',
        'tipo': 'int', 'nullable': False, 'ejemplo': 101
    },

    # --- Datos académicos ---
    'curso_aca': {
        'descripcion': 'Curso académico (año de inicio)',
        'tipo': 'int', 'nullable': False,
        'rango': (2000, 2030), 'ejemplo': 2020
    },
    'curso_aca_ini': {
        'descripcion': 'Curso de inicio en la titulación',
        'tipo': 'int', 'nullable': False, 'ejemplo': 2018
    },
    'curso_aca_fin': {
        'descripcion': 'Curso de finalización (si aplica)',
        'tipo': 'int', 'nullable': True, 'ejemplo': 2022
    },
    'nota': {
        'descripcion': 'Nota de acceso a la universidad',
        'tipo': 'float', 'nullable': True,
        'rango': (0, 14), 'ejemplo': 8.75
    },
    'nota_acceso': {
        'descripcion': 'Nota de acceso (puede diferir de nota selectividad)',
        'tipo': 'float', 'nullable': True,
        'rango': (0, 14), 'ejemplo': 7.50
    },
    'nota_selectividad': {
        'descripcion': 'Nota de selectividad (PAU/EBAU)',
        'tipo': 'float', 'nullable': True,
        'rango': (0, 14), 'ejemplo': 7.85
    },
    'cred_matriculados': {
        'descripcion': 'Créditos matriculados en el curso',
        'tipo': 'int', 'nullable': True,
        'rango': (0, 90), 'ejemplo': 60
    },
    'cred_superados': {
        'descripcion': 'Créditos superados en el curso',
        'tipo': 'int', 'nullable': True,
        'rango': (0, 90), 'ejemplo': 54
    },
    'media_curso': {
        'descripcion': 'Nota media del curso académico',
        'tipo': 'float', 'nullable': True,
        'rango': (0, 10), 'ejemplo': 6.85
    },
    'media_titulacion_curso': {
        'descripcion': 'Media de la titulación en ese curso',
        'tipo': 'float', 'nullable': True,
        'rango': (0, 10), 'ejemplo': 6.50
    },
    'media_titulacion_alumno': {
        'descripcion': 'Media acumulada del alumno en la titulación',
        'tipo': 'float', 'nullable': True,
        'rango': (0, 10), 'ejemplo': 7.20
    },

    # --- Variables objetivo ---
    'egresado': {
        'descripcion': 'Si el alumno ha finalizado la titulación',
        'tipo': 'str', 'nullable': False,
        'valores': ['S', 'N'], 'ejemplo': 'N'
    },
    'nuevo': {
        'descripcion': 'Si es alumno de nuevo ingreso ese curso',
        'tipo': 'str', 'nullable': False,
        'valores': ['S', 'N'], 'ejemplo': 'S'
    },

    # --- Datos demográficos ---
    'sexo': {
        'descripcion': 'Sexo del alumno',
        'tipo': 'str', 'nullable': False,
        'valores': ['H', 'M'], 'ejemplo': 'M'
    },
    'fecha_nacimiento': {
        'descripcion': 'Fecha de nacimiento',
        'tipo': 'datetime', 'nullable': True, 'ejemplo': '1998-05-15'
    },
    'anio_nacimiento': {
        'descripcion': 'Año de nacimiento',
        'tipo': 'int', 'nullable': True,
        'rango': (1950, 2010), 'ejemplo': 1998
    },
    'edad': {
        'descripcion': 'Edad del alumno (calculada a 31/12/2021)',
        'tipo': 'int', 'nullable': True,
        'rango': (16, 80), 'ejemplo': 23
    },
    'pais_nombre': {
        'descripcion': 'País de nacionalidad',
        'tipo': 'str', 'nullable': True, 'ejemplo': 'España'
    },

    # --- Ubicación ---
    'poblacion': {
        'descripcion': 'Municipio de residencia',
        'tipo': 'str', 'nullable': True, 'ejemplo': 'Castelló de la Plana'
    },
    'provincia': {
        'descripcion': 'Provincia de residencia',
        'tipo': 'str', 'nullable': True, 'ejemplo': 'Castellón'
    },
    'pais': {
        'descripcion': 'País de residencia',
        'tipo': 'str', 'nullable': True, 'ejemplo': 'España'
    },

    # --- Situación personal ---
    'nombre_beca': {
        'descripcion': 'Tipo de beca recibida',
        'tipo': 'str', 'nullable': True, 'ejemplo': 'Becario'
    },
    'nombre_trabajo': {
        'descripcion': 'Situación laboral del alumno',
        'tipo': 'str', 'nullable': True, 'ejemplo': 'Inactivo o desempleado'
    },

    # --- Titulación ---
    'titulacion': {
        'descripcion': 'Nombre de la titulación',
        'tipo': 'str', 'nullable': False,
        'ejemplo': 'Grado en Ingeniería Informática'
    },
    'rama': {
        'descripcion': 'Rama de conocimiento (código)',
        'tipo': 'str', 'nullable': True,
        'valores': ['SO', 'HU', 'EX', 'TE', 'SA'], 'ejemplo': 'TE'
    },
    'cred_titulacion': {
        'descripcion': 'Créditos totales de la titulación',
        'tipo': 'int', 'nullable': True,
        'rango': (180, 360), 'ejemplo': 240
    },

    # --- Pagos ---
    'forma_de_pago': {
        'descripcion': 'Forma de pago de la matrícula (código)',
        'tipo': 'str', 'nullable': True,
        'valores': ['D', 'N', 'T'], 'ejemplo': 'D'
    },
    'numero_pagos': {
        'descripcion': 'Número de plazos de pago',
        'tipo': 'int', 'nullable': True,
        'rango': (1, 12), 'ejemplo': 2
    },
}


# ============================================================================
# 3. DICCIONARIOS DE CÓDIGOS
# ============================================================================

DICCIONARIO_RAMAS: Dict[str, str] = {
    'SO': 'Ciencias Sociales y Jurídicas',
    'HU': 'Artes y Humanidades',
    'EX': 'Ciencias',
    'TE': 'Ingeniería y Arquitectura',
    'SA': 'Ciencias de la Salud'
}

DICCIONARIO_FORMA_PAGO: Dict[str, str] = {
    'D': 'Domiciliación bancaria',
    'N': 'Pago en efectivo',
    'T': 'Transferencia'
}

DICCIONARIO_CP_PROVINCIA: Dict[str, str] = {
    '01': 'Álava', '02': 'Albacete', '03': 'Alicante', '04': 'Almería',
    '05': 'Ávila', '06': 'Badajoz', '07': 'Islas Baleares', '08': 'Barcelona',
    '09': 'Burgos', '10': 'Cáceres', '11': 'Cádiz', '12': 'Castellón',
    '13': 'Ciudad Real', '14': 'Córdoba', '15': 'La Coruña', '16': 'Cuenca',
    '17': 'Gerona', '18': 'Granada', '19': 'Guadalajara', '20': 'Guipúzcoa',
    '21': 'Huelva', '22': 'Huesca', '23': 'Jaén', '24': 'León',
    '25': 'Lérida', '26': 'La Rioja', '27': 'Lugo', '28': 'Madrid',
    '29': 'Málaga', '30': 'Murcia', '31': 'Navarra', '32': 'Orense',
    '33': 'Asturias', '34': 'Palencia', '35': 'Las Palmas', '36': 'Pontevedra',
    '37': 'Salamanca', '38': 'Santa Cruz de Tenerife', '39': 'Cantabria',
    '40': 'Segovia', '41': 'Sevilla', '42': 'Soria', '43': 'Tarragona',
    '44': 'Teruel', '45': 'Toledo', '46': 'Valencia', '47': 'Valladolid',
    '48': 'Vizcaya', '49': 'Zamora', '50': 'Zaragoza', '51': 'Ceuta',
    '52': 'Melilla'
}


# ============================================================================
# 4. VALORES ESPECIALES
# ============================================================================
# Cadenas que representan "sin dato" en los Excel originales.

VALORES_VACIOS: List[str] = [
    '-', '', ' ', 'NA', 'N/A', 'null', 'NULL', 'None', 'nan', 'NaN',
    'n/a', 'na', 'none'
]

VALORES_NULOS: List[str] = [
    'Sin información', 'SIN INFORMACIÓN', 'sin información',
    'No consta', 'NO CONSTA', 'no consta',
] + VALORES_VACIOS


# ============================================================================
# 5. CONSTANTES DE REFERENCIA TEMPORAL
# ============================================================================
# El último curso académico completo en los datos.

CURSO_REFERENCIA: int = 2021
FECHA_REFERENCIA: str = '2021-12-31'


# ============================================================================
# 6. ETIQUETAS LEGIBLES DE VARIABLES (PARA GRAFICOS Y HTML)
# ============================================================================
# Nombres cortos y legibles para usar en ejes, tablas y titulos.
# Cubre las 21 columnas del dataset final (df_eda_final.parquet).
#
# Uso:
#   from src.config import ETIQUETAS_VARIABLES
#   ax.set_ylabel(ETIQUETAS_VARIABLES.get(col, col))

ETIQUETAS_VARIABLES: Dict[str, str] = {
    'cred_superados_anio_1er': 'Créd. superados 1er año',
    'nota_1er_anio': 'Nota 1er año',
    'nota_acceso': 'Nota acceso',
    'titulacion': 'Titulación',
    'rama': 'Rama',
    'sexo': 'Sexo',
    'edad_entrada': 'Edad entrada',
    'pais_nombre': 'País',
    'provincia': 'Provincia',
    'via_acceso': 'Vía acceso',
    'orden_preferencia': 'Orden preferencia',
    'universidad_origen': 'Univ. origen',
    'tuvo_beca': 'Tuvo beca',
    'n_anios_beca': 'Años con beca',
    'situacion_laboral': 'Situación laboral',
    'nota_selectividad': 'Nota selectividad',
    'max_pagos': 'Máx. pagos',
    'indicador_interrupcion': 'Interrupción',
    'anios_gap': 'Años gap',
    'anios_sin_beca': 'Años sin beca',
    'abandono': 'Abandono',
}
