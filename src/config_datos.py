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
    'EX': 'Ciencias Experimentales',
    'TE': 'Ingeniería y Arquitectura',
    'SA': 'Ciencias de la Salud'
}
# Nombres completos de universidades de origen (códigos → nombres)
UNIVERSIDAD_ORIGEN_NOMBRES: Dict[str, str] = {
    'UJI': 'Universitat Jaume I',
    'UV':  'Universitat de València',
    'UPV': 'Universitat Politècnica de València',
    'UMH': 'Universidad Miguel Hernández de Elche',
    'UA':  'Universidad de Alicante',
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
    # --- Notas y créditos ---
    'cred_superados_anio_1er': 'Créd. superados 1er año',
    'cred_repetidos':          'Créd. repetidos',
    'tasa_repeticion':         'Tasa de repetición',
    'nota_1er_anio':           'Nota 1er año',
    'nota_acceso':             'Nota acceso',
    'nota_selectividad':       'Nota selectividad',

    # --- Identificación / categóricas ---
    'titulacion':              'Titulación',
    'rama':                    'Rama',
    'sexo':                    'Sexo',
    'edad_entrada':            'Edad entrada',
    'pais_nombre':             'País',
    'provincia':               'Provincia',
    'via_acceso':              'Vía acceso',
    'cupo':                    'Cupo',
    'orden_preferencia':       'Orden preferencia',
    'universidad_origen':      'Univ. origen',

    # --- Beca y situación económica ---
    'tuvo_beca':               'Tuvo beca',
    'n_anios_beca':            'Años con beca',
    'anios_sin_beca':          'Años sin beca',
    'pago_fraccionado':        'Pago fraccionado',
    'forma_pago':              'Forma de pago',
    'max_pagos':               'Máx. pagos',

    # --- Trabajo y trayectoria ---
    'situacion_laboral':       'Situación laboral',
    'n_anios_trabajando':      'Años trabajando',
    'n_anios_sin_notas':       'Años sin notas',
    'indicador_interrupcion':  'Interrupción',
    'anios_gap':               'Años gap',

    # --- Métricas agregadas / target ---
    'tasa_abandono_titulacion': 'Tasa aband. titulación',
    'abandono':                 'Abandono',
}


# ============================================================================
# 7. MAPAS DE ENCODING — Variables categóricas a numéricas
# ============================================================================
# FUENTE ÚNICA DE VERDAD para todos los encodings del proyecto.
# Usado en:
#   - notebooks/fase3/f3_m04a_automl_target.ipynb
#   - app/config_app.py (importa desde aquí)
#
# Regla: si un valor cambia de nombre entre años (ej: 'Pruebas acceso
# Bachiller Logse' → 'Bachillerato / PAU'), se añade AMBAS entradas
# apuntando al mismo código. Así el mapa es robusto a cambios históricos.
#
# Valores no mapeados → fillna(0) → código 0 = 'Sin datos / otro'
# ============================================================================

# --- Vía de acceso ---
# Fuente: df_expediente_base.parquet, campo via_acceso
# Incluye variantes históricas de nombre (mismo concepto, distinto texto)
VIA_ACCESO_MAP: dict = {
    # Bachillerato / PAU (nombre antiguo y nuevo)
    'Pruebas acceso Bachiller Logse':              10,
    'Bachillerato / PAU':                          10,
    # FP Grado Superior
    'Ciclo Formativo de Grado sup. o equivalente':  5,
    'FP Grado Superior':                            5,
    # Titulados universitarios
    'Titulados Universitarios':                     4,
    'Titulados universitarios':                     4,
    # Mayores de 25
    'Pruebas acceso mayores 25 años':               7,
    'Pruebas acceso mayores 25años':                7,
    'Mayores de 25 años':                           7,
    # Mayores de 40
    'Pruebas acceso mayores 40 años':              13,
    'Pruebas acceso mayores 40años':               13,
    'Mayores de 40 años':                          13,
    # Mayores de 45
    'Pruebas acceso mayores 45 años':              12,
    'Pruebas acceso mayores 45años':               12,
    'Mayores de 45 años':                          12,
    # Extranjeros
    'Extranjeros CEE':                             11,
    'Extranjeros (UE)':                            11,
    'Extranjeros no CEE':                           6,
    'Extranjeros (fuera UE)':                       6,
    # Otras vías (traslados, adaptaciones, cambios de plan, cupos especiales)
    'Adaptación a Grado':                           3,
    'Traslado':                                     2,
    'Cambio de plan':                               2,
    'Minusválidos':                                 1,
    'Deportistas de élite':                         1,
    # Sin datos
    'Sin datos':                                    0,
    'Sin datos / otro':                             0,
}

# --- Rama de conocimiento ---
# Fuente: rama (abreviatura de 2 letras)
RAMA_MAP: dict = {
    'TE': 1,  # Ingeniería y Arquitectura
    'HU': 2,  # Artes y Humanidades
    'SO': 3,  # Ciencias Sociales y Jurídicas
    'SA': 4,  # Ciencias de la Salud
    'EX': 5,  # Ciencias Experimentales
}

# --- Sexo ---
SEXO_MAP: dict = {
    'Mujer':  0,
    'Hombre': 1,
}

# --- Provincia ---
# Solo las más frecuentes tienen código propio; resto → 0
PROVINCIA_MAP: dict = {
    'Castelló':    1,
    'València':    2,
    'Alacant':     3,
    'Tarragona':   4,
    'Terol':       5,
    # Todo lo demás → 0 (fillna)
}

# --- País de nacionalidad (agrupado por región geográfica) ---
# Fuente: pais_nombre — valores reales del dataset
PAIS_NOMBRE_MAP: dict = {
    # España
    'España': 1,
    # Europa (UE + asociados)
    'Rumania': 2, 'Italia': 2, 'Bulgaria': 2, 'Francia': 2,
    'Alemania': 2, 'Polonia': 2, 'Portugal': 2, 'Reino Unido': 2,
    'Países Bajos': 2, 'Bélgica': 2, 'Hungría': 2, 'Lituania': 2,
    'Finlandia': 2, 'Suecia': 2, 'Irlanda': 2, 'Noruega': 2,
    'Grecia': 2, 'Suiza': 2, 'Estonia': 2, 'Letonia': 2,
    'Eslovaquia': 2, 'Checa, República': 2, 'Andorra': 2,
    'Albania': 2, 'Serbia': 2, 'Bosnia y Herzegovina': 2,
    # Europa no UE / Este
    'Ucrania': 2, 'Moldavia': 2, 'Rusia': 2, 'Georgia': 2, 'Armenia': 2,
    # América Latina
    'Colombia': 3, 'Venezuela': 3, 'Perú': 3, 'Brasil': 3,
    'Ecuador': 3, 'Argentina': 3, 'Bolivia': 3, 'Uruguay': 3,
    'Chile': 3, 'México': 3, 'Paraguay': 3, 'Honduras': 3,
    'Cuba': 3, 'Haití': 3, 'Salvador, El': 3, 'Panamá': 3,
    'Dominicana, República': 3, 'Costa Rica': 3,
    # América del Norte
    'Estados Unidos de Norteamérica': 3,
    # Asia
    'China': 4, 'Pakistán': 4, 'India': 4, 'Vietnam': 4,
    'Bangladesh': 4, 'Turquía': 4, 'Japón': 4, 'Filipinas': 4,
    'Irán': 4, 'Jordania': 4, 'Líbano': 4, 'Siria': 4,
    # África / Oriente Medio
    'Marruecos': 5, 'Argelia': 5, 'Guinea Ecuatorial': 5,
    'Nigeria': 5, 'Guinea': 5, 'Níger': 5, 'Gambia': 5,
    'Rwanda': 5, 'Ruanda': 5, 'Sáhara Occidental': 5,
    'Senegal': 5, 'São Tomé y Príncipe': 5, 'Sudáfrica': 5,
    'Cisjordania': 5, 'Túnez': 5,
    # Sin datos → 0 (fillna)
}

# --- Universidad de origen ---
# Solo universidades con volumen significativo tienen código propio
# NaN y resto → 0 (sin traslado o universidad desconocida)
UNIVERSIDAD_ORIGEN_MAP: dict = {
    'UJI':  40,
    'UV':   18,
    'UPV':  27,
    'UA':    1,
    'UMH':  55,
    # NaN y resto → 0 (fillna)
}

# --- Situación laboral ---
# Fuente: situacion_laboral — valores reales del dataset
# Agrupados en 4 categorías: 0=sin datos, 1=inactivo, 2=parcial, 3=completo/cualif
SITUACION_LABORAL_MAP: dict = {
    'Inactivo o desempleado':                                                    1,
    'No trabaja (inactivo/desempleado)':                                         1,
    'Trabajadores no calificados':                                               2,
    'Trabajadores de los servicios de restauración, personales, protección y vendedores de los comercios': 2,
    'Empleados de tipo administrativo':                                          2,
    'Operadores de instalaciones y maquinaria y montadores':                     2,
    'Artesanos y trabajadores calificados de las industrias manufactureras, la construcción y la minería, excepto los operadores de instalaciones y maquinaria.': 2,
    'Trabajadores calificados en la agricultura y la pesca':                     2,
    'Trabaja a tiempo parcial':                                                  2,
    'Técnicos y Profesionales de apoyo':                                         3,
    'Técnicos y Profesionales científicos e intelectuales':                      3,
    'Dirección de Empresas y de las Administraciones Públicas':                  3,
    'Fuerzas Armadas':                                                           3,
    'Trabaja a tiempo completo':                                                 3,
    # NaN y resto → 0 (fillna)
}

# --- Cupo de acceso ---
# General es la mayoría (27.844). Cupos especiales agrupados en 1.
# NaN → 0 (alumnos sin preinscripción, ya tratados en M01)
CUPO_MAP: dict = {
    'General':               1,
    'Mayor 25 Años':         2,
    'Titulados':             3,
    'Mayor 40años':          4,
    'Mayor 40 Años':         4,
    'Mayor 45años':          5,
    'Mayor 45 Años':         5,
    'Minusvalidos':          6,
    'Diversidad funcional':  6,
    'Deportistas Alto Nivel': 7,
    # NaN → 0 (fillna)
}

# --- Egresado (solo para M04a — leakage, M05 elimina) ---
EGRESADO_MAP: dict = {
    'S': 1,
    'N': 0,
}

# --- Diccionarios inversos (código → etiqueta, para gráficos y app) ---
VIA_ACCESO_INV:         dict = {10: 'Bachillerato/PAU', 5: 'FP Superior',
                                 4: 'Titulado Univ.', 7: 'Mayor 25', 13: 'Mayor 40',
                                 12: 'Mayor 45', 11: 'Extranjero UE', 6: 'Extranjero no UE',
                                 3: 'Adaptación/Traslado', 2: 'Traslado/Cambio plan',
                                 1: 'Cupo especial', 0: 'Sin datos'}
RAMA_INV:               dict = {v: k for k, v in RAMA_MAP.items()}
SEXO_INV:               dict = {v: k for k, v in SEXO_MAP.items()}
SITUACION_LABORAL_INV:  dict = {0: 'Sin datos', 1: 'Inactivo', 2: 'Trabaja parcial', 3: 'Trabaja completo/cualif'}
UNIVERSIDAD_ORIGEN_INV: dict = {v: k for k, v in UNIVERSIDAD_ORIGEN_MAP.items()}
