# ============================================================================
# CONSTANTES.PY — Constantes específicas del dominio del proyecto
# ============================================================================
# TFM: Predicción de Abandono Universitario
#
# Este archivo contiene constantes que son específicas de ESTE proyecto
# (no reutilizables en otros proyectos). Cosas como:
#   - Nombres de tablas y sus relaciones
#   - Emojis y colores asignados a cada tabla
#   - Categorías de variables del alumno
#   - Pipeline de la Fase 1
#
# ¿En qué se diferencia de config_datos.py?
#   - config_datos.py → Estructura de datos (columnas, tipos, rangos)
#   - constantes.py → Presentación y lógica de negocio (emojis, colores, pipeline)
# ============================================================================


# ============================================================================
# 1. IDENTIFICADORES DE UNIVERSIDAD
# ============================================================================

UJI_ID = 40  # Código de Universitat Jaume I en preinscripción


# ============================================================================
# 2. NOMBRES DE TABLAS
# ============================================================================

# Tabla de preinscripción (se une en m04b, separada del Excel principal)
TABLA_PREINSCRIPCION = 'preinscripcion'

# Tablas del Excel principal (se unen en m04a)
TABLAS_EXCEL_PRINCIPAL = [
    'expedientes',
    'titulaciones',
    'demograficos',
    'domicilios',
    'becas',
    'trabajo',
    'notas',
    'recibos'
]

# Todas las tablas
TABLAS_TODAS = TABLAS_EXCEL_PRINCIPAL + [TABLA_PREINSCRIPCION]


# ============================================================================
# 3. CLAVES DE MERGE POR TABLA
# ============================================================================
# Se usa en la Fase 1 (módulo m04) para documentar las uniones
# y en los grafos de relaciones entre tablas.

CLAVES_MERGE_TABLAS = {
    'expedientes': 'BASE',
    'titulaciones': 'exp_tit_id',
    'demograficos': 'per_id',
    'domicilios': 'per_id+curso',
    'becas': 'per_id+curso',
    'trabajo': 'per_id+tit+curso',
    'notas': 'per_id+tit+curso',
    'recibos': 'per_id+curso',
    'preinscripcion': 'per_id+tit+curso_ini'
}

# Relaciones entre tablas (para el grafo D3/PyVis)
RELACIONES_TABLAS = [
    ('expedientes', 'titulaciones'),
    ('expedientes', 'notas'),
    ('becas', 'recibos'),
]


# ============================================================================
# 4. EMOJIS Y COLORES POR TABLA
# ============================================================================
# Se usan en los HTML generados y en los notebooks para dar
# identidad visual a cada tabla.

EMOJIS_TABLAS = {
    'expedientes': '📋',
    'titulaciones': '🎓',
    'demograficos': '👤',
    'domicilios': '🏠',
    'becas': '💰',
    'trabajo': '💼',
    'notas': '📝',
    'recibos': '💳',
    'preinscripcion': '📄'
}

COLORES_TABLAS = {
    'expedientes': '#3182ce',
    'titulaciones': '#805ad5',
    'demograficos': '#ed8936',
    'domicilios': '#e53e3e',
    'becas': '#38a169',
    'trabajo': '#319795',
    'notas': '#d69e2e',
    'recibos': '#667eea',
    'preinscripcion': '#ed64a6'
}


# ============================================================================
# 5. CATEGORÍAS DE VARIABLES DEL ALUMNO
# ============================================================================
# Agrupación lógica de las variables para documentación y HTML.

CATEGORIAS_VARIABLES = {
    'demografico': {
        'titulo': 'Demográfico',
        'emoji': '👤',
        'clase_css': 'cat-demografico',
        'variables': ['sexo', 'fecha_nacimiento', 'id_pais', 'pais_nombre']
    },
    'academico': {
        'titulo': 'Académico',
        'emoji': '🎓',
        'clase_css': 'cat-academico',
        'variables': [
            'titulacion', 'rama', 'cred_matriculados', 'cred_superados',
            'nota_selectividad', 'nota_acceso', 'media_titulacion_curso',
            'egresado', 'nuevo'
        ]
    },
    'economico': {
        'titulo': 'Económico',
        'emoji': '💰',
        'clase_css': 'cat-economico',
        'variables': [
            'tiene_beca', 'nombre_beca', 'forma_de_pago',
            'numero_pagos', 'nombre_trabajo'
        ]
    },
    'geografico': {
        'titulo': 'Geográfico',
        'emoji': '🏠',
        'clase_css': 'cat-geografico',
        'variables': ['poblacion', 'provincia', 'pais_domicilio', 'vive_fuera']
    },
    'acceso': {
        'titulo': 'Acceso',
        'emoji': '📋',
        'clase_css': 'cat-acceso',
        'variables': ['via_acceso', 'orden_preferencia', 'cupo', 'universidad_origen']
    },
    'temporal': {
        'titulo': 'Temporal',
        'emoji': '⏱️',
        'clase_css': 'cat-temporal',
        'variables': ['curso_aca', 'curso_aca_ini', 'curso_aca_fin']
    }
}


# ============================================================================
# 6. MAPEO DE COLUMNAS PREINSCRIPCIÓN
# ============================================================================
# Las columnas del Excel de preinscripción tienen nombres distintos
# a los que usamos internamente. Este mapeo se aplica en la Fase 1.

MAPEO_COLUMNAS_PREINSCRIPCION = {
    'via_estudios': 'via_acceso',
    'orden_titulacion': 'orden_preferencia',
    'nom_cupo': 'cupo',
    'universidad': 'universidad_origen'
}

DESCRIPCIONES_COLUMNAS_PREINSCRIPCION = {
    'via_acceso': 'Vía de acceso (Bachiller, FP...)',
    'orden_preferencia': 'Orden de preferencia (1-20)',
    'cupo': 'Tipo de cupo',
    'universidad_origen': 'Universidad más cercana'
}

CLAVES_MERGE_PREINSCRIPCION = ['per_id_ficticio', 'exp_tit_id']
COLUMNAS_NUEVAS_PREINSCRIPCION = ['via_acceso', 'orden_preferencia', 'cupo', 'universidad_origen']


# ============================================================================
# 7. FASE 1: PIPELINE Y TRANSFORMACIONES
# ============================================================================
# Metadatos del pipeline de la Fase 1 para los HTML y dashboards.

PIPELINE_FASE1 = [
    {'id': 'M01', 'nombre': 'Raw', 'emoji': '📋', 'color': '#e53e3e'},
    {'id': 'M02', 'nombre': 'Limpieza', 'emoji': '🧹', 'color': '#ed8936'},
    {'id': 'M03', 'nombre': 'Clean', 'emoji': '✨', 'color': '#ecc94b'},
    {'id': 'M04', 'nombre': 'Final', 'emoji': '🎯', 'color': '#38a169'},
]

TRANSFORMACIONES_FASE1 = [
    {'emoji': '📤', 'nombre': 'Normalización'},
    {'emoji': '🔢', 'nombre': 'Tipos datos'},
    {'emoji': '🌐', 'nombre': 'Codificación'},
    {'emoji': '🔗', 'nombre': 'Unión tablas'},
    {'emoji': '➕', 'nombre': 'Variables derivadas'},
    {'emoji': '✅', 'nombre': 'Validación'},
]

# Colores para KPIs del dashboard
COLORES_KPIS = {
    'tablas': '#3182ce',
    'registros': '#38a169',
    'variables': '#805ad5',
    'alumnos': '#ed8936'
}
