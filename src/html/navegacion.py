# ============================================================================
# NAVEGACION.PY - ESTRUCTURA DE FASES Y MÓDULOS DEL PROYECTO
# ============================================================================
# TFM: Predicción de Abandono Universitario
# Autora: María José Morte
# ============================================================================

from typing import Dict, List, Any
import re

# ============================================================================
# ESTRUCTURA DE FASES Y MÓDULOS
# ============================================================================

FASES: Dict[str, Dict[str, Any]] = {
    'inicio': {
        'nombre': 'Inicio',
        'emoji': '🏠',
        'archivo': '../index.html',
        'carpeta': '',
        'estado': 'activo'
    },
    'fase1': {
        'nombre': 'Transformación',
        'emoji': '📥',
        'archivo': 'fase1_index.html',
        'carpeta': 'fase1',
        'estado': 'activo',
        'modulos': [
            {'id': 'indice', 'nombre': 'Índice', 'archivo': 'fase1_index.html', 'emoji': '📋'},
            {'id': 'm01', 'nombre': 'Reportes Raw', 'archivo': 'm01_reportes_raw.html', 'emoji': '📋'},
            {'id': 'm02', 'nombre': 'Limpieza', 'archivo': 'm02_limpieza.html', 'emoji': '🧹'},
            {'id': 'm03', 'nombre': 'Reportes Clean', 'archivo': 'm03_reportes_clean.html', 'emoji': '✨'},
            {'id': 'm04', 'nombre': 'Dataset Final', 'archivo': 'm04_dataset_final.html', 'emoji': '🎯'},
            {'id': 'm05', 'nombre': 'Dashboard', 'archivo': 'm05_dashboard.html', 'emoji': '📊'},
            {'id': 'm06', 'nombre': 'Grafo', 'archivo': 'm06_grafo.html', 'emoji': '🕸️'},
        ]
    },
    'fase2': {
        'nombre': 'EDA Raw',
        'emoji': '📊',
        'archivo': 'fase2_index.html',
        'carpeta': 'fase2',
        'estado': 'activo',
        'modulos': [
            {'id': 'indice', 'nombre': 'Índice', 'archivo': 'fase2_index.html', 'emoji': '📋'},
            {'id': 'm01', 'nombre': 'Inspección', 'archivo': 'm01_inspeccion.html', 'emoji': '🔍'},
            {'id': 'm02', 'nombre': 'Calidad', 'archivo': 'm02_calidad.html', 'emoji': '✅'},
            {'id': 'm03', 'nombre': 'Nulos', 'archivo': 'm03_nulos.html', 'emoji': '❓'},
            {'id': 'm04', 'nombre': 'Univariante Num', 'archivo': 'm04_univariante_num.html', 'emoji': '📈'},
            {'id': 'm05', 'nombre': 'Univariante Cat', 'archivo': 'm05_univariante_cat.html', 'emoji': '📊'},
            {'id': 'm06', 'nombre': 'Evolución', 'archivo': 'm06_evolucion.html', 'emoji': '📈'},
            {'id': 'm07', 'nombre': 'Conclusiones', 'archivo': 'm07_conclusiones.html', 'emoji': '📝'},
        ]
    },
    'fase3': {
        'nombre': 'Features',
        'emoji': '🔧',
        'archivo': 'fase3_index.html',
        'carpeta': 'fase3',
        'estado': 'activo',
        'modulos': [
            {'id': 'indice', 'nombre': 'Índice', 'archivo': 'fase3_index.html', 'emoji': '📋'},
            {'id': 'm01', 'nombre': 'Validación', 'archivo': 'm01_validacion.html', 'emoji': '✅'},
            {'id': 'm02', 'nombre': 'Agregación', 'archivo': 'm02_agregacion.html', 'emoji': '🔗'},
            {'id': 'm03', 'nombre': 'Features', 'archivo': 'm03_features.html', 'emoji': '🧪'},
            {'id': 'm04', 'nombre': 'Encoding', 'archivo': 'm04_encoding.html', 'emoji': '🏷️'},
            {'id': 'm05', 'nombre': 'Target y Export', 'archivo': 'm05_target_export.html', 'emoji': '🎯'},
            {'id': 'm06', 'nombre': 'Alerta Temprana', 'archivo': 'm06_alerta_temprana.html', 'emoji': '⚠️'},
            {'id': 'm07', 'nombre': 'Validación', 'archivo': 'm07_validacion.html', 'emoji': '✅'},
            {'id': 'm08', 'nombre': 'Perfiles',     'archivo': 'm08_perfiles_riesgo.html', 'emoji': '👤'},
        ]
    },
    'fase4': {
        'nombre': 'EDA Final',
        'emoji': '🔬',
        'archivo': 'fase4_index.html',
        'carpeta': 'fase4',
        'estado': 'activo',
        'modulos': [
            {'id': 'indice', 'nombre': 'Índice', 'archivo': 'fase4_index.html', 'emoji': '📋'},
            {'id': 'm01', 'nombre': 'Inspección', 'archivo': 'm01_inspeccion.html', 'emoji': '🔍'},
            {'id': 'm02', 'nombre': 'Target', 'archivo': 'm02_target.html', 'emoji': '🎯'},
            {'id': 'm03', 'nombre': 'Distrib. Num', 'archivo': 'm03_distribuciones_num.html', 'emoji': '📊'},
            {'id': 'm04', 'nombre': 'Distrib. Cat', 'archivo': 'm04_distribuciones_cat.html', 'emoji': '📈'},
            {'id': 'm05', 'nombre': 'Bivariante', 'archivo': 'm05_bivariante.html', 'emoji': '🔗'},
            {'id': 'm06', 'nombre': 'Correlaciones', 'archivo': 'm06_correlaciones.html', 'emoji': '🔥'},
            {'id': 'm07', 'nombre': 'Selección', 'archivo': 'm07_seleccion_features.html', 'emoji': '🎯'},
            {'id': 'm08', 'nombre': 'Comparativa Grupos', 'archivo': 'm08_perfiles_riesgo.html', 'emoji': '📊'},
            {'id': 'm09', 'nombre': 'Conclusiones', 'archivo': 'm09_conclusiones_eda.html', 'emoji': '📝'},
        ]
    },
    'fase5': {
        'nombre': 'Modelado',
        'emoji': '🤖',
        'archivo': 'fase5_index.html',
        'carpeta': 'fase5',
        'estado': 'pendiente',
        'modulos': [
            {'id': 'indice', 'nombre': 'Índice', 'archivo': 'fase5_index.html', 'emoji': '📋'},
            {'id': 'm01', 'nombre': 'Baseline', 'archivo': 'm01_baseline.html', 'emoji': '📏'},
            {'id': 'm02', 'nombre': 'Logistic Regression', 'archivo': 'm02_logistic.html', 'emoji': '📈'},
            {'id': 'm03', 'nombre': 'Random Forest', 'archivo': 'm03_random_forest.html', 'emoji': '🌲'},
            {'id': 'm04', 'nombre': 'XGBoost', 'archivo': 'm04_xgboost.html', 'emoji': '🚀'},
            {'id': 'm05', 'nombre': 'Redes Neuronales', 'archivo': 'm05_neural.html', 'emoji': '🧠'},
            {'id': 'm06', 'nombre': 'Ensemble', 'archivo': 'm06_ensemble.html', 'emoji': '🎭'},
        ]
    },
    'fase6': {
        'nombre': 'Evaluación',
        'emoji': '📈',
        'archivo': 'fase6_index.html',
        'carpeta': 'fase6',
        'estado': 'pendiente',
        'modulos': [
            {'id': 'indice', 'nombre': 'Índice', 'archivo': 'fase6_index.html', 'emoji': '📋'},
            {'id': 'm01', 'nombre': 'Comparativa', 'archivo': 'm01_comparativa.html', 'emoji': '⚖️'},
            {'id': 'm02', 'nombre': 'Métricas', 'archivo': 'm02_metricas.html', 'emoji': '📊'},
            {'id': 'm03', 'nombre': 'SHAP', 'archivo': 'm03_shap.html', 'emoji': '🔍'},
            {'id': 'm04', 'nombre': 'Modelo Final', 'archivo': 'm04_modelo_final.html', 'emoji': '🏆'},
            {'id': 'm05', 'nombre': 'Conclusiones', 'archivo': 'm05_conclusiones.html', 'emoji': '📝'},
        ]
    },
    'fase7': {
        'nombre': 'Aplicación',
        'emoji': '🚀',
        'archivo': 'fase7_index.html',
        'carpeta': 'fase7',
        'estado': 'pendiente',
        'modulos': [
            {'id': 'indice', 'nombre': 'Índice', 'archivo': 'fase7_index.html', 'emoji': '📋'},
            {'id': 'm01', 'nombre': 'Dashboard Gestor', 'archivo': 'm01_dashboard_gestor.html', 'emoji': '📊'},
            {'id': 'm02', 'nombre': 'Vista Profesor', 'archivo': 'm02_vista_profesor.html', 'emoji': '👨‍🏫'},
            {'id': 'm03', 'nombre': 'Vista Alumno', 'archivo': 'm03_vista_alumno.html', 'emoji': '🎓'},
            {'id': 'm04', 'nombre': 'API', 'archivo': 'm04_api.html', 'emoji': '🔌'},
            {'id': 'm05', 'nombre': 'Documentación', 'archivo': 'm05_documentacion.html', 'emoji': '📖'},
        ]
    },
    # ------------------------------------------------------------------
    # ANEXO: Pre-Modelado AutoML (fuera del flujo principal)
    # ------------------------------------------------------------------
    'fase_automl': {
        'nombre': 'Pre-Modelado: AutoML',
        'emoji': '⚡',
        'archivo': 'fase_automl_index.html',
        'carpeta': 'fase_automl',
        'estado': 'activo',
        'modulos': [
            {'id': 'indice', 'nombre': 'Índice', 'archivo': 'fase_automl_index.html', 'emoji': '📋'},
            {'id': 'm01', 'nombre': 'Baselines', 'archivo': 'm01_baselines.html', 'emoji': '📊'},
            {'id': 'm02', 'nombre': 'LazyPredict', 'archivo': 'm02_lazypredict.html', 'emoji': '⚡'},
            {'id': 'm03', 'nombre': 'PyCaret', 'archivo': 'm03_pycaret.html', 'emoji': '🤖'},
            {'id': 'm04', 'nombre': 'H2O', 'archivo': 'm04_h2o.html', 'emoji': '💧'},
            {'id': 'm05', 'nombre': 'AutoGluon', 'archivo': 'm05_autogluon.html', 'emoji': '🚀'},
            {'id': 'm06', 'nombre': 'Comparativa', 'archivo': 'm06_comparativa.html', 'emoji': '🏆'},
        ]
    },
}


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def obtener_fase_siguiente(fase_id: str) -> Dict:
    """Devuelve info de la fase siguiente."""
    fases_lista = list(FASES.keys())
    try:
        idx = fases_lista.index(fase_id)
        if idx + 1 < len(fases_lista):
            siguiente_id = fases_lista[idx + 1]
            return {
                'id': siguiente_id,
                **FASES[siguiente_id]
            }
    except (ValueError, IndexError):
        pass
    return {}


# ============================================================================
# FUNCIONES PARA TÍTULO DINÁMICO
# ============================================================================

def extraer_titulo_de_fichero(nombre_fichero: str) -> str:
    """
    Extrae título legible del nombre del fichero.
    
    Ejemplos:
        'f1_m03_reportes_clean.ipynb' -> 'Reportes Clean'
        'f2_m01_inspeccion.ipynb' -> 'Inspección'
        'f3_m00_indice.ipynb' -> 'Índice'
        'fautoml_m00_indice.ipynb' -> 'Índice'
        'fautoml_m03_pycaret.ipynb' -> 'Pycaret'
    """
    nombre = nombre_fichero.replace('.ipynb', '').replace('.html', '')
    # Soportar fautoml_mXX_ además del patrón numérico fN_mXX_
    patron = r'^(?:f\d+|fautoml)_m\d+_'
    nombre = re.sub(patron, '', nombre)
    
    if nombre.lower() in ['indice', 'resumen', 'index']:
        return 'Índice'
    
    titulo = nombre.replace('_', ' ').title()
    return titulo


def extraer_fase_de_fichero(nombre_fichero: str) -> str:
    """
    Extrae el ID de fase del nombre del fichero.
    
    Ejemplos:
        'f1_m03_reportes_clean.ipynb' -> 'fase1'
        'f3_m01_validacion.ipynb' -> 'fase3'
        'fautoml_m00_indice.ipynb' -> 'fase_automl'
    """
    # Primero intentar patrón fautoml_
    if nombre_fichero.startswith('fautoml_'):
        return 'fase_automl'
    # Patrón numérico normal
    match = re.match(r'^f(\d+)_', nombre_fichero)
    if match:
        return f'fase{match.group(1)}'
    return 'fase1'


def extraer_modulo_de_fichero(nombre_fichero: str) -> str:
    """
    Extrae el ID de módulo del nombre del fichero.
    
    Ejemplos:
        'f1_m03_reportes_clean.ipynb' -> 'm03'
        'f3_m00_indice.ipynb' -> 'indice'
        'fautoml_m01_baselines.ipynb' -> 'm01'
        'fautoml_m00_indice.ipynb' -> 'indice'
    """
    # Soportar fautoml_mXX_
    match = re.match(r'^(?:f\d+|fautoml)_m(\d+)_', nombre_fichero)
    if match:
        num = match.group(1)
        if num == '00':
            return 'indice'
        return f'm{num}'
    return 'indice'


def obtener_subtitulo_fase(fase_id: str) -> str:
    """
    Devuelve el subtítulo de la fase.
    
    Ejemplo:
        'fase3' -> 'Fase 3: Features | TFM Abandono Universitario'
        'fase_automl' -> 'Pre-Modelado: AutoML | TFM Abandono Universitario'
    """
    if fase_id not in FASES:
        return 'TFM Abandono Universitario'
    
    fase = FASES[fase_id]
    nombre = fase['nombre']
    
    # Caso especial: fase_automl no tiene número
    if fase_id == 'fase_automl':
        return f'{nombre} | TFM Abandono Universitario'
    
    num = fase_id.replace('fase', '')
    return f'Fase {num}: {nombre} | TFM Abandono Universitario'


# ============================================================================
# FUNCIONES DE NAVEGACIÓN
# ============================================================================

def obtener_fases_para_nav(fase_activa: str = None) -> List[Dict]:
    """Devuelve lista de fases para la navegación principal."""
    nav = []
    for fase_id, info in FASES.items():
        nav.append({
            'id': fase_id,
            'nombre': info['nombre'],
            'emoji': info['emoji'],
            'archivo': info['archivo'],
            'carpeta': info.get('carpeta', ''),
            'activo': fase_id == fase_activa,
            'estado': info.get('estado', 'activo')
        })
    return nav


def obtener_modulos_para_nav(fase_id: str, modulo_activo: str = None) -> List[Dict]:
    """Devuelve lista de módulos para la navegación secundaria."""
    if fase_id not in FASES or 'modulos' not in FASES[fase_id]:
        return []
    
    nav = []
    for mod in FASES[fase_id]['modulos']:
        nav.append({
            'id': mod['id'],
            'nombre': mod['nombre'],
            'emoji': mod.get('emoji', '📄'),
            'archivo': mod['archivo'],
            'activo': mod['id'] == modulo_activo
        })
    return nav


def obtener_info_fase(fase_id: str) -> Dict:
    """Devuelve info completa de una fase."""
    return FASES.get(fase_id, {})


def obtener_info_modulo(fase_id: str, modulo_id: str) -> Dict:
    """Devuelve info de un módulo específico."""
    fase = FASES.get(fase_id, {})
    modulos = fase.get('modulos', [])
    for mod in modulos:
        if mod['id'] == modulo_id:
            return mod
    return {}


# ============================================================================
# GENERACIÓN DE HTML DE NAVEGACIÓN
# ============================================================================

def generar_html_nav_fases(fase_activa: str = None, ruta_base: str = '..') -> str:
    """Genera HTML de navegación de fases."""
    fases = obtener_fases_para_nav(fase_activa)
    
    html = '<nav class="nav-fases">\n'
    for fase in fases:
        clase = 'active' if fase['activo'] else ''
        if fase['estado'] == 'pendiente':
            clase += ' disabled'
        
        if fase['id'] == 'inicio':
            href = f'{ruta_base}/index.html'
        else:
            href = f'{ruta_base}/{fase["carpeta"]}/{fase["archivo"]}'
        
        html += f'  <a href="{href}" class="{clase}">{fase["emoji"]} {fase["nombre"]}</a>\n'
    
    html += '</nav>'
    return html


def generar_html_nav_modulos(fase_id: str, modulo_activo: str = None) -> str:
    """Genera HTML de navegación de módulos de una fase."""
    modulos = obtener_modulos_para_nav(fase_id, modulo_activo)
    
    if not modulos:
        return ''
    
    html = '<nav class="nav-modulos">\n'
    for mod in modulos:
        clase = 'active' if mod['activo'] else ''
        html += f'  <a href="{mod["archivo"]}" class="{clase}">{mod["emoji"]} {mod["nombre"]}</a>\n'
    
    html += '</nav>'
    return html


def generar_html_navegacion_completa(fase_activa: str, modulo_activo: str = None, ruta_base: str = '..') -> tuple:
    """
    Genera HTML completo de navegación (fases + módulos).
    
    Returns:
        tuple: (nav_fases_html, nav_modulos_html)
    """
    nav_fases = generar_html_nav_fases(fase_activa, ruta_base)
    nav_modulos = generar_html_nav_modulos(fase_activa, modulo_activo)
    
    return nav_fases, nav_modulos
