# ============================================================================
# ORQUESTADOR.PY - Orquestador de fases para notebooks de ejecución
# ============================================================================
# Ubicación: src/utils/orquestador.py
#
# Funciones disponibles:
#   orquestador_fase()   → crea y devuelve un ProgresoFase listo para usar
#
# Uso en f6_m00_ejecucion.ipynb:
#   from src.utils.orquestador import orquestador_fase
#
#   orch = orquestador_fase("Fase 6 — Evaluación")
#   orch.mostrar()
#
#   orch.iniciar('m01a')
#   ejecutar_notebook('f6_m01a_shap_global.ipynb')
#   orch.ok('m01a')
#
# Internamente delega en progress.progreso_fase para todo el rendering.
# Este módulo existe para poder importar el orquestador con un nombre
# semántico distinto de la barra de progreso de bucles.
# ============================================================================

from src.utils.progress import progreso_fase, _ProgresoFase
import subprocess
import sys
import time
from pathlib import Path


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def orquestador_fase(titulo: str, modulos: list = None) -> _ProgresoFase:
    """
    Crea un orquestador de progreso de fase.

    Devuelve un objeto _ProgresoFase con métodos:
        .mostrar()       → renderiza la tabla de progreso en el notebook
        .iniciar(id)     → marca módulo como ⏳ en ejecución
        .ok(id)          → marca módulo como ✅ completado + tiempo
        .error(id)       → marca módulo como ❌ fallido

    Parameters
    ----------
    titulo : str
        Título que aparece en la cabecera de la tabla.
        Ej: "Fase 6 — Evaluación"
    modulos : list of dict, optional
        Lista de módulos con claves: 'id', 'nombre', 'emoji'
        Si se omite, se usa la lista estándar de Fase 6.

    Returns
    -------
    _ProgresoFase
        Objeto de progreso listo para usar.

    Examples
    --------
    >>> from src.utils.orquestador import orquestador_fase
    >>> orch = orquestador_fase("Fase 6 — Evaluación")
    >>> orch.mostrar()
    >>> orch.iniciar('m01a')
    >>> # ... ejecutar notebook ...
    >>> orch.ok('m01a')
    """
    if modulos is None:
        modulos = MODULOS_FASE6

    return progreso_fase(modulos, titulo)


# ============================================================================
# LISTA ESTÁNDAR DE MÓDULOS FASE 6
# ============================================================================

MODULOS_FASE6 = [
    # --- Gestión ---
    {'id': 'm00_prep',  'nombre': 'Preparación',            'emoji': '⚙️'},
    {'id': 'm00_ejec',  'nombre': 'Ejecución',              'emoji': '▶️'},
    # --- SHAP ---
    {'id': 'm01a',      'nombre': 'SHAP Global',            'emoji': '🌍'},
    {'id': 'm01b',      'nombre': 'SHAP Local',             'emoji': '🔬'},
    {'id': 'm01c',      'nombre': 'SHAP Cohortes',          'emoji': '👥'},
    {'id': 'm01d',      'nombre': 'Shapash',                'emoji': '📊'},
    # --- Interpretabilidad Alternativa ---
    {'id': 'm02a',      'nombre': 'LIME',                   'emoji': '🍋'},
    {'id': 'm02b',      'nombre': 'DiCE',                   'emoji': '🎲'},
    # --- Fairness y Errores ---
    {'id': 'm03a',      'nombre': 'Fairness',               'emoji': '⚖️'},
    {'id': 'm03b',      'nombre': 'Errores FP/FN',          'emoji': '❌'},
    # --- Robustez y Calibración ---
    {'id': 'm04a',      'nombre': 'Stress Testing',         'emoji': '💪'},
    {'id': 'm04b',      'nombre': 'Calibración',            'emoji': '🎯'},
    {'id': 'm04c',      'nombre': 'Sostenibilidad',         'emoji': '🌱'},
    # --- Informe Final ---
    {'id': 'm05',       'nombre': 'Informe Final',          'emoji': '🏆'},
]


# ============================================================================
# EJECUCIÓN DE NOTEBOOKS (helper opcional)
# ============================================================================

def ejecutar_notebook(ruta_notebook: str | Path, timeout: int = 3600) -> bool:
    """
    Ejecuta un notebook con nbconvert y devuelve True si tuvo éxito.

    Pensado para uso desde f6_m00_ejecucion.ipynb junto al orquestador:
        orch.iniciar('m01a')
        ok = ejecutar_notebook('f6_m01a_shap_global.ipynb')
        orch.ok('m01a') if ok else orch.error('m01a')

    Parameters
    ----------
    ruta_notebook : str | Path
        Ruta al fichero .ipynb (absoluta o relativa al CWD).
    timeout : int
        Tiempo máximo en segundos por celda (por defecto 3600 = 1h).

    Returns
    -------
    bool
        True si nbconvert terminó sin errores, False en caso contrario.
    """
    ruta = Path(ruta_notebook)
    cmd = [
        sys.executable, '-m', 'nbconvert',
        '--to', 'notebook',
        '--execute',
        f'--ExecutePreprocessor.timeout={timeout}',
        '--inplace',
        str(ruta)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error ejecutando {ruta.name}: {e}")
        return False
