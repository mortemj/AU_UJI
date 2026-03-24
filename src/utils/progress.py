# ============================================================================
# PROGRESS.PY - Barra de progreso y spinners para notebooks
# ============================================================================
# Ubicación: src/utils/progress.py
#
# Funciones disponibles:
#   progreso()          → barra tqdm para bucles for
#   progreso_manual()   → barra tqdm para actualizar paso a paso
#   spinner()           → context manager con widget HTML animado
#   con_loading()       → decorador que envuelve funciones con spinner
#   progreso_fase()     → tabla de progreso de fase completa (para orquestador)
#
# Uso básico:
#   from src.utils.progress import progreso, spinner, con_loading
# ============================================================================

import time
import threading
import functools


# ============================================================================
# FUNCIONES ORIGINALES (sin cambios)
# ============================================================================

def progreso(iterable, desc="Procesando", total=None):
    """
    Barra de progreso con color para Jupyter Notebooks.

    Detecta automáticamente el entorno (Jupyter/terminal) y usa
    la mejor visualización disponible.

    Parameters
    ----------
    iterable : iterable
        Elementos a iterar
    desc : str
        Descripción que aparece antes de la barra
    total : int, optional
        Total de elementos (se calcula automáticamente si no se indica)

    Returns
    -------
    tqdm
        Iterador con barra de progreso

    Examples
    --------
    >>> from src.utils.progress import progreso
    >>> for tabla in progreso(tablas, "📖 Leyendo Excels"):
    ...     procesar(tabla)
    """
    try:
        from tqdm.notebook import tqdm
    except ImportError:
        try:
            from tqdm import tqdm
            return tqdm(iterable, desc=desc, total=total, colour='#3182ce')
        except ImportError:
            print("⚠️ tqdm no instalado. Ejecuta: pip install tqdm")
            return iterable

    return tqdm(iterable, desc=desc, total=total)


def progreso_manual(total, desc="Procesando"):
    """
    Barra de progreso manual para actualizar paso a paso.

    Útil cuando no se puede usar un bucle for simple.

    Parameters
    ----------
    total : int
        Número total de pasos
    desc : str
        Descripción de la tarea

    Returns
    -------
    tqdm
        Objeto tqdm para actualizar manualmente con .update(1)

    Examples
    --------
    >>> pbar = progreso_manual(10, "📊 Generando reportes")
    >>> for i in range(10):
    ...     hacer_algo()
    ...     pbar.update(1)
    >>> pbar.close()
    """
    try:
        from tqdm.notebook import tqdm
    except ImportError:
        try:
            from tqdm import tqdm
            return tqdm(total=total, desc=desc, colour='#3182ce')
        except ImportError:
            print("⚠️ tqdm no instalado")
            return None

    return tqdm(total=total, desc=desc)


# ============================================================================
# NUEVAS FUNCIONES — v2 (2026-03-11)
# ============================================================================

class spinner:
    """
    Context manager que muestra un spinner HTML animado mientras corre un
    proceso largo. No requiere bucle — envuelve cualquier bloque de código.

    Usa ipywidgets si está disponible; si no, imprime mensajes de texto.

    Parameters
    ----------
    mensaje : str
        Texto que aparece junto al spinner (ej. "⏳ Calculando SHAP global...")
    mensaje_ok : str, optional
        Texto que aparece al terminar (por defecto "✅ {mensaje} completado")

    Examples
    --------
    >>> from src.utils.progress import spinner
    >>> with spinner("⏳ Calculando SHAP global"):
    ...     shap_values = explainer(X_test_sample)
    """

    # CSS del spinner — se inyecta una sola vez por sesión
    _css_inyectado = False

    _CSS = """
    <style>
    .tfm-spinner-container {
        display: flex; align-items: center; gap: 12px;
        padding: 10px 16px; margin: 8px 0;
        background: #ebf8ff; border-left: 4px solid #3182ce;
        border-radius: 6px; font-family: monospace; font-size: 14px;
    }
    .tfm-spinner-container.ok {
        background: #f0fff4; border-left-color: #38a169;
    }
    .tfm-spinner-container.err {
        background: #fff5f5; border-left-color: #e53e3e;
    }
    @keyframes tfm-spin {
        0%   { transform: rotate(0deg);   }
        100% { transform: rotate(360deg); }
    }
    .tfm-spinner-ring {
        width: 20px; height: 20px; border-radius: 50%;
        border: 3px solid #bee3f8;
        border-top-color: #3182ce;
        animation: tfm-spin 0.8s linear infinite;
        flex-shrink: 0;
    }
    </style>
    """

    def __init__(self, mensaje: str, mensaje_ok: str = None):
        self.mensaje    = mensaje
        self.mensaje_ok = mensaje_ok or f"✅ {mensaje.lstrip('⏳ ').strip()} completado"
        self._widget    = None
        self._t_inicio  = None

    def __enter__(self):
        self._t_inicio = time.time()
        try:
            import ipywidgets as widgets
            from IPython.display import display

            if not spinner._css_inyectado:
                from IPython.display import HTML
                display(HTML(spinner._CSS))
                spinner._css_inyectado = True

            html_ini = f"""
            <div class="tfm-spinner-container">
                <div class="tfm-spinner-ring"></div>
                <span>{self.mensaje}</span>
            </div>"""
            self._widget = widgets.HTML(value=html_ini)
            display(self._widget)
        except ImportError:
            print(f"{self.mensaje}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self._t_inicio
        elapsed_str = f"{elapsed:.1f}s" if elapsed < 60 else f"{elapsed/60:.1f}min"

        if exc_type is None:
            msg = self.mensaje_ok
            clase = "ok"
        else:
            msg = f"❌ Error en: {self.mensaje.lstrip('⏳ ').strip()}"
            clase = "err"

        if self._widget is not None:
            self._widget.value = f"""
            <div class="tfm-spinner-container {clase}">
                <span>{msg} &nbsp;<small style="color:#718096">({elapsed_str})</small></span>
            </div>"""
        else:
            print(f"{msg} ({elapsed_str})")

        return False  # no suprimir excepciones


def con_loading(mensaje: str, mensaje_ok: str = None):
    """
    Decorador que envuelve una función con spinner automático.

    Muestra el spinner al llamar a la función y lo actualiza al terminar
    (con tiempo transcurrido). Si la función lanza excepción, muestra error.

    Parameters
    ----------
    mensaje : str
        Texto que aparece mientras corre la función
    mensaje_ok : str, optional
        Texto al terminar (por defecto "✅ {mensaje} completado")

    Examples
    --------
    >>> from src.utils.progress import con_loading
    >>>
    >>> @con_loading("⏳ Calculando SHAP global")
    ... def calcular_shap(explainer, X):
    ...     return explainer(X)
    >>>
    >>> shap_values = calcular_shap(explainer, X_test_sample)
    """
    def decorador(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with spinner(mensaje, mensaje_ok):
                return func(*args, **kwargs)
        return wrapper
    return decorador


def progreso_fase(modulos: list, titulo: str = "Progreso de fase"):
    """
    Muestra una tabla de progreso de fase completa con estados por módulo.

    Devuelve un objeto que permite actualizar el estado de cada módulo
    en tiempo real. Pensado para uso en f6_m00_ejecucion.ipynb.

    Parameters
    ----------
    modulos : list of dict
        Lista de módulos con claves: 'id', 'nombre', 'emoji'
        Ej: [{'id': 'm01a', 'nombre': 'SHAP Global', 'emoji': '🌍'}, ...]
    titulo : str
        Título de la tabla

    Returns
    -------
    ProgresoFase
        Objeto con métodos:
            .iniciar(id)    → marca módulo como en ejecución ⏳
            .ok(id)         → marca módulo como completado ✅ + tiempo
            .error(id)      → marca módulo como fallido ❌
            .mostrar()      → renderiza la tabla (llamar al inicio)

    Examples
    --------
    >>> from src.utils.progress import progreso_fase
    >>> modulos = [
    ...     {'id': 'm01a', 'nombre': 'SHAP Global',  'emoji': '🌍'},
    ...     {'id': 'm01b', 'nombre': 'SHAP Local',   'emoji': '🔬'},
    ... ]
    >>> pf = progreso_fase(modulos, "Fase 6 — Ejecución")
    >>> pf.mostrar()
    >>> pf.iniciar('m01a')
    >>> ejecutar_modulo_m01a()
    >>> pf.ok('m01a')
    """
    return _ProgresoFase(modulos, titulo)


class _ProgresoFase:
    """Implementación interna de progreso_fase."""

    _ESTADO_ICONO = {
        'pendiente':  '⬜',
        'corriendo':  '⏳',
        'ok':         '✅',
        'error':      '❌',
    }

    def __init__(self, modulos: list, titulo: str):
        self.titulo   = titulo
        self.modulos  = [dict(m) for m in modulos]   # copia defensiva
        self._estados = {m['id']: 'pendiente' for m in self.modulos}
        self._tiempos = {}
        self._t_ini   = {}
        self._widget  = None

        for m in self.modulos:
            m.setdefault('emoji', '📄')

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def mostrar(self):
        """Renderiza la tabla inicial en el notebook."""
        try:
            import ipywidgets as widgets
            from IPython.display import display
            self._widget = widgets.HTML(value=self._render_html())
            display(self._widget)
        except ImportError:
            print(self._render_texto())

    def iniciar(self, modulo_id: str):
        """Marca el módulo como en ejecución."""
        self._estados[modulo_id] = 'corriendo'
        self._t_ini[modulo_id]   = time.time()
        self._actualizar()

    def ok(self, modulo_id: str):
        """Marca el módulo como completado con éxito."""
        self._estados[modulo_id] = 'ok'
        if modulo_id in self._t_ini:
            self._tiempos[modulo_id] = time.time() - self._t_ini[modulo_id]
        self._actualizar()

    def error(self, modulo_id: str):
        """Marca el módulo como fallido."""
        self._estados[modulo_id] = 'error'
        self._actualizar()

    # ------------------------------------------------------------------
    # Renderizado
    # ------------------------------------------------------------------

    def _tiempo_str(self, modulo_id: str) -> str:
        t = self._tiempos.get(modulo_id)
        if t is None:
            return ''
        return f"{t:.0f}s" if t < 60 else f"{t/60:.1f}min"

    def _render_html(self) -> str:
        n_ok    = sum(1 for v in self._estados.values() if v == 'ok')
        n_total = len(self.modulos)
        pct     = int(n_ok / n_total * 100) if n_total else 0

        filas = ""
        for m in self.modulos:
            mid    = m['id']
            estado = self._estados[mid]
            icono  = self._ESTADO_ICONO[estado]
            tiempo = self._tiempo_str(mid)
            color_fila = {
                'pendiente': '',
                'corriendo': 'background:#ebf8ff',
                'ok':        'background:#f0fff4',
                'error':     'background:#fff5f5',
            }[estado]
            filas += f"""
            <tr style="{color_fila}">
                <td style="padding:6px 12px">{icono}</td>
                <td style="padding:6px 12px">{m['emoji']} {m['nombre']}</td>
                <td style="padding:6px 12px; color:#718096; font-size:12px">{mid}</td>
                <td style="padding:6px 12px; color:#718096; font-size:12px">{tiempo}</td>
            </tr>"""

        barra_color = '#38a169' if pct == 100 else '#3182ce'
        return f"""
        <div style="font-family:monospace; margin:12px 0">
            <div style="font-size:15px; font-weight:bold; margin-bottom:8px">
                🗂️ {self.titulo}
            </div>
            <table style="border-collapse:collapse; width:100%; font-size:13px">
                <thead>
                    <tr style="background:#edf2f7">
                        <th style="padding:6px 12px; text-align:left">Estado</th>
                        <th style="padding:6px 12px; text-align:left">Módulo</th>
                        <th style="padding:6px 12px; text-align:left">ID</th>
                        <th style="padding:6px 12px; text-align:left">Tiempo</th>
                    </tr>
                </thead>
                <tbody>{filas}</tbody>
            </table>
            <div style="margin-top:10px">
                <div style="font-size:12px; color:#4a5568; margin-bottom:4px">
                    Progreso: {n_ok}/{n_total} módulos ({pct}%)
                </div>
                <div style="background:#e2e8f0; border-radius:4px; height:8px; width:100%">
                    <div style="background:{barra_color}; border-radius:4px;
                                height:8px; width:{pct}%;
                                transition: width 0.4s ease">
                    </div>
                </div>
            </div>
        </div>"""

    def _render_texto(self) -> str:
        """Fallback sin ipywidgets."""
        lineas = [f"\n{'='*50}", f"  {self.titulo}", f"{'='*50}"]
        for m in self.modulos:
            mid    = m['id']
            estado = self._estados[mid]
            icono  = self._ESTADO_ICONO[estado]
            tiempo = self._tiempo_str(mid)
            t_str  = f" ({tiempo})" if tiempo else ""
            lineas.append(f"  {icono}  {m['emoji']} {m['nombre']}{t_str}")
        n_ok = sum(1 for v in self._estados.values() if v == 'ok')
        lineas.append(f"{'='*50}")
        lineas.append(f"  {n_ok}/{len(self.modulos)} completados")
        return "\n".join(lineas)

    def _actualizar(self):
        """Actualiza el widget si existe, si no re-imprime."""
        if self._widget is not None:
            self._widget.value = self._render_html()
        else:
            print(self._render_texto())
