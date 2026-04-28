# =============================================================================
# utils/ui_helpers.py
# Helpers de UI compartidos entre páginas de la app
#
# ¿QUÉ HACE ESTE FICHERO?
#   Centraliza componentes visuales reutilizables (tarjetas KPI, formato de
#   nombres de titulación, etc.) para evitar que distintas páginas dupliquen
#   el mismo código y para que cualquier cambio se propague automáticamente.
#
# CONTENIDO:
#   - _pie_pagina(): pie de página con autoría
#   - _leer_metricas_modelo(): lee metricas_modelo.json (cacheado)
#   - _hex_a_rgba(hex, alpha): convierte hex a rgba con transparencia
#   - _guardia_df_vacio(df, titulo): aviso visual si df está vacío
#   - _clasificar_riesgo(prob): nivel + color + emoji + mensaje
#   - _nombre_titulacion_corto(nombre): acorta nombre quitando "Grado en"
#   - _tarjeta_kpi(...): tarjeta KPI compacta con barra lateral 4px +
#                       sparkline opcional (Chat p00, 28/04/2026)
#
# QUIÉN LA USA:
#   - p00_inicio.py
#   - p01_institucional.py
#   - p02_titulacion.py
#   - utils/pronostico_shared.py (p03, p04)
#
# REFACTOR (Chat p03, 27/04/2026): centralización de funciones que vivían
# duplicadas en distintos ficheros con implementaciones inconsistentes.
# =============================================================================

import _path_setup  # noqa: F401

from config_app import COLORES, COLORES_RIESGO, UMBRALES, APP_CONFIG


# =============================================================================
# HELPER — Pie de página (autoría)
# =============================================================================
# REFACTOR p03 (Chat p03, 27/04/2026): antes había 4 versiones:
#   - p00_inicio.py: función _pie_pagina()
#   - p02_titulacion.py: función _pie_pagina() (idéntica a p00)
#   - p01_institucional.py: bloque inline (no función)
#   - pronostico_shared.py: función _pie_pagina() (añadida ayer)
# Ahora todas usan ESTA, importada desde ui_helpers.


def _pie_pagina():
    """Información de autoría y créditos (usar al final de cada página)."""
    import streamlit as st  # import local: ui_helpers no debería forzar st global
    st.markdown("<br>", unsafe_allow_html=True)  # espaciado
    st.markdown(f"""
    <div style="
        text-align: center;
        font-size: 0.78rem;
        color: {COLORES["texto_suave"]};
        padding: 1rem;
        border-top: 1px solid {COLORES["borde"]};
    ">
        {APP_CONFIG['autora']} &nbsp;·&nbsp;
        {APP_CONFIG['tipo_trabajo']} &nbsp;·&nbsp;
        {APP_CONFIG['universidad_master']} + {APP_CONFIG['universidad_datos']} &nbsp;·&nbsp; {APP_CONFIG['año']}<br>
        <span style="font-size: 0.72rem;">
            {APP_CONFIG['email_master']} &nbsp;·&nbsp; {APP_CONFIG['email_datos']}
        </span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# HELPER — Leer métricas del modelo desde metricas_modelo.json
# =============================================================================
# REFACTOR p03 (Chat p03, 27/04/2026): antes había 2 copias idénticas en
# p01 y p02. Centralizada aquí. Cacheada con st.cache_data para no releer
# el fichero en cada rerun. NUNCA lanza excepción: devuelve {} si falla.


def _leer_metricas_modelo() -> dict:
    """
    Lee el fichero metricas_modelo.json generado por la Fase 6 (evaluación).

    Returns
    -------
    dict
        Diccionario con las métricas del modelo. Claves típicas:
          - 'tasa_abandono' (float entre 0 y 1)
          - 'f1', 'auc', 'accuracy', 'precision', 'recall' (floats)
          - 'fecha_entrenamiento' (str, opcional)
          - 'modelo' (str, opcional — p.ej. "Stacking__balanced")
          - 'n_alumnos_unicos', 'n_registros', 'n_test' (int, opcional)
        Si el fichero no existe o falla la lectura, devuelve {} (dict vacío).

    Notas
    -----
    - NUNCA lanza excepción: si algo falla, devuelve {} silenciosamente.
      Esto permite que las funciones que lo usen traten la ausencia de
      datos con su propia lógica (ej. mostrar "N/D" o usar fallback).
    """
    try:
        import json as _json
        from config_app import RUTAS as _RUTAS
        ruta_m = _RUTAS.get("metricas_modelo")
        if ruta_m and ruta_m.exists():
            with open(ruta_m, encoding="utf-8") as _f:
                return _json.load(_f)
    except Exception:
        pass
    return {}


# =============================================================================
# HELPER — Convertir color hex a rgba con transparencia
# =============================================================================
# Centralizado en ui_helpers (Chat p00, 28/04/2026) para que cualquier
# página pueda derivar fondos suaves a partir de COLORES sin hardcodear.
# Antes se repetía en pronostico_shared (línea 1665) y p00 usaba hex de
# 8 caracteres tipo "{COLORES['primario']}15" que no son universales.


def _hex_a_rgba(hex_color: str, alpha: float) -> str:
    """
    Convierte un color hex (#rrggbb) a rgba(r,g,b,alpha).

    Útil para crear fondos suaves derivados de la paleta COLORES sin
    hardcodear hex auxiliares en config_app. Si el color de base cambia,
    el fondo derivado se actualiza automáticamente.

    Parameters
    ----------
    hex_color : str
        Color hex de 7 caracteres (#rrggbb). Si ya viene en rgba u otro
        formato, se devuelve tal cual.
    alpha : float
        Opacidad entre 0 y 1.

    Returns
    -------
    str
        rgba(r,g,b,alpha) si el hex es válido. El input original si no.

    Examples
    --------
    >>> _hex_a_rgba(COLORES['exito'], 0.10)
    'rgba(16,185,129,0.1)'
    >>> _hex_a_rgba(COLORES['primario'], 0.08)
    'rgba(30,77,140,0.08)'
    """
    if isinstance(hex_color, str) and hex_color.startswith("#") and len(hex_color) == 7:
        try:
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            return f"rgba({r},{g},{b},{alpha})"
        except ValueError:
            pass
    return hex_color


# =============================================================================
# HELPER — Guardia para DataFrame vacío
# =============================================================================
# REFACTOR p03 (Chat p03, 27/04/2026): antes había 2 copias idénticas en
# p01 y p02. Centralizada aquí. Devuelve True si el DataFrame está vacío
# (para que el bloque llamante haga return) y muestra un aviso visual
# coherente con el estilo de la app.


def _guardia_df_vacio(df, titulo_bloque: str) -> bool:
    """
    Comprueba si un DataFrame está vacío y, si lo está, renderiza un aviso
    visual coherente con el estilo de la app.

    Devuelve:
      - True  → df está vacío, el bloque que lo llame debe hacer `return`
      - False → df tiene datos, el bloque puede continuar normalmente

    Parámetros:
      df            → DataFrame a comprobar (típicamente df_filtrado)
      titulo_bloque → Nombre del bloque para el aviso (ej: "Evolución temporal")
    """
    import streamlit as st  # import local: ui_helpers no fuerza streamlit global
    if df is None or len(df) == 0:
        st.markdown(f"""
        <div style="background:{COLORES['fondo']};
            border:1px dashed {COLORES['texto_muy_suave']};
            border-radius:8px;
            padding:1.5rem 1rem;
            text-align:center;
            color:{COLORES['texto_suave']};
            font-size:0.85rem;
            margin:0.5rem 0;">
            <div style="font-weight:600; margin-bottom:0.3rem;
                color:{COLORES['texto']};">
                {titulo_bloque}
            </div>
            📭 No hay datos para mostrar con los filtros actuales.
            <br>
            <span style="font-size:0.78rem;">
                Prueba a relajar los filtros para ver este bloque.
            </span>
        </div>
        """, unsafe_allow_html=True)
        return True
    return False


# =============================================================================
# HELPER — Clasificar nivel de riesgo
# =============================================================================
# REFACTOR p03 (Chat p03, 27/04/2026): antes había 4 implementaciones:
#   - _clasificar_riesgo en pronostico_shared (la base, completa)
#   - _color_tasa en p01 (devolvía solo color+emoji)
#   - lógica inline en p01 L408 (if/elif/else)
#   - lógica inline en p02
# Ahora todas usan ESTA. Si _color_tasa solo necesita color+emoji, hace:
#   _, color, emoji, _ = _clasificar_riesgo(prob)


def _clasificar_riesgo(prob: float) -> tuple[str, str, str, str]:
    """
    Clasifica una probabilidad de abandono en nivel + color + emoji + mensaje.

    Usa UMBRALES y COLORES_RIESGO de config_app para coherencia global con
    los gauges, las tablas y los velocímetros del modelo.

    Parameters
    ----------
    prob : float
        Probabilidad en escala 0-1 (no porcentaje).

    Returns
    -------
    tuple[str, str, str, str]
        (nivel, color_hex, emoji, mensaje_descriptivo).
        nivel ∈ {'Bajo', 'Medio', 'Alto'}.

    Examples
    --------
    >>> nivel, color, emoji, msg = _clasificar_riesgo(0.85)
    >>> nivel
    'Alto'
    """
    if prob < UMBRALES['riesgo_bajo']:
        return (
            'Bajo', COLORES_RIESGO['bajo'], '✅',
            'Tu perfil muestra factores protectores importantes. '
            'El riesgo de abandono es reducido según el modelo.',
        )
    elif prob < UMBRALES['riesgo_medio']:
        return (
            'Medio', COLORES_RIESGO['medio'], '⚠️',
            'Hay algunos factores de riesgo en tu perfil. '
            'Con el apoyo adecuado y buena planificación, es muy manejable.',
        )
    else:
        return (
            'Alto', COLORES_RIESGO['alto'], '🔴',
            'Tu perfil presenta varios factores de riesgo. '
            'Te recomendamos consultar con la Unidad de Soporte Educativo '
            '(USE) de la UJI.',
        )


# =============================================================================
# HELPER — Acortar nombre de titulación
# =============================================================================
# REGLA: "acortar" = quitar el prefijo redundante "Grado en" porque todas
# son grados. NO es resumir ni inventar abreviaturas.
#
# Caso especial Doble Grado: el prefijo "Doble Grado en" se sustituye por
# "Doble: " para mantener la pista de que es un doble grado sin la
# redundancia. Sin esto, "Doble Grado en ADE y Derecho" se confunde con
# "Grado en ADE" si solo se muestra el resto.
#
# REFACTOR p03 (Chat p03, 27/04/2026): antes había 4 implementaciones:
#   - _nombre_titulacion_corto en p02 (la más completa, base de esta)
#   - _nombre_corto_tit en pronostico_shared (no manejaba Doble Grado)
#   - _partir_label en p01 (solo quitaba "Grado en", sin Doble)
#   - regex inline en p01 L2122
# Ahora todas usan ESTA, que es la versión completa.
_PREFIJOS_TITULACION = (
    "Grado en Ingeniería en ",
    "Grado en Ingeniería ",
    "Grado en Maestro en ",
    "Grado en Maestro o Maestra en ",
    "Grado en ",
    "Grado de ",
    "Grado ",
)


def _nombre_titulacion_corto(nombre: str, partir_lineas: bool = False,
                              max_chars: int = 22) -> str:
    """
    Acorta el nombre de una titulación quitando redundancia.

    Caso especial Doble Grado: el prefijo "Doble Grado en" se sustituye por
    "Doble: " para que el lector siga sabiendo que es un doble grado pero
    sin redundancia.

    Parameters
    ----------
    nombre : str
        Nombre completo de la titulación (ej: "Grado en Medicina").
    partir_lineas : bool
        Si True, parte el resultado en 2 líneas con <br> usando el espacio
        más cercano al centro. Default False.
    max_chars : int
        Si partir_lineas=True, solo parte si supera este umbral. Default 22.

    Returns
    -------
    str
        Nombre acortado (y opcionalmente partido en 2 líneas).

    Examples
    --------
    >>> _nombre_titulacion_corto("Grado en Medicina")
    'Medicina'
    >>> _nombre_titulacion_corto("Grado en Ingeniería en Tecnologías Industriales")
    'Tecnologías Industriales'
    >>> _nombre_titulacion_corto("Doble Grado en Administración y Dirección de Empresas y Derecho")
    'Doble: Administración y Dirección de Empresas y Derecho'
    """
    if not nombre:
        return ""

    # Caso especial: Doble Grado — sustituir "Doble Grado en" por "Doble: "
    nombre_limpio = nombre.strip().rstrip(",").strip()
    if nombre_limpio.startswith("Doble Grado en "):
        nombre = "Doble: " + nombre_limpio[len("Doble Grado en "):]
    elif nombre_limpio.startswith("Doble Grado de "):
        nombre = "Doble: " + nombre_limpio[len("Doble Grado de "):]
    else:
        # Caso general: quitar prefijo (el primero que coincida gana)
        for prefijo in _PREFIJOS_TITULACION:
            if nombre.startswith(prefijo):
                nombre = nombre[len(prefijo):]
                break

    # Paso 2: opcional, partir en 2 líneas
    if not partir_lineas or len(nombre) <= max_chars:
        return nombre

    mid = len(nombre) // 2
    izq = nombre.rfind(" ", 0, mid)
    der = nombre.find(" ", mid)
    if izq == -1 and der == -1:
        return nombre
    if izq == -1:
        corte = der
    elif der == -1:
        corte = izq
    else:
        corte = izq if (mid - izq) <= (der - mid) else der
    return nombre[:corte] + "<br>" + nombre[corte+1:]


# =============================================================================
# HELPER — Tarjeta KPI compacta (border-left 4px)
# =============================================================================
# Réplica del patrón de _tarjeta_kpi_html de p01, en versión compacta sin
# sparkline. Para vista operativa (p02 modo detalle/comparativa, p03 KPIs).
#
# REGLAS DE COLORES SEMÁNTICOS (alineadas con p01):
#   👥 alumnos          → COLORES["primario"]    azul
#   📉 abandono real    → COLORES["abandono"]    rojo
#   🔮 predicción       → COLORES["primario"]    azul
#   🚨 riesgo alto      → COLORES["advertencia"] ámbar  (NO rojo)
#   🎯 calidad modelo   → COLORES["exito"]       verde


def _tarjeta_kpi(icono: str, etiqueta: str, valor: str,
                 delta: str = "", delta_color: str = "",
                 tooltip: str = "", color_barra: str = None,
                 sparkline: tuple = None,
                 sparkline_labels: tuple = None) -> str:
    """
    Genera el HTML de una tarjeta KPI compacta con barra lateral de color.

    Función UNIFICADA (Chat p00, 28/04/2026): única card de KPI usada en
    p00, p02, p03, p04, p05. Sustituye a _kpi_card local de p00 y al
    _tarjeta_kpi_html de p01 (excepto cuando p01 necesita evolución
    temporal — para eso se usa _tarjeta_kpi_evolucion).

    Argumentos:
        icono: emoji del KPI (👥, 📉, 🔮, 🚨, 🎯…)
        etiqueta: nombre del indicador (en MAYÚSCULAS, va arriba)
        valor: valor principal del KPI (cifra grande)
        delta: texto secundario en cajita (vs cohorte anterior, vs media…).
               Si delta_color es 'red'/'green', se renderiza en cajita
               coloreada (estilo unificado p00). Si '' o 'gray', va en
               texto suelto sin cajita.
        delta_color: '' (gris), 'red' (peor), 'green' (mejor), 'gray' (info)
        tooltip: texto de tooltip nativo HTML
        color_barra: hex o COLORES[clave]. Si None, usa COLORES["primario"].
        sparkline: tupla (val_base, val_modelo) en escala 0-1 para mostrar
                   mini-barra comparativa. Si None, no se muestra.
                   Ejemplo: (0.927, 0.954) para AUC AutoML→Stacking.
        sparkline_labels: tupla (label_base, label_modelo) — leyenda bajo
                   la barra. Ejemplo: ("CatBoost", "Stacking").
                   Si None, no se muestra leyenda.

    Devuelve:
        str con el HTML de la tarjeta. Para renderizar:
            st.markdown(_tarjeta_kpi(...), unsafe_allow_html=True)
    """
    # Mapeo semántico de colores del delta
    _color_delta = {
        "red":   COLORES["abandono"],
        "green": COLORES["exito"],
        "gray":  COLORES["texto_muy_suave"],
        "":      COLORES["texto_muy_suave"],
    }.get(delta_color, COLORES["texto_muy_suave"])

    # Color de la barra lateral (default = primario azul, igual que p01)
    _color_barra = color_barra if color_barra else COLORES["primario"]

    # BUG FIX (Chat p03, 27/04/2026): antes generábamos DOS atributos style=
    # en el mismo div cuando había tooltip:
    #     <div title="..." style="cursor:help;" style="background:..."> ❌
    # El navegador solo respeta el PRIMER style= e ignora el segundo, así
    # que la barra lateral, fondo, padding y bordes desaparecían cuando
    # se pasaba un tooltip. Solución: tooltip y cursor van como atributos
    # separados (title="..." + cursor en el style único de la tarjeta).
    title_attr  = f' title="{tooltip}"' if tooltip else ""
    cursor_css  = "cursor:help;" if tooltip else ""

    # --- DELTA: cajita coloreada (estilo p00 unificado, 28/04/2026) ---
    # Si delta_color es 'red' o 'green', se renderiza en cajita coloreada
    # con fondo suave + flecha. Si '' o 'gray', texto suelto sin cajita.
    if delta:
        if delta_color in ("red", "green"):
            _bg_delta = ("rgba(16,185,129,0.10)" if delta_color == "green"
                         else "rgba(220,38,38,0.10)")
            _arrow = "▲ " if delta.lstrip().startswith("+") else (
                     "▼ " if delta.lstrip().startswith("-") else "")
            delta_html = (
                f'<div style="display:inline-block; font-size:0.78rem; '
                f'color:{_color_delta}; background:{_bg_delta}; '
                f'padding:0.15rem 0.5rem; border-radius:4px; '
                f'margin-top:0.35rem; font-weight:500; white-space:nowrap;">'
                f'{_arrow}{delta}</div>'
            )
        else:
            delta_html = (
                f'<div style="font-size:0.78rem; color:{_color_delta}; '
                f'margin-top:0.15rem; font-weight:500;">{delta}</div>'
            )
    else:
        delta_html = '<div style="font-size:0.78rem; margin-top:0.15rem;">&nbsp;</div>'

    # --- SPARKLINE OPCIONAL (Chat p00, 28/04/2026) ---
    # Mini-barra comparativa "antes → después". Solo aparece si se pasa
    # el parámetro sparkline=(val_base, val_modelo) en escala 0-1.
    sparkline_html = ""
    if sparkline is not None and len(sparkline) == 2:
        _vb, _vm = float(sparkline[0]), float(sparkline[1])
        _pct_b = round(_vb * 100, 1)
        _pct_m = round(_vm * 100, 1)
        _diff  = round((_vm - _vb) * 100, 1)
        _color_mejora = (COLORES["exito"] if _vm >= _vb
                         else COLORES["abandono"])
        sparkline_html = (
            f'<div style="margin-top:0.5rem; display:flex; align-items:center; gap:3px;">'
            f'<div style="width:{_pct_b}%; height:5px; '
            f'background:{COLORES["borde"]}; border-radius:3px 0 0 3px;"></div>'
            f'<div style="width:{abs(_diff)}%; height:5px; '
            f'background:{_color_mejora}; border-radius:0 3px 3px 0;"></div>'
            f'</div>'
        )
        if sparkline_labels and len(sparkline_labels) == 2:
            _lb, _lm = sparkline_labels
            sparkline_html += (
                f'<div style="font-size:0.68rem; color:{COLORES["texto_muy_suave"]}; '
                f'margin-top:0.2rem;">'
                f'{_lb}: {str(_pct_b).replace(".", ",")}% → '
                f'<span style="color:{_color_mejora}; font-weight:600;">'
                f'{_lm}: {str(_pct_m).replace(".", ",")}%</span>'
                f'</div>'
            )

    return (
        f'<div{title_attr} style="{cursor_css}'
        f'background:{COLORES["blanco"]};'
        f'border:1px solid {COLORES["borde"]};'
        f'border-left:4px solid {_color_barra};'
        f'border-radius:10px;'
        f'padding:0.85rem 1rem;box-shadow:0 1px 2px rgba(0,0,0,0.04);'
        f'height:100%;min-height:6.2rem;display:flex;'
        f'flex-direction:column;justify-content:space-between;">'
        f'<div style="display:flex;align-items:center;gap:0.5rem;">'
        f'<div style="font-size:1.4rem;line-height:1;flex-shrink:0;">{icono}</div>'
        f'<div style="font-size:0.72rem;color:{COLORES["texto_suave"]};font-weight:600;'
        f'text-transform:uppercase;letter-spacing:0.03em;line-height:1.2;">'
        f'{etiqueta}</div>'
        f'</div>'
        f'<div>'
        f'<div style="font-size:1.7rem;font-weight:700;color:{COLORES["texto"]};'
        f'line-height:1.1;margin-top:0.3rem;">{valor}</div>'
        f'{delta_html}'
        f'{sparkline_html}'
        f'</div>'
        f'</div>'
    )
