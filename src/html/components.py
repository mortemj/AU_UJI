# ============================================================================
# COMPONENTS.PY — COMPONENTES HTML REUTILIZABLES
# ============================================================================
# TFM: Predicción de Abandono Universitario
# Autora: María José Morte
# ============================================================================
# NOTA: Los datos de autora/email se importan de config_proyecto.
# NO están hardcodeados en las funciones.
# ============================================================================

from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path


# ============================================================================
# IMPORTAR DATOS DE IDENTIDAD (FUENTE ÚNICA)
# ============================================================================

try:
    from ..config_proyecto import AUTORA, EMAIL_UOC, EMAIL_UJI, GITHUB_REPO
    _CONFIG_DISPONIBLE = True
except ImportError:
    try:
        from src.config_proyecto import AUTORA, EMAIL_UOC, EMAIL_UJI, GITHUB_REPO
        _CONFIG_DISPONIBLE = True
    except ImportError:
        # Fallback solo si no se puede importar config_proyecto
        AUTORA = "María José Morte"
        EMAIL_UOC = "mjmorteruiz@uoc.edu"
        EMAIL_UJI = "morte@uji.es"
        GITHUB_REPO = "https://github.com/mortemj/AU_UJI"
        _CONFIG_DISPONIBLE = False


# ============================================================================
# HEADER (CABECERA)
# ============================================================================

def get_header_html(
    titulo: str,
    subtitulo: str,
    ruta_assets: str = '../../assets'
) -> str:
    """
    Genera cabecera con logos UOC/UJI y título dinámico.

    Parameters
    ----------
    titulo : str
        Título del módulo (ej: 'Reportes Clean')
    subtitulo : str
        Subtítulo con fase (ej: 'Fase 1: Transformación | TFM Abandono Universitario')
    ruta_assets : str
        Ruta relativa a la carpeta assets
    """
    return f'''
    <header class="header">
        <div class="header-logo">
            <img src="{ruta_assets}/logo_uoc.png" alt="Logo UOC">
        </div>
        <div class="header-title">
            <h1>{titulo}</h1>
            <p>{subtitulo}</p>
        </div>
        <div class="header-logo">
            <img src="{ruta_assets}/logo_uji.jpg" alt="Logo UJI">
        </div>
    </header>
    '''


# ============================================================================
# FOOTER CON ENLACES DINÁMICOS
# ============================================================================

def get_footer_html(
    autora: str = None,
    email_uoc: str = None,
    email_uji: str = None,
    notebook_url: str = None,
    github_repo: str = None
) -> str:
    """
    Genera footer con enlaces opcionales a notebook y GitHub.

    Los valores por defecto se toman de config_proyecto.py (fuente única).

    Parameters
    ----------
    autora : str, optional
        Nombre de la autora. Default: config_proyecto.AUTORA
    email_uoc : str, optional
        Email UOC. Default: config_proyecto.EMAIL_UOC
    email_uji : str, optional
        Email UJI. Default: config_proyecto.EMAIL_UJI
    notebook_url : str, optional
        URL completa al notebook en GitHub
    github_repo : str, optional
        URL del repositorio GitHub
    """
    # Usar valores de config_proyecto si no se pasan explícitamente
    autora = autora or AUTORA
    email_uoc = email_uoc or EMAIL_UOC
    email_uji = email_uji or EMAIL_UJI

    fecha = datetime.now().strftime('%d/%m/%Y %H:%M')

    # Enlaces dinámicos
    enlaces_html = ''
    if notebook_url or github_repo:
        enlaces = []
        if notebook_url:
            enlaces.append(f'<a href="{notebook_url}" target="_blank">📓 Ver Notebook</a>')
        if github_repo:
            enlaces.append(f'<a href="{github_repo}" target="_blank">🐙 GitHub</a>')
        enlaces_html = f'<p class="footer-links">{" | ".join(enlaces)}</p>'

    return f'''
    <footer class="footer">
        <p><strong>TFM: Predicción de Abandono Universitario</strong></p>
        <p>Autora: {autora} |
            <a href="mailto:{email_uoc}">{email_uoc}</a> (UOC) |
            <a href="mailto:{email_uji}">{email_uji}</a> (UJI)
        </p>
        {enlaces_html}
        <p>Generado: {fecha}</p>
    </footer>
    '''


# ============================================================================
# KPI
# ============================================================================

def get_kpi_html(valor: str, titulo: str, color: str = '#3182ce') -> str:
    """Genera una tarjeta KPI."""
    return f'''
    <div class="kpi">
        <div class="value" style="color:{color}">{valor}</div>
        <div class="label">{titulo}</div>
    </div>
    '''


def generar_kpis_html(kpis: List[Dict]) -> str:
    """
    Genera HTML para múltiples KPIs.

    Parameters
    ----------
    kpis : List[Dict]
        Lista de dicts con 'valor', 'titulo', 'color' (opcional)
    """
    colores_default = ['#3182ce', '#38a169', '#ed8936', '#e53e3e', '#805ad5', '#319795']

    html = '<div class="kpis">\n'
    for i, kpi in enumerate(kpis):
        color = kpi.get('color', colores_default[i % len(colores_default)])
        html += get_kpi_html(
            valor=kpi.get('valor', ''),
            titulo=kpi.get('titulo', ''),
            color=color
        )
    html += '</div>'
    return html


# ============================================================================
# TARJETAS
# ============================================================================

def generar_tarjeta_html(
    titulo: str,
    descripcion: str,
    emoji: str = '📄',
    link: str = '',
    link_texto: str = 'Ver más →',
    color: str = '#3182ce'
) -> str:
    """Genera una tarjeta individual con color específico."""
    if link:
        return f'''
        <a href="{link}" class="tarjeta" style="border-left-color: {color}">
            <h3 style="color: {color}">{emoji} {titulo}</h3>
            <p>{descripcion}</p>
            <div class="link" style="color: {color}">{link_texto}</div>
        </a>
        '''
    else:
        return f'''
        <div class="tarjeta" style="border-left-color: {color}">
            <h3 style="color: {color}">{emoji} {titulo}</h3>
            <p>{descripcion}</p>
        </div>
        '''


def generar_tarjetas_html(tarjetas: List[Dict]) -> str:
    """
    Genera HTML para múltiples tarjetas.

    Parameters
    ----------
    tarjetas : List[Dict]
        Lista de dicts con 'titulo', 'descripcion', 'emoji', 'link', 'link_texto', 'color'
    """
    html = '<div class="grid-tarjetas">\n'
    for t in tarjetas:
        html += generar_tarjeta_html(
            titulo=t.get('titulo', ''),
            descripcion=t.get('descripcion', ''),
            emoji=t.get('emoji', '📄'),
            link=t.get('link', ''),
            link_texto=t.get('link_texto', 'Ver más →'),
            color=t.get('color', '#3182ce')
        )
    html += '</div>'
    return html


# ============================================================================
# SECCIONES
# ============================================================================

def generar_seccion_html(titulo: str, contenido: str, icono: str = '') -> str:
    """Genera una sección con título y contenido."""
    icono_html = f'{icono} ' if icono else ''
    return f'''
    <section class="seccion">
        <h2>{icono_html}{titulo}</h2>
        {contenido}
    </section>
    '''


# ============================================================================
# TABLAS
# ============================================================================

def generar_tabla_html(
    cabeceras: List[str],
    filas: List[List[str]],
    fila_total: List[str] = None
) -> str:
    """Genera una tabla HTML."""
    html = '<table class="tabla">\n<thead>\n<tr>\n'
    for cab in cabeceras:
        html += f'<th>{cab}</th>\n'
    html += '</tr>\n</thead>\n<tbody>\n'

    for fila in filas:
        html += '<tr>\n'
        for celda in fila:
            html += f'<td>{celda}</td>\n'
        html += '</tr>\n'

    if fila_total:
        html += '<tr class="total-row">\n'
        for celda in fila_total:
            html += f'<td>{celda}</td>\n'
        html += '</tr>\n'

    html += '</tbody>\n</table>'
    return html


def generar_tabla_con_tooltip(
    datos: List[Dict], cabeceras: List[str] = None
) -> str:
    """
    Genera tabla HTML con tooltips que muestran las columnas de cada tabla.

    Parameters
    ----------
    datos : List[Dict]
        Lista de dicts con: 'nombre', 'filas', 'columnas', 'nulos_pct', 'duplicados', 'cols_lista'
    cabeceras : List[str], optional
        Cabeceras personalizadas.
    """
    if not cabeceras:
        cabeceras = ['Tabla', 'Filas', 'Columnas', '% Nulos', 'Duplicados']

    html = '<table>\n<thead>\n<tr>'
    for i, cab in enumerate(cabeceras):
        align = 'left' if i == 0 else ('right' if i in [1, 4] else 'center')
        html += f'<th style="text-align:{align}">{cab}</th>'
    html += '</tr>\n</thead>\n<tbody>\n'

    total_filas = 0
    total_cols = 0
    total_dups = 0

    for d in datos:
        tooltip = str(d.get('cols_lista', ''))
        nombre = d.get('nombre', '')
        filas = d.get('filas', 0)
        cols = d.get('columnas', 0)
        nulos = d.get('nulos_pct', '0.0%')
        dups = d.get('duplicados', 0)

        if isinstance(filas, int):
            total_filas += filas
        else:
            try:
                total_filas += int(str(filas).replace('.', '').replace(',', ''))
            except (ValueError, TypeError):
                pass
        total_cols += int(cols) if cols else 0
        total_dups += int(dups) if dups else 0

        filas_fmt = f'{filas:,}'.replace(',', '.') if isinstance(filas, int) else filas

        html += f'<tr>'
        html += f'<td style="text-align:left" class="tooltip-cell" data-tooltip="{tooltip}">📊 {nombre}</td>'
        html += f'<td style="text-align:right">{filas_fmt}</td>'
        html += f'<td style="text-align:center">{cols}</td>'
        html += f'<td style="text-align:center">{nulos}</td>'
        html += f'<td style="text-align:right">{dups}</td>'
        html += '</tr>\n'

    total_filas_fmt = f'{total_filas:,}'.replace(',', '.')
    html += f'<tr class="total-row">'
    html += f'<td style="text-align:left">TOTAL</td>'
    html += f'<td style="text-align:right">{total_filas_fmt}</td>'
    html += f'<td style="text-align:center">{total_cols}</td>'
    html += f'<td style="text-align:center">-</td>'
    html += f'<td style="text-align:right">{total_dups}</td>'
    html += '</tr>\n'

    html += '</tbody>\n</table>'
    return html


# ============================================================================
# MENSAJES
# ============================================================================

def generar_mensaje_html(texto: str, tipo: str = 'info') -> str:
    """
    Genera un mensaje con estilo.

    Parameters
    ----------
    texto : str
        Texto del mensaje
    tipo : str
        Tipo: 'info', 'ok', 'alerta', 'error'
    """
    iconos = {'info': 'ℹ️', 'ok': '✅', 'alerta': '⚠️', 'error': '❌'}
    icono = iconos.get(tipo, 'ℹ️')
    return f'<div class="mensaje mensaje-{tipo}">{icono} {texto}</div>'


# ============================================================================
# GUARDAR HTML
# ============================================================================

def guardar_html(html: str, ruta: Path) -> Path:
    """Guarda HTML en archivo, creando directorios si es necesario."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(html, encoding='utf-8')
    print(f'✅ HTML guardado: {ruta}')
    return ruta
