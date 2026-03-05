"""
src/display.py - Visualizaciones animadas para Jupyter Notebooks

TFM: Predicción de Abandono Universitario
Autora: María José Morte

Funciones modulares con animaciones CSS para mostrar banners, 
progreso y mensajes de forma muy visual en notebooks.
"""

from IPython.display import display, HTML, clear_output
import time


# =============================================================================
# CONFIGURACIÓN DE COLORES
# =============================================================================

COLORES = {
    'azul': {'gradient': 'linear-gradient(135deg, #3182ce 0%, #2c5282 100%)', 'solid': '#3182ce', 'light': '#bee3f8', 'dark': '#2c5282'},
    'verde': {'gradient': 'linear-gradient(135deg, #38a169 0%, #2f855a 100%)', 'solid': '#38a169', 'light': '#c6f6d5', 'dark': '#276749'},
    'naranja': {'gradient': 'linear-gradient(135deg, #ed8936 0%, #c05621 100%)', 'solid': '#ed8936', 'light': '#feebc8', 'dark': '#c05621'},
    'morado': {'gradient': 'linear-gradient(135deg, #805ad5 0%, #553c9a 100%)', 'solid': '#805ad5', 'light': '#e9d8fd', 'dark': '#553c9a'},
    'rosa': {'gradient': 'linear-gradient(135deg, #ed64a6 0%, #97266d 100%)', 'solid': '#ed64a6', 'light': '#fed7e2', 'dark': '#97266d'},
    'rojo': {'gradient': 'linear-gradient(135deg, #e53e3e 0%, #c53030 100%)', 'solid': '#e53e3e', 'light': '#fed7d7', 'dark': '#c53030'},
    'cyan': {'gradient': 'linear-gradient(135deg, #319795 0%, #234e52 100%)', 'solid': '#319795', 'light': '#b2f5ea', 'dark': '#234e52'},
}


# =============================================================================
# ESTILOS CSS ANIMADOS
# =============================================================================

CSS_ANIMACIONES = """
<style>
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}

@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

@keyframes progress {
    0% { width: 0%; }
    100% { width: 100%; }
}

@keyframes bounce {
    0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
    40% { transform: translateY(-10px); }
    60% { transform: translateY(-5px); }
}

@keyframes glow {
    0%, 100% { box-shadow: 0 0 5px rgba(255,255,255,0.5); }
    50% { box-shadow: 0 0 20px rgba(255,255,255,0.8), 0 0 30px rgba(255,255,255,0.6); }
}

.animate-fadeInUp {
    animation: fadeInUp 0.6s ease-out forwards;
}

.animate-pulse {
    animation: pulse 2s ease-in-out infinite;
}

.animate-bounce {
    animation: bounce 1s ease infinite;
}

.animate-glow {
    animation: glow 2s ease-in-out infinite;
}
</style>
"""


# =============================================================================
# BANNER PRINCIPAL CON ANIMACIÓN
# =============================================================================

def mostrar_banner(titulo: str, subtitulo: str = '', icono: str = '📊', 
                   color: str = 'azul', animacion: bool = True) -> None:
    """
    Muestra un banner decorativo animado.
    
    Args:
        titulo: Texto principal
        subtitulo: Texto secundario
        icono: Emoji
        color: 'azul', 'verde', 'naranja', 'morado', 'rosa', 'cyan'
        animacion: Si True, aplica animación de entrada
    """
    colores = COLORES.get(color, COLORES['azul'])
    anim_class = 'animate-fadeInUp' if animacion else ''
    
    subtitulo_html = f'''
        <p style="color: rgba(255,255,255,0.9); margin: 12px 0 0 0; text-align: center; font-size: 1.1em;">
            {subtitulo}
        </p>
    ''' if subtitulo else ''
    
    html = f'''
    {CSS_ANIMACIONES}
    <div class="{anim_class}" style="
        background: {colores['gradient']}; 
        padding: 30px 40px; 
        border-radius: 16px; 
        margin: 20px 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(
                45deg,
                transparent 30%,
                rgba(255,255,255,0.1) 50%,
                transparent 70%
            );
            animation: shimmer 3s infinite;
        "></div>
        <h2 style="
            color: white; 
            margin: 0; 
            text-align: center; 
            font-size: 1.8em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
            position: relative;
        ">
            <span style="font-size: 1.3em; margin-right: 10px;">{icono}</span> {titulo}
        </h2>
        {subtitulo_html}
    </div>
    '''
    display(HTML(html))


# =============================================================================
# BARRA DE PROGRESO ANIMADA
# =============================================================================

def mostrar_progreso_animado(titulo: str, descripcion: str = '', icono: str = '⏳', 
                              color: str = 'azul') -> None:
    """
    Muestra una barra de progreso animada.
    """
    colores = COLORES.get(color, COLORES['azul'])
    
    html = f'''
    {CSS_ANIMACIONES}
    <div class="animate-fadeInUp" style="
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid {colores['solid']};
    ">
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <span style="font-size: 1.8em; margin-right: 12px;" class="animate-bounce">{icono}</span>
            <div>
                <strong style="color: {colores['dark']}; font-size: 1.1em;">{titulo}</strong>
                <p style="margin: 5px 0 0 0; color: #718096; font-size: 0.9em;">{descripcion}</p>
            </div>
        </div>
        <div style="
            background: #e2e8f0;
            border-radius: 10px;
            height: 8px;
            overflow: hidden;
        ">
            <div style="
                background: {colores['gradient']};
                height: 100%;
                border-radius: 10px;
                animation: progress 2s ease-in-out infinite;
            "></div>
        </div>
    </div>
    '''
    display(HTML(html))


# =============================================================================
# MENSAJE DE COMPLETADO CON EFECTO
# =============================================================================

def mostrar_completado(mensaje: str, tiempo: float = None, color: str = 'verde') -> None:
    """
    Muestra mensaje de completado con efecto de éxito.
    """
    colores = COLORES.get(color, COLORES['verde'])
    
    tiempo_html = f'''
        <span style="
            background: rgba(255,255,255,0.3);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            margin-left: 10px;
        ">⏱️ {tiempo:.1f}s</span>
    ''' if tiempo else ''
    
    html = f'''
    {CSS_ANIMACIONES}
    <div class="animate-fadeInUp" style="
        background: {colores['gradient']};
        color: white;
        padding: 18px 25px;
        border-radius: 12px;
        margin: 15px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    ">
        <div style="display: flex; align-items: center;">
            <span style="font-size: 1.5em; margin-right: 12px;" class="animate-pulse">✅</span>
            <span style="font-size: 1.05em;">{mensaje}</span>
        </div>
        {tiempo_html}
    </div>
    '''
    display(HTML(html))


# =============================================================================
# TARJETAS DE ETAPAS
# =============================================================================

def mostrar_etapa(numero: int, titulo: str, descripcion: str, icono: str = '📋',
                  color: str = 'azul', estado: str = 'pendiente') -> None:
    """
    Muestra una tarjeta de etapa del proceso.
    
    Args:
        estado: 'pendiente', 'ejecutando', 'completado'
    """
    colores = COLORES.get(color, COLORES['azul'])
    
    # Estilos según estado
    estados = {
        'pendiente': {
            'bg': '#f7fafc',
            'border': '#e2e8f0',
            'opacity': '0.7',
            'badge_bg': '#e2e8f0',
            'badge_color': '#718096'
        },
        'ejecutando': {
            'bg': colores['light'],
            'border': colores['solid'],
            'opacity': '1',
            'badge_bg': colores['solid'],
            'badge_color': 'white'
        },
        'completado': {
            'bg': '#f0fff4',
            'border': '#38a169',
            'opacity': '1',
            'badge_bg': '#38a169',
            'badge_color': 'white'
        }
    }
    
    est = estados.get(estado, estados['pendiente'])
    anim = 'animate-pulse' if estado == 'ejecutando' else ''
    check = '✓' if estado == 'completado' else numero
    
    html = f'''
    {CSS_ANIMACIONES}
    <div class="animate-fadeInUp {anim}" style="
        background: {est['bg']};
        border: 2px solid {est['border']};
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        opacity: {est['opacity']};
        display: flex;
        align-items: center;
        gap: 15px;
        transition: all 0.3s ease;
    ">
        <div style="
            background: {est['badge_bg']};
            color: {est['badge_color']};
            width: 45px;
            height: 45px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 1.2em;
            flex-shrink: 0;
        ">{check}</div>
        <div style="flex: 1;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 1.3em;">{icono}</span>
                <strong style="font-size: 1.1em; color: #2d3748;">{titulo}</strong>
            </div>
            <p style="margin: 5px 0 0 0; color: #718096; font-size: 0.9em;">{descripcion}</p>
        </div>
    </div>
    '''
    display(HTML(html))


# =============================================================================
# RESUMEN FINAL CON ESTADÍSTICAS
# =============================================================================

def mostrar_resumen_final(titulo: str, stats: list, icono: str = '🎉', 
                          color: str = 'verde') -> None:
    """
    Muestra resumen final con estadísticas animadas.
    
    Args:
        stats: Lista de diccionarios con 'valor', 'titulo', 'icono'
    """
    colores = COLORES.get(color, COLORES['verde'])
    
    stats_html = ''
    for i, stat in enumerate(stats):
        delay = i * 0.1
        stats_html += f'''
        <div style="
            text-align: center;
            padding: 15px;
            animation: fadeInUp 0.5s ease-out {delay}s forwards;
            opacity: 0;
        ">
            <div style="font-size: 1.5em; margin-bottom: 5px;">{stat.get('icono', '📊')}</div>
            <div style="font-size: 1.8em; font-weight: bold; color: {colores['dark']};">
                {stat['valor']}
            </div>
            <div style="font-size: 0.85em; color: #718096;">{stat['titulo']}</div>
        </div>
        '''
    
    html = f'''
    {CSS_ANIMACIONES}
    <div class="animate-fadeInUp" style="
        background: {colores['gradient']};
        border-radius: 16px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    ">
        <h3 style="
            color: white;
            text-align: center;
            margin: 0 0 20px 0;
            font-size: 1.4em;
        ">
            <span class="animate-bounce" style="display: inline-block; margin-right: 10px;">{icono}</span>
            {titulo}
        </h3>
        <div style="
            background: white;
            border-radius: 12px;
            padding: 20px;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 10px;
        ">
            {stats_html}
        </div>
    </div>
    '''
    display(HTML(html))


# =============================================================================
# PIPELINE VISUAL
# =============================================================================

def mostrar_pipeline(etapas: list, etapa_actual: int = 0) -> None:
    """
    Muestra un pipeline visual con etapas.
    
    Args:
        etapas: Lista de diccionarios con 'titulo', 'icono'
        etapa_actual: Índice de la etapa actual (0-based)
    """
    etapas_html = ''
    for i, etapa in enumerate(etapas):
        if i < etapa_actual:
            estado = 'completado'
            bg = '#38a169'
            border = '#38a169'
        elif i == etapa_actual:
            estado = 'actual'
            bg = '#3182ce'
            border = '#3182ce'
        else:
            estado = 'pendiente'
            bg = '#e2e8f0'
            border = '#cbd5e0'
        
        anim = 'animate-pulse' if estado == 'actual' else ''
        
        etapas_html += f'''
        <div style="display: flex; align-items: center;">
            <div class="{anim}" style="
                width: 50px;
                height: 50px;
                border-radius: 50%;
                background: {bg};
                border: 3px solid {border};
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.3em;
                color: {'white' if estado != 'pendiente' else '#718096'};
                box-shadow: {'0 0 15px rgba(49,130,206,0.5)' if estado == 'actual' else 'none'};
            ">
                {'✓' if estado == 'completado' else etapa.get('icono', '📋')}
            </div>
            <span style="
                margin-left: 10px;
                font-weight: {'bold' if estado == 'actual' else 'normal'};
                color: {'#2d3748' if estado != 'pendiente' else '#a0aec0'};
            ">{etapa['titulo']}</span>
        </div>
        '''
        
        # Añadir flecha entre etapas
        if i < len(etapas) - 1:
            etapas_html += '''
            <div style="
                flex: 1;
                height: 3px;
                background: #e2e8f0;
                margin: 0 15px;
                position: relative;
            ">
                <div style="
                    position: absolute;
                    right: -8px;
                    top: -5px;
                    color: #cbd5e0;
                ">▶</div>
            </div>
            '''
    
    html = f'''
    {CSS_ANIMACIONES}
    <div class="animate-fadeInUp" style="
        background: white;
        border-radius: 12px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    ">
        <div style="
            display: flex;
            align-items: center;
            justify-content: space-between;
        ">
            {etapas_html}
        </div>
    </div>
    '''
    display(HTML(html))


# =============================================================================
# SPINNER DE CARGA
# =============================================================================

def mostrar_spinner(mensaje: str = 'Procesando...', color: str = 'azul') -> None:
    """
    Muestra un spinner de carga animado.
    """
    colores = COLORES.get(color, COLORES['azul'])
    
    html = f'''
    {CSS_ANIMACIONES}
    <div style="
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        gap: 15px;
    ">
        <div style="
            width: 40px;
            height: 40px;
            border: 4px solid #e2e8f0;
            border-top-color: {colores['solid']};
            border-radius: 50%;
            animation: spin 1s linear infinite;
        "></div>
        <span style="color: #4a5568; font-size: 1.1em;">{mensaje}</span>
    </div>
    '''
    display(HTML(html))
