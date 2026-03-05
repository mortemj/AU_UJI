# ============================================================================
# GRAFICOS.PY - FUNCIONES DE VISUALIZACIÓN
# ============================================================================
# TFM: Predicción de Abandono Universitario
# Autora: María José Morte
# Ubicación: src/utils/graficos.py
# ============================================================================
# Tres funciones reutilizables para todos los notebooks.
# Paleta coherente con el proyecto.
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Dict
import io
import base64


# ============================================================================
# PALETA DE COLORES DEL PROYECTO
# ============================================================================

COLORES = {
    'primary': '#3182ce',
    'secondary': '#2c5282',
    'success': '#38a169',
    'warning': '#ed8936',
    'danger': '#e53e3e',
    'gray': '#718096',
}


# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

def _configurar_estilo():
    """Aplica estilo coherente a todos los gráficos."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10


# ============================================================================
# FUNCIONES PRINCIPALES
# ============================================================================

def histograma_con_kde(
    serie: pd.Series,
    titulo: str = "",
    xlabel: str = "",
    ylabel: str = "Frecuencia",
    color: str = COLORES['primary'],
    bins: int = 30,
    figsize: tuple = (6, 4)
) -> plt.Figure:
    """
    Histograma con curva KDE superpuesta.
    
    Parameters
    ----------
    serie : pd.Series
        Datos numéricos a graficar
    titulo : str
        Título del gráfico
    xlabel : str
        Etiqueta eje X
    ylabel : str
        Etiqueta eje Y
    color : str
        Color de las barras
    bins : int
        Número de bins
    figsize : tuple
        Tamaño de la figura
    
    Returns
    -------
    plt.Figure
        Figura de matplotlib
    
    Examples
    --------
    >>> fig = histograma_con_kde(df['edad'], titulo='Distribución de edad')
    >>> fig.savefig('edad.png')
    """
    _configurar_estilo()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filtrar NaN
    datos = serie.dropna()
    
    sns.histplot(datos, kde=True, color=color, bins=bins, ax=ax)
    
    ax.set_title(titulo, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    plt.tight_layout()
    return fig


def barras_categoricas(
    datos: Dict[str, float],
    titulo: str = "",
    xlabel: str = "",
    ylabel: str = "",
    color: str = COLORES['primary'],
    horizontal: bool = False,
    figsize: tuple = (6, 4)
) -> plt.Figure:
    """
    Gráfico de barras para datos categóricos.
    
    Parameters
    ----------
    datos : Dict[str, float]
        Diccionario {categoría: valor}
    titulo : str
        Título del gráfico
    xlabel, ylabel : str
        Etiquetas de los ejes
    color : str
        Color de las barras
    horizontal : bool
        Si True, barras horizontales
    figsize : tuple
        Tamaño de la figura
    
    Returns
    -------
    plt.Figure
        Figura de matplotlib
    
    Examples
    --------
    >>> datos = {'Hombre': 45.2, 'Mujer': 54.8}
    >>> fig = barras_categoricas(datos, titulo='Distribución por sexo')
    """
    _configurar_estilo()
    fig, ax = plt.subplots(figsize=figsize)
    
    categorias = list(datos.keys())
    valores = list(datos.values())
    
    if horizontal:
        ax.barh(categorias, valores, color=color)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
    else:
        ax.bar(categorias, valores, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    
    ax.set_title(titulo, fontweight='bold', pad=10)
    
    plt.tight_layout()
    return fig


def boxplot_outliers(
    serie: pd.Series,
    titulo: str = "",
    xlabel: str = "",
    color: str = COLORES['warning'],
    figsize: tuple = (6, 3)
) -> plt.Figure:
    """
    Boxplot horizontal para detectar outliers.
    
    Parameters
    ----------
    serie : pd.Series
        Datos numéricos
    titulo : str
        Título del gráfico
    xlabel : str
        Etiqueta del eje
    color : str
        Color del boxplot
    figsize : tuple
        Tamaño de la figura
    
    Returns
    -------
    plt.Figure
        Figura de matplotlib
    
    Examples
    --------
    >>> fig = boxplot_outliers(df['edad'], titulo='Outliers de edad')
    """
    _configurar_estilo()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filtrar NaN
    datos = serie.dropna()
    
    sns.boxplot(x=datos, color=color, ax=ax)
    
    ax.set_title(titulo, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel)
    
    plt.tight_layout()
    return fig


# ============================================================================
# UTILIDADES
# ============================================================================

def figura_a_base64(fig: plt.Figure, formato: str = 'png', dpi: int = 100) -> str:
    """
    Convierte una figura de matplotlib a string base64.
    
    Útil para incrustar gráficos en HTML.
    
    Parameters
    ----------
    fig : plt.Figure
        Figura de matplotlib
    formato : str
        Formato de imagen ('png', 'svg')
    dpi : int
        Resolución
    
    Returns
    -------
    str
        String base64 de la imagen
    
    Examples
    --------
    >>> fig = histograma_con_kde(df['edad'])
    >>> b64 = figura_a_base64(fig)
    >>> html = f'<img src="data:image/png;base64,{b64}" />'
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format=formato, dpi=dpi, bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def grafico_a_html(fig: plt.Figure, alt: str = "Gráfico") -> str:
    """
    Convierte figura a tag HTML <img> completo.
    
    Parameters
    ----------
    fig : plt.Figure
        Figura de matplotlib
    alt : str
        Texto alternativo
    
    Returns
    -------
    str
        Tag HTML <img> con la imagen embebida
    """
    b64 = figura_a_base64(fig)
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;" />'
