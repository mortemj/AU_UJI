# 🎓 Predicción de Abandono y Éxito Académico en la UJI

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-yellowgreen?logo=scikit-learn)](https://scikit-learn.org/)
[![Estado](https://img.shields.io/badge/Estado-En%20desarrollo-yellow)]()
[![Licencia](https://img.shields.io/badge/Licencia-Académica-lightgrey)]()

> Trabajo Final de Máster (TFM) — Universitat Oberta de Catalunya (UOC)  
> Datos: Universitat Jaume I (UJI) · Período 2010–2021

---

## 📌 Descripción

Este proyecto desarrolla un sistema de **predicción temprana del abandono universitario** a partir de datos académicos, demográficos y administrativos de la Universitat Jaume I (UJI).

El objetivo es identificar estudiantes en riesgo de abandono antes de que este se produzca, con el fin de facilitar intervenciones institucionales efectivas. El modelo final combina técnicas de machine learning avanzadas con análisis interpretativo para ofrecer resultados accionables.

**Definición de abandono aplicada:**
> Estudiante que no ha egresado, no ha completado créditos suficientes para considerarse egresado de hecho, y lleva 2 o más años sin actividad académica.

---

## 🌐 Resultados y Web

👉 **[Ver resultados en GitHub Pages](https://mortemj.github.io/AU_UJI/)**

---

## 🗂️ Estructura del Proyecto

```
AU_UJI/
│
├── 📁 docs/               # Informes HTML por fase (visualización web)
├── 📁 notebooks/          # Jupyter Notebooks organizados por fase
├── 📁 notes/              # Notas y documentación de trabajo
├── 📁 src/                # Módulos Python reutilizables
│   ├── config/            # Configuración centralizada
│   └── utils/             # Funciones auxiliares
├── 📁 data/
│   ├── 00_raw/            # Datos originales (sin procesar)
│   ├── 01_interim/        # Datos intermedios
│   ├── 02_processed/      # Datos procesados
│   └── 03_features/       # Dataset final para modelado
└── README.md
```

---

## 🔬 Fases del Proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| **Fase 1** · Ingesta y Calidad | Carga de 9 tablas fuente, auditoría de calidad, limpieza y trazabilidad de datos | ✅ Completada |
| **Fase 2** · EDA Inicial | Análisis exploratorio univariante y bivariante con Plotly | ✅ Completada |
| **Fase 3** · Agregación | Construcción del dataset analítico por estudiante, definición del target `abandono` | ✅ Completada |
| **Fase 4** · EDA Final | Distribuciones numéricas y categóricas, detección de anomalías, visualizaciones interactivas | ✅ Completada |
| **Fase 5** · Modelado | Entrenamiento de 21 algoritmos en 7 familias, comparación cruzada, selección del mejor modelo | ✅ Completada |
| **Fase 6** · Interpretabilidad | SHAP, análisis por subgrupos demográficos (`sexo`, `via_acceso`, `titulacion`) | 🔄 En progreso |

---

## 🏆 Resultados del Modelado (Fase 5)

| Modelo | AUC CV | Observaciones |
|--------|--------|---------------|
| **Stacking** | **0.9308** | Mejor modelo global |
| EBM (InterpretML) | 0.9202 | Mejor modelo interpretable |
| Random Forest | ~0.91 | Baseline robusto |

Dataset de trabajo: **D_strict** · 33.621 registros · 19 features + target `abandono`

---

## 🛠️ Tecnologías

### Lenguajes y entornos
![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white&style=flat)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?logo=jupyter&logoColor=white&style=flat)
![Anaconda](https://img.shields.io/badge/-Anaconda-44A833?logo=anaconda&logoColor=white&style=flat)

### Machine Learning
![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?logo=scikit-learn&logoColor=white&style=flat)
![XGBoost](https://img.shields.io/badge/-XGBoost-EC4A28?style=flat)
![LightGBM](https://img.shields.io/badge/-LightGBM-02569B?style=flat)
![InterpretML](https://img.shields.io/badge/-InterpretML-5C2D91?style=flat)

### Visualización
![Plotly](https://img.shields.io/badge/-Plotly-3F4F75?logo=plotly&logoColor=white&style=flat)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat)
![Seaborn](https://img.shields.io/badge/-Seaborn-4C72B0?style=flat)

### Infraestructura
![GitHub](https://img.shields.io/badge/-GitHub-181717?logo=github&logoColor=white&style=flat)
![GitHub Pages](https://img.shields.io/badge/-GitHub%20Pages-222222?logo=github&logoColor=white&style=flat)

---

## 📊 Datos

- **Fuente:** Universitat Jaume I (UJI) — datos anonimizados
- **Período:** 2010–2021
- **Universo:** ~30.872 estudiantes únicos · 42 titulaciones de grado
- **Variables originales:** 9 tablas fuente · 37+ columnas
- **Dataset final:** 33.621 registros · 19 features + target

> ⚠️ Los datos son de carácter académico y uso restringido. No se incluyen en el repositorio.

---

## 👩‍💻 Autora

**María José Morte Ruiz**  
Máster en Ciencia de Datos — Universitat Oberta de Catalunya (UOC)  
📧 morte@uji.es

Tutora: Raúl Parada · rparada@uoc.edu

---

## 📄 Licencia

Proyecto académico. Datos proporcionados por la UJI bajo acuerdo de confidencialidad.  
Uso restringido a fines de investigación y formación.
