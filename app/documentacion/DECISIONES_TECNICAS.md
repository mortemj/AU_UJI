# =============================================================================
# DECISIONES_TECNICAS.md
# Explicación de decisiones técnicas de la app Streamlit (Fase 7)
#
# Para qué sirve este fichero:
#   Si dentro de 3 meses (o 3 años) alguien revisa el código y se pregunta
#   "¿por qué se hizo así?", aquí está la respuesta.
#   No es documentación de uso — es documentación de decisiones.
# =============================================================================

---

## 1. ¿Por qué `cargar_meta_test()` carga DOS ficheros y los une?

### El contexto
En Fase 5 y Fase 6 se generaron dos ficheros separados con los datos de test:

| Fichero | Dónde | Qué contiene | Para qué se usó |
|---|---|---|---|
| `X_test_prep.parquet` | `data/05_modelado/` | 19 features preprocesadas | Entrenar y evaluar el modelo |
| `meta_test.parquet` | `data/06_evaluacion/` | Metadatos: titulación, rama, sexo, abandono... | Análisis de interpretabilidad en Fase 6 |

Ambos tienen **6.725 filas con el mismo índice** — están perfectamente alineados.

### El problema
La app necesita las dos cosas a la vez:
- Las **features** → para que el pipeline calcule probabilidades de abandono
- Los **metadatos** → para mostrar titulación, rama, sexo en los gráficos

Si pasas solo `meta_test` al pipeline, falla porque el pipeline no conoce
"titulación" ni "rama" — solo conoce las 19 features numéricas/categóricas
del modelo.

### La solución
`cargar_meta_test()` en `loaders.py` carga ambos ficheros y los une por índice
antes de devolver el resultado:

```python
meta  = pd.read_parquet(ruta_meta)   # metadatos
xtest = pd.read_parquet(ruta_xtest)  # features
df    = xtest.join(meta[cols_solo_meta])  # unión por índice
```

### ¿Es seguro este join?
Sí. Los índices son idénticos — verificado con diagnóstico el 23/03/2026:
```
Índices X_test_prep: [16823, 4904, 11906, 25232, 21637]
Índices meta_test:   [16823, 4904, 11906, 25232, 21637]
```

### ¿Afecta a los notebooks de Fase 5 o Fase 6?
No. No tocamos ningún notebook ni ningún fichero de datos. Solo cambia
cómo la app los lee. Los ficheros originales quedan intactos.

---

## 2. ¿Por qué existe `_path_setup.py`?

Streamlit en Windows (especialmente con OneDrive) ejecuta las páginas de
`pages/` desde un directorio de trabajo que no siempre es `app/`. Esto hace
que `from utils.loaders import ...` falle con `ModuleNotFoundError`.

`_path_setup.py` usa `os.path.abspath(__file__)` para calcular la ruta
a `app/` de forma robusta y añadirla a `sys.path`. Cada página lo importa
al inicio con `import _path_setup`.

---

## 3. ¿Por qué el proyecto funciona en cualquier disco o carpeta?

Todos los ficheros usan rutas relativas calculadas en tiempo de ejecución:

- `config_app.py` → sube niveles desde su propia ubicación hasta encontrar `src/`
- `_path_setup.py` → usa `os.path.abspath(__file__)` 
- `lanzar_app_windows.bat` → usa `%~dp0` (ruta del propio .bat)
- `lanzar_app_mac.sh` → usa `${BASH_SOURCE[0]}`

Ningún fichero tiene rutas hardcodeadas. El proyecto funciona igual en
`C:\`, `D:\`, `E:\proyectos\` o cualquier otra ruta.

---

## 4. ¿Por qué `pages/_old/` existe?

Es una copia de seguridad de las versiones anteriores de las páginas,
guardada por precaución durante el desarrollo. No afecta a la app.
Se llama `_old` con guión bajo para que Python no la procese como módulo.

---

## 5. Estructura de ficheros de la app

```
app/
├── main.py                  Punto de entrada — navegación y enrutamiento
├── config_app.py            Configuración central — rutas, colores, pestañas
├── _path_setup.py           Fix de paths para Windows/OneDrive
├── pages/
│   ├── p00_inicio.py        Pantalla de bienvenida
│   ├── p01_institucional.py Visión global UJI (gestores)
│   ├── p02_titulacion.py    Análisis por titulación (profesores)
│   ├── p03_prospecto.py     Pronóstico antes de matricularse
│   ├── p04_en_curso.py      Pronóstico alumno matriculado
│   └── p05_equidad.py       Fairness y simulador de política
└── utils/
    ├── loaders.py           Carga de datos y modelos con caché
    └── pronostico_shared.py Lógica compartida entre p03 y p04
```

---

## 6. Ficheros de datos que usa la app

| Clave en RUTAS | Fichero | Generado en |
|---|---|---|
| `modelo` | `data/05_modelado/models/CatBoost__balanced.pkl` | Fase 5 |
| `pipeline` | `data/05_modelado/pipeline_preprocesamiento.pkl` | Fase 5 |
| `X_test_prep` | `data/05_modelado/X_test_prep.parquet` | Fase 5 |
| `meta_test` | `data/06_evaluacion/meta_test.parquet` | Fase 6 |
| `shap_global` | `results/fase6/shap_global_catboost.pkl` | Fase 6 |
| `fairness` | `results/fase6/fairness_metricas.parquet` | Fase 6 |

---

*Última actualización: 23/03/2026 — María José Morte Ruiz*
