# Análisis de Cambios de Plan en Titulaciones de la UJI

**Proyecto:** TFM — Predicción de Abandono Universitario en la UJI  
**Autora:** María José Morte Ruiz  
**Fecha:** Marzo 2025  
**Fichero de datos:** `parejas_planes_abandono.xlsx`  
**Ubicación:** `docs/notas_analiticas/`

---

## 1. Contexto

Durante la exploración del dataset se detectó que varias titulaciones aparecen con
**más de un nombre** en los datos. Esto responde a tres fenómenos distintos:

| Tipo | Descripción | Ejemplo |
|------|-------------|---------|
| **Cambio de nombre** | La carrera se renombra sin cambiar el plan de estudios | Ing. de la Edificación → Arquitectura Técnica |
| **Renovación de plan** | Se mantiene el nombre pero se actualiza el plan curricular | Medicina → Medicina (Plan 2017) |
| **Error de datos** | Nombre con coma al final por error de entrada | "...y Derecho," |

Cuando una titulación renueva su plan, la normativa permite a los alumnos matriculados
en el plan antiguo **hasta 2 años adicionales** para superar las asignaturas pendientes
sin asistir a clase. Esto genera un **solapamiento temporal** entre plan viejo y plan nuevo.

---

## 2. Parejas identificadas

Se identificaron **9 familias** de titulaciones con nombres múltiples:

| Nombre unificado | Plan viejo | Plan nuevo | Tipo |
|-----------------|------------|------------|------|
| Arquitectura Técnica | Ing. de la Edificación (2009–2011) | Arquitectura Técnica (Plan 2020) | Cambio nombre + plan |
| Criminología y Seguridad | 2010–2019 | Plan 2020 | Renovación plan |
| Historia y Patrimonio | 2010–2015 | Plan 2015 (2011–2020) | Renovación plan |
| Ing. Agroalimentaria y del Medio Rural | 2010–2017 | Plan 2018 (2012–2020) | Renovación plan |
| Maestro en Educación Infantil | 2010–2017 | Plan 2018 (2012–2020) | Renovación plan |
| Maestro en Educación Primaria | 2010–2018 | Plan 2018 (2014–2020) | Renovación plan |
| Medicina | 2011–2016 | Plan 2017 (2014–2020) | Renovación plan |
| Ing. en Diseño Industrial y Desarrollo de Productos | 2010–2020 | Plan 2020 | Renovación plan |
| Doble Grado ADE y Derecho | "...y Derecho," | "...y Derecho" | Error de datos |

---

## 3. Tasas de abandono por plan

El análisis sobre el conjunto de test (20% del dataset, 6.725 observaciones) revela
diferencias notables en la tasa de abandono entre plan viejo y plan nuevo:

| Titulación | Plan viejo | Plan nuevo | Diferencia | Interpretación |
|------------|-----------|-----------|------------|----------------|
| Historia y Patrimonio | **52.9%** | **20.4%** | −32.5 pp | Mejora real del plan + truncamiento |
| Ing. Agroalimentaria | **52.9%** | **5.9%** | −47.0 pp | Mejora real del plan + truncamiento |
| Maestro Ed. Primaria | **22.6%** | **3.3%** | −19.3 pp | Truncamiento dominante |
| Maestro Ed. Infantil | **15.0%** | **2.1%** | −12.9 pp | Truncamiento dominante |
| Medicina | **11.1%** | **3.0%** | −8.1 pp | Truncamiento dominante |
| Criminología | **20.0%** | **0.0%** | −20.0 pp | Solo truncamiento (plan 2020, 1 año) |
| Arquitectura Técnica | **33.0%** | **0.0%** | −33.0 pp | Solo truncamiento (plan 2020, 1 año) |

> **Nota sobre truncamiento temporal:** los alumnos del plan nuevo llevan pocos años en el
> dataset (máximo hasta 2020). Un alumno de cohorte 2018 solo tiene 2 años de observación,
> insuficiente para que se materialice el abandono que típicamente ocurre en años 3–4.
> Por ello, las tasas del plan nuevo están **subestimadas** y no son directamente comparables
> con las del plan viejo.

---

## 4. Decisión de tratamiento

### Decisión adoptada: **Unificación con flag `cambio_plan`**

Se unifican todos los nombres bajo el nombre actual de la titulación, añadiendo
una columna `cambio_plan` que indica el estado de cada alumno:

| Valor | Descripción |
|-------|-------------|
| `plan_original` | Alumno matriculado en el plan vigente desde el inicio |
| `transicion` | Alumno matriculado durante el solapamiento de los 2 planes |
| `plan_nuevo` | Alumno matriculado ya en el plan renovado |

### ¿Por qué no separar las carreras?

- Para el tribunal y los gestores, son la **misma titulación**
- Los patrones de abandono son **estructuralmente comparables**
- La app gana **claridad**: una entrada por carrera en todos los selectores
- El flag permite análisis específicos si se necesitan

### ¿Por qué no unificar sin flag?

- Las diferencias de abandono entre planes son **grandes en algunos casos** (hasta 47 pp)
- Unificar sin señalar el cambio distorsionaría el abandono medio mostrado en la app
- El flag permite al analista **filtrar o ponderar** según necesidad

---

## 5. Implementación

La unificación se aplica en **`app/utils/loaders.py`**, función `cargar_meta_test_app()`,
antes de devolver el DataFrame a la app. Así el cambio es transparente para todas
las pestañas (p01, p02, p03...) sin tocar el dataset original.

```python
# Mapa de unificación de nombres
MAPA_TITULACIONES = {
    "Grado en Ingeniería de la Edificación":                              "Grado en Arquitectura Técnica",
    "Grado en Arquitectura Técnica (Plan 2020)":                          "Grado en Arquitectura Técnica",
    "Grado en Criminologia y Seguridad  (Plan 2020)":                     "Grado en Criminologia y Seguridad",
    "Grado en Historia y Patrimonio (Plan 2015)":                         "Grado en Historia y Patrimonio",
    "Grado en Ingeniería Agroalimentaria y del Medio Rural (Plan 2018)":  "Grado en Ingeniería Agroalimentaria y del Medio Rural",
    "Grado en Maestro en Educación Infantil (Plan 2018)":                 "Grado en Maestro en Educación Infantil",
    "Grado en Maestro en Educación Primaria (Plan 2018)":                 "Grado en Maestro en Educación Primaria",
    "Grado en Medicina (Plan 2017)":                                      "Grado en Medicina",
    "Doble Grado en Administración y Dirección de Empresas y Derecho,":   "Doble Grado en Administración y Dirección de Empresas y Derecho",
}
```

---

## 6. Ficheros relacionados

| Fichero | Descripción |
|---------|-------------|
| `parejas_planes_abandono.xlsx` | Tasas de abandono por plan (conjunto de test) |
| `titulaciones_revision.xlsx` | Tabla completa de titulaciones con inicio, fin y tipo |
| `app/utils/loaders.py` | Implementación de la unificación |
| `data/06_evaluacion/meta_test.parquet` | Dataset de evaluación con `titulacion` y `abandono` |

---

*Nota: este análisis se realizó sobre el conjunto de test (20% del dataset).
Los patrones observados son consistentes con el dataset completo pero los
conteos de alumnos son menores.*
