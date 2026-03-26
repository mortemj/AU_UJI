# Limitación del modelo: titulación vs rama de conocimiento

**TFM: Pronóstico del Éxito y del Abandono en los Títulos de Grado de la UJI**  
María José Morte Ruiz · UOC + UJI · 2026

---

## El problema

El modelo predice la misma probabilidad de abandono para dos titulaciones
de la misma rama (por ejemplo, Ingeniería Eléctrica e Ingeniería Mecánica),
aunque su tasa histórica de abandono sea muy diferente.

Esto es porque **`titulacion` no es una feature del modelo** — solo lo es `rama`.

---

## Por qué `titulacion` no está en el modelo

### 1. El dataset D_strict tiene 19 features, construidas a nivel de expediente

El pipeline de Feature Engineering (Fase 3) agrega los datos a nivel de
alumno/expediente. La variable `titulacion` tiene **alta cardinalidad**
(40 titulaciones) y **pocos datos por clase** (algunas titulaciones tienen
50-150 alumnos en el período 2010-2020).

Con tan pocos datos por titulación, el modelo sobreajustaría: aprendería
el comportamiento histórico de cada titulación en lugar de los factores
individuales del alumno.

### 2. Riesgo de leakage

La variable `titulacion` está correlacionada con el abandono porque
**las titulaciones difíciles atraen perfiles de más riesgo**. Incluirla
mezclaria la causa (el perfil del alumno) con el contexto (la titulación),
dificultando la defensa metodológica ante el tribunal.

### 3. La auditoría de leakage de Fase 3 ya eliminó 8 variables

Los casos A y B (AUC=1.0) demostraron que incluir variables con información
temporal o contextual demasiado cercana al target produce modelos que
"memorizan" en vez de generalizar. `titulacion` entraría en esa categoría
de riesgo por su alta correlación con el target.

### 4. La diferencia de rendimiento era mínima

Pasar de D (21 features, con `titulacion` como posibilidad) a D_strict
(19 features) solo costó 1.7 puntos de F1 (0.8138 → 0.7970) a cambio
de una defensa metodológica sólida.

---

## Comparativa: modelo con rama vs modelo con titulación

| Aspecto | Modelo actual (rama) | Modelo futuro (titulación) |
|---|---|---|
| Features | `rama` (5 categorías) | `titulacion` (40 categorías) |
| Datos por clase | ~6.700 alumnos/rama | ~150-800 alumnos/titulación |
| Riesgo sobreajuste | Bajo | Medio-alto |
| Diferenciación | Entre ramas | Entre titulaciones |
| Interacciones aprendidas | Nota×rama, beca×rama... | Nota×titulación, beca×titulación... |
| Datos mínimos necesarios | Los actuales (33.621) | ~5.000+ por titulación (no disponibles) |
| Defensa ante tribunal | ✅ Sólida | Requiere justificación extra |

---

## Lo que aprendería un modelo con `titulacion`

Si en el futuro se dispusiera de suficientes datos, incluir `titulacion`
permitiría aprender:

- **Interacciones multivariantes**: si la nota de acceso importa más en
  Medicina que en Historia, o si trabajar a tiempo parcial afecta más
  en Ingeniería que en Derecho
- **Efectos no lineales por titulación**: el umbral de riesgo no es el
  mismo en todas las titulaciones
- **Calibración automática**: el modelo ajustaría las probabilidades
  internamente sin necesidad de corrección post-hoc

---

## Solución actual: corrección heurística por tasa histórica (Opción A)

Dado que el modelo opera a nivel de rama, la app aplica una **corrección
post-hoc** para diferenciar entre titulaciones de la misma rama:

```
prob_ajustada = prob_modelo × (tasa_hist_titulacion / tasa_hist_rama)
```

**Ejemplo:**
- Modelo predice 20% para Ingeniería (rama)
- Ingeniería Eléctrica tiene 56% de abandono histórico
- Media de Ingeniería y Arquitectura tiene 35% de abandono histórico
- Factor = 56% / 35% = 1.60
- Prob ajustada = 20% × 1.60 = **32%**

### Limitaciones de esta corrección

1. Es **univariante**: solo usa la tasa histórica, no considera interacciones
2. Es **lineal**: asume que el factor de escala es constante para cualquier perfil
3. Puede **sobreestimar** en perfiles extremos (si el factor es muy alto)
4. Se aplica **post-hoc**, no es parte del entrenamiento del modelo

### Aviso en la app

La app muestra explícitamente:

> *"La probabilidad base del modelo opera a nivel de rama de conocimiento.
> El valor mostrado incluye un ajuste por la tasa histórica de abandono
> de esta titulación específica (período 2010-2020), que no forma parte
> del modelo entrenado."*

---

## Línea de mejora futura

Para una versión futura con predicción real por titulación:

### Opción 1: Más datos
Esperar a tener suficientes cohortes (recomendado: 10+ años por titulación,
mínimo 500 alumnos por clase). Incluir `titulacion` como feature categórica
con target encoding o embeddings.

### Opción 2: Modelos jerárquicos
Entrenar un modelo global (rama) + modelos locales por titulación
que refinan la predicción. Los modelos locales solo se activan si
hay suficientes datos para esa titulación.

### Opción 3: Transfer learning
Usar el modelo de rama como base y hacer fine-tuning por titulación
con los pocos datos disponibles (few-shot approach).

---

## Referencia en el TFM

- **Fase 3, M08**: Auditoría de leakage — justificación de D_strict (19 features)
- **Fase 5, M07**: Comparativa final de modelos — modelo elegido (Stacking)
- **Fase 6**: Interpretabilidad y fairness — análisis por titulación con SHAP

---

*Fichero generado como documentación de decisión de diseño.*  
*TFM: Pronóstico del Éxito y del Abandono en los Títulos de Grado de la UJI*  
*María José Morte Ruiz · mjmorteruiz@uoc.edu · morte@uji.es · 2026*
