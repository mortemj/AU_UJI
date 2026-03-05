# ✅ Checklist de Tareas

**TFM: Predicción de Abandono Universitario**  
**Autora:** María José Morte

---

## 🔧 Configuración inicial

- [ ] Descomprimir ZIP de estructura
- [ ] Copiar Excel a `data/01_raw/`
- [ ] Copiar logos a `docs/assets/` y `docs/html/assets/`
- [ ] Verificar imports: `from src.config import info_entorno`
- [ ] Ejecutar `info_entorno()` para confirmar rutas

---

## 📥 Fase 1: Transformación de Datos

### Notebooks
- [ ] Mover `01_limpieza_datos.ipynb` a `notebooks/fase1_transformacion/`
- [ ] Mover `02_genera_reportes_sweetviz.ipynb` → renombrar a `02_genera_reportes.ipynb`
- [ ] Mover `03_union_dataset.ipynb`
- [ ] Mover `03b_documentacion_trazabilidad.ipynb` → renombrar a `04_documentacion.ipynb`

### Verificación
- [ ] Ejecutar `01_limpieza_datos.ipynb` → genera parquets en `data/02_interim/`
- [ ] Ejecutar `02_genera_reportes.ipynb` → genera HTMLs en `docs/html/reportes/`
- [ ] Ejecutar `03_union_dataset.ipynb` → genera `data/03_processed/df_alumno.parquet`
- [ ] Ejecutar `04_documentacion.ipynb` → genera `docs/html/fase1/*.html`

### Entregable
- [ ] Crear `docs/entregables/fase1/fase1_informe.docx`
- [ ] Exportar a PDF
- [ ] Crear mapa conceptual

---

## 📊 Fase 2: Análisis Exploratorio (EDA)

### Notebooks
- [ ] Mover `04_EDA_01_inspeccion.ipynb` → `notebooks/fase2_eda/01_inspeccion.ipynb`
- [ ] Mover `04_EDA_02_describe.ipynb` → `02_describe.ipynb`
- [ ] Mover `04_EDA_03_nulos.ipynb` → `03_nulos.ipynb`
- [ ] Mover `04_EDA_04_duplicados_outliers.ipynb` → `04_duplicados.ipynb`
- [ ] Mover `04_EDA_05_univariado_numericas.ipynb` → `05_univariado.ipynb`

### HTMLs
- [ ] Mover `docs/html/eda/01_inspeccion.html` → `docs/html/fase2/m01_inspeccion.html`
- [ ] Mover `02_describe.html` → `m02_describe.html`
- [ ] Mover `03_nulos.html` → `m03_nulos.html`

### Verificación
- [ ] Ejecutar todos los notebooks de Fase 2
- [ ] Verificar HTMLs generados

### Entregable
- [ ] Crear `docs/entregables/fase2/fase2_informe.docx`

---

## 🔧 Fase 3: Feature Engineering

- [ ] Pendiente

---

## 🤖 Fase 4: Modelado

- [ ] Pendiente

---

## 📈 Fase 5: Evaluación

- [ ] Pendiente

---

## 🚀 Fase 6: Aplicación

- [ ] Pendiente

---

## 📤 Entrega final

- [ ] Revisión completa de todos los HTMLs
- [ ] Memoria final en `docs/entregables/TFM_Morte_Maria_Jose.pdf`
- [ ] Presentación en `docs/entregables/presentacion.pptx`
- [ ] Subir a GitHub
- [ ] Probar en Colab
- [ ] Preparar demo para tribunal

---
