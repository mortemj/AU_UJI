# Entornos Conda — Fase AutoML
## TFM: Pronóstico del Éxito y del Abandono — UJI
### Autora: María José Morte Ruiz

---

## Resumen de entornos

El proyecto usa **5 entornos conda**. El entorno principal `tfm_abandono`
contiene todo el TFM excepto los 4 frameworks AutoML que tienen dependencias
incompatibles entre sí.

| Entorno | Uso | Notebooks |
|---|---|---|
| `tfm_abandono` | Entorno principal — Fases 1-7 completas | Todos excepto M02-M05 AutoML |
| `env_lazypredict` | LazyPredict screening | `fautoml_m02_lazypredict.ipynb` |
| `env_pycaret` | PyCaret AutoML | `fautoml_m03_pycaret.ipynb` |
| `env_h2o` | H2O AutoML (requiere Java 8+) | `fautoml_m04_h2o.ipynb` |
| `env_autogluon` | AutoGluon stacking | `fautoml_m05_autogluon.ipynb` |

---

## Entorno principal: tfm_abandono

Contiene todas las librerías del TFM. Ver `requirements.txt` para versiones exactas.

```bash
conda activate tfm_abandono
```

Notebooks que usa: Fases 1, 2, 3, 4, 5, 6, 7 completas +
AutoML M01 (baselines), M06 (TabPFN v2), M07 (comparativa).

---

## env_lazypredict

```bash
# Crear (si no existe)
conda create -n env_lazypredict python=3.11 -y
conda run -n env_lazypredict pip install lazypredict scikit-learn pandas pyarrow jupyter ipykernel matplotlib

# Registrar kernel en Jupyter
conda run -n env_lazypredict python -m ipykernel install --user --name env_lazypredict --display-name "Python (LazyPredict)"

# Activar
conda activate env_lazypredict
```

Librerías principales: `lazypredict`, `scikit-learn`, `pandas`, `pyarrow`

---

## env_pycaret

```bash
conda create -n env_pycaret python=3.11 -y
conda run -n env_pycaret pip install pycaret[full] jupyter ipykernel pandas pyarrow

conda run -n env_pycaret python -m ipykernel install --user --name env_pycaret --display-name "Python (PyCaret)"
```

Librerías principales: `pycaret>=3.0`, `scikit-learn`, `lightgbm`, `xgboost`

---

## env_h2o

```bash
conda create -n env_h2o python=3.11 -y
conda run -n env_h2o pip install h2o jupyter ipykernel pandas pyarrow requests

conda run -n env_h2o python -m ipykernel install --user --name env_h2o --display-name "Python (H2O)"
```

**Requisito:** Java 8+ instalado en el sistema.
Descarga: https://adoptium.net/

Librerías principales: `h2o>=3.44`

---

## env_autogluon

```bash
conda create -n env_autogluon python=3.11 -y
conda run -n env_autogluon pip install autogluon jupyter ipykernel pandas pyarrow

conda run -n env_autogluon python -m ipykernel install --user --name env_autogluon --display-name "Python (AutoGluon)"
```

**Nota:** AutoGluon puede dar errores con OneDrive en Windows.
Solución: usar ruta local fuera de OneDrive para los modelos temporales.

```python
# En fautoml_m05_autogluon.ipynb, celda 3:
ruta_modelos = Path('C:/tmp/autogluon_models')
```

Librerías principales: `autogluon>=1.0`, `lightgbm`, `catboost`, `xgboost`

---

## Verificar entornos

```bash
# Ver todos los entornos disponibles
conda env list

# Verificar librerías de cada entorno
conda run -n env_lazypredict pip show lazypredict
conda run -n env_pycaret pip show pycaret
conda run -n env_h2o pip show h2o
conda run -n env_autogluon pip show autogluon

# Verificar kernel de un notebook
python -c "
import json
nb = json.load(open('fautoml_m03_pycaret.ipynb', encoding='utf-8'))
print(nb['metadata']['kernelspec']['display_name'])
"
```
