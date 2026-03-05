#!/bin/bash
# ============================================================
# FASE 3.5: PRE-MODELADO AutoML
# TFM: Prediccion de Abandono Universitario
# Maria Jose Morte - UOC/UJI
# ============================================================

set -e
ERRORES=0

echo "============================================================"
echo " FASE 3.5: PRE-MODELADO AutoML"
echo "============================================================"

run_notebook() {
    local env=$1
    local notebook=$2
    local step=$3
    local total=$4
    
    echo ""
    echo "[$step/$total] $notebook (entorno: $env)..."
    
    eval "$(conda shell.bash hook 2>/dev/null)"
    conda activate "$env"
    
    if jupyter nbconvert --to notebook --execute "$notebook" \
        --output "${notebook%.ipynb}_executed.ipynb" \
        --ExecutePreprocessor.timeout=1200 2>/dev/null; then
        echo "   ✓ OK"
    else
        echo "   ✗ ERROR"
        ERRORES=$((ERRORES + 1))
    fi
}

run_notebook "tfm_abandono" "fautoml_m01_baselines.ipynb" 1 6
run_notebook "env_lazypredict" "fautoml_m02_lazypredict.ipynb" 2 6
run_notebook "env_pycaret" "fautoml_m03_pycaret.ipynb" 3 6
run_notebook "env_h2o" "fautoml_m04_h2o.ipynb" 4 6
run_notebook "env_autogluon" "fautoml_m05_autogluon.ipynb" 5 6
run_notebook "tfm_abandono" "fautoml_m06_comparativa.ipynb" 6 6

echo ""
echo "============================================================"
if [ $ERRORES -eq 0 ]; then
    echo " ✅ COMPLETADO SIN ERRORES"
else
    echo " ⚠️ COMPLETADO CON $ERRORES ERRORES"
fi
echo ""
echo " Resultados: data/automl/"
echo " HTML: docs/html/fase_automl/"
echo "============================================================"
