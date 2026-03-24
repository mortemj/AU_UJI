@echo off
chcp 65001 >nul
echo ============================================================
echo  FASE 3.5: PRE-MODELADO AutoML
echo  TFM: Prediccion de Abandono Universitario
echo  Maria Jose Morte - UOC/UJI
echo ============================================================
echo.

set ERRORES=0

echo [1/6] Baselines (entorno: tfm_abandono)...
call conda activate tfm_abandono
jupyter nbconvert --to notebook --execute fautoml_m01_baselines.ipynb --output fautoml_m01_executed.ipynb --ExecutePreprocessor.timeout=600
if %errorlevel% neq 0 (
    echo    ERROR en Baselines
    set /a ERRORES+=1
) else (
    echo    OK
)

echo.
echo [2/6] LazyPredict (entorno: env_lazypredict)...
call conda activate env_lazypredict
jupyter nbconvert --to notebook --execute fautoml_m02_lazypredict.ipynb --output fautoml_m02_executed.ipynb --ExecutePreprocessor.timeout=600
if %errorlevel% neq 0 (
    echo    ERROR en LazyPredict
    set /a ERRORES+=1
) else (
    echo    OK
)

echo.
echo [3/6] PyCaret (entorno: env_pycaret)...
call conda activate env_pycaret
jupyter nbconvert --to notebook --execute fautoml_m03_pycaret.ipynb --output fautoml_m03_executed.ipynb --ExecutePreprocessor.timeout=1200
if %errorlevel% neq 0 (
    echo    ERROR en PyCaret
    set /a ERRORES+=1
) else (
    echo    OK
)

echo.
echo [4/6] H2O (entorno: env_h2o)...
call conda activate env_h2o
jupyter nbconvert --to notebook --execute fautoml_m04_h2o.ipynb --output fautoml_m04_executed.ipynb --ExecutePreprocessor.timeout=1200
if %errorlevel% neq 0 (
    echo    ERROR en H2O
    set /a ERRORES+=1
) else (
    echo    OK
)

echo.
echo [5/6] AutoGluon (entorno: env_autogluon)...
call conda activate env_autogluon
jupyter nbconvert --to notebook --execute fautoml_m05_autogluon.ipynb --output fautoml_m05_executed.ipynb --ExecutePreprocessor.timeout=1200
if %errorlevel% neq 0 (
    echo    ERROR en AutoGluon
    set /a ERRORES+=1
) else (
    echo    OK
)

echo.
echo [6/6] Comparativa final (entorno: tfm_abandono)...
call conda activate tfm_abandono
jupyter nbconvert --to notebook --execute fautoml_m06_comparativa.ipynb --output fautoml_m06_executed.ipynb --ExecutePreprocessor.timeout=300
if %errorlevel% neq 0 (
    echo    ERROR en Comparativa
    set /a ERRORES+=1
) else (
    echo    OK
)

echo.
echo ============================================================
if %ERRORES% equ 0 (
    echo  COMPLETADO SIN ERRORES
) else (
    echo  COMPLETADO CON %ERRORES% ERRORES
)
echo.
echo  Resultados en: data\automl\
echo  HTML en: docs\html\fase_automl\
echo ============================================================
pause
