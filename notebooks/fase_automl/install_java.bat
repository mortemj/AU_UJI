@echo off
chcp 65001 >nul
echo ============================================================
echo  INSTALACIÓN DE JAVA PARA H2O
echo  TFM: Prediccion de Abandono Universitario
echo ============================================================
echo.

REM Verificar si Java ya está instalado
java -version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Java ya está instalado:
    java -version 2>&1
    echo.
    echo No es necesario hacer nada.
    pause
    exit /b 0
)

echo ❌ Java no encontrado.
echo.
echo Opciones para instalar Java:
echo.
echo   1. AUTOMÁTICA (winget - recomendado):
echo      Abre PowerShell como Administrador y ejecuta:
echo      winget install EclipseAdoptium.Temurin.21.JDK
echo.
echo   2. MANUAL:
echo      Descarga desde: https://adoptium.net/
echo      Elige: Windows x64, JDK 21, .msi
echo      Instala con las opciones por defecto.
echo.
echo   3. CONDA (dentro del entorno env_h2o):
echo      conda activate env_h2o
echo      conda install -c conda-forge openjdk=21 -y
echo.
echo Después de instalar, cierra y abre una nueva terminal.
echo Luego ejecuta este script otra vez para verificar.
echo.
pause
