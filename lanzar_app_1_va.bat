@echo off
cd /d "C:\Users\mjmor\OneDrive - Universitat Jaume I\2.- AU_UJI"

:: Intentar rutas comunes de conda
if exist "C:\Users\mjmor\anaconda3\Scripts\activate.bat" (
    call "C:\Users\mjmor\anaconda3\Scripts\activate.bat" tfm_abandono
    goto run
)
if exist "C:\Users\mjmor\miniconda3\Scripts\activate.bat" (
    call "C:\Users\mjmor\miniconda3\Scripts\activate.bat" tfm_abandono
    goto run
)
if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    call "C:\ProgramData\anaconda3\Scripts\activate.bat" tfm_abandono
    goto run
)
if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" (
    call "C:\ProgramData\miniconda3\Scripts\activate.bat" tfm_abandono
    goto run
)

echo ERROR: No se encontro conda. Ejecuta "where conda" en Anaconda Prompt y dime la ruta.
pause
exit /b 1

:run
streamlit run app/main.py
pause
