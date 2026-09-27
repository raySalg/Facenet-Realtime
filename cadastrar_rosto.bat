@echo off
title FaceNet - Cadastro de Rostos
echo ========================================================
echo   Cadastro de Faces para o FaceNet
echo ========================================================

REM Verifica se o ambiente virtual existe
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

python cadastrar_rosto.py %*

pause
