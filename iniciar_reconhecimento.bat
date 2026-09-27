@echo off
title FaceNet - Reconhecimento Facial em Tempo Real
echo ========================================================
echo   Iniciando Reconhecimento Facial FaceNet
echo ========================================================

REM Verifica se o ambiente virtual existe
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [AVISO] Ambiente .venv nao encontrado. Executando com Python global...
)

REM Executa a aplicacao principal
python main.py %*

if %errorlevel% neq 0 (
    echo.
    echo Ocorreu um erro durante a execucao.
    pause
)
