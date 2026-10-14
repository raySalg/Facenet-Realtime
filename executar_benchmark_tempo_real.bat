@echo off
title FaceNet - Benchmark em Tempo Real (TCC)
echo ========================================================
echo   Iniciando Benchmark em Tempo Real FaceNet para o TCC
echo ========================================================

REM Verifica se o ambiente virtual existe
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

python benchmark_tempo_real.py %*

if %errorlevel% neq 0 (
    echo.
    echo Ocorreu um erro durante a execucao.
    pause
)
