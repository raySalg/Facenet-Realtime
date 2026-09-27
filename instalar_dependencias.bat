@echo off
title Instalação FaceNet - Windows
echo ========================================================
echo   Instalando Dependencias FaceNet para Windows
echo ========================================================

REM Verifica se o Python esta instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERRO] Python nao encontrado no PATH do Windows!
    echo Instale o Python 3.10+ pelo site oficial python.org
    echo e certifique-se de marcar a opcao "Add Python to PATH".
    pause
    exit /b 1
)

REM Criar ambiente virtual se nao existir
if not exist ".venv" (
    echo [1/3] Criando ambiente virtual .venv...
    python -m venv .venv
) else (
    echo [1/3] Ambiente virtual .venv ja existe.
)

REM Ativar o ambiente virtual
call .venv\Scripts\activate.bat

echo [2/3] Atualizando pip...
python -m pip install --upgrade pip

echo [3/3] Instalando dependencias do requirements.txt...
pip install -r requirements.txt

echo.
echo ========================================================
echo   Instalacao concluida com sucesso!
echo   Para iniciar: execute iniciar_reconhecimento.bat
echo   Para cadastrar: execute cadastrar_rosto.bat
echo ========================================================
pause
