#!/usr/bin/env bash
set -e

echo "========================================================"
echo "  Instalação de Dependências FaceNet (Linux)"
echo "========================================================"

# Diretório atual do script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Criação do virtualenv se não existir
if [ ! -d ".venv" ]; then
    echo "[1/3] Criando ambiente virtual .venv..."
    if command -v python3 -m venv > /dev/null 2>&1; then
        python3 -m venv .venv
    else
        python3 -m virtualenv .venv
    fi
else
    echo "[1/3] Ambiente virtual .venv já existe."
fi

# Ativação do ambiente
source .venv/bin/activate

echo "[2/3] Atualizando pip..."
pip install --upgrade pip

echo "[3/3] Instalando dependências..."
pip install -r requirements.txt

echo ""
echo "========================================================"
echo "  Instalação finalizada com sucesso!"
echo "  Para iniciar: ./iniciar_reconhecimento.sh"
echo "  Para cadastrar: ./cadastrar_rosto.sh"
echo "========================================================"
