# 👁️ Reconhecimento Facial em Tempo Real com FaceNet (Cross-Platform)

Aplicação profissional em **Python** para reconhecimento facial em tempo real utilizando o modelo **FaceNet** (`InceptionResnetV1` pré-treinado no VGGFace2) e **MTCNN** para detecção e alinhamento facial.

Projetada com arquitetura modular para funcionar **nativamente no Windows e no Linux**, com detecção assíncrona da câmera (eliminando latências de buffer), suporte automático a **GPU (NVIDIA CUDA)** ou **CPU**, e interface gráfica moderna com exibição de métricas e taxa de quadros (FPS).

---

## 🚀 Principais Recursos

- **Modelo FaceNet de Alta Precisão**: Utiliza embeddings vetoriais de 512 dimensões com alinhamento facial preciso via MTCNN.
- **100% Multiplataforma (Windows & Linux)**:
  - Gerenciamento inteligente de backend de captura (`DirectShow` no Windows para inicialização instantânea e sem travamentos; `V4L2` no Linux).
  - Caminhos de arquivo agnósticos ao sistema operacional (`pathlib`).
  - Scripts executáveis em um clique (`.bat` para Windows e `.sh` para Linux).
- **Processamento em Tempo Real Fluido (30+ FPS)**:
  - Captura de câmera em *thread* assíncrona dedicada (evita atrasos acumulados no buffer do OpenCV).
  - *Frame skipping* adaptativo e escala de detecção inteligente para operação suave em processadores modernos.
- **Cadastro Dinâmico de Rostos**:
  - **Via Webcam**: Pressione `[C]` durante o vídeo ou execute o utilitário dedicado para capturar múltiplas amostras com instruções em tela.
  - **Via Fotos (Dataset)**: Basta colocar fotos organizadas por pasta (`dataset/Nome_Pessoa/foto.jpg`) e sincronizar.
- **HUD Visual Moderno**:
  - Caixas delimitadoras com cantos angulares estilizados.
  - Badges coloridos dinâmicos: **Verde** para rostos reconhecidos (com porcentagem de confiança) e **Vermelho** para desconhecidos.
  - Barra de status superior com FPS, processador ativo (CPU/CUDA) e atalhos rápidos.

---

## 📁 Estrutura do Projeto

```text
Facenet/
│
├── dataset/                      # Imagens de referência organizadas por pessoa
│   ├── Maria/
│   │   ├── foto1.jpg
│   │   └── foto2.jpg
│   └── Joao/
│       └── foto1.jpg
│
├── data/                         # Embeddings compilados e capturas de tela
│   └── database.pt               # Banco vetorial de embeddings
│
├── src/                          # Módulos do sistema
│   ├── config.py                 # Parâmetros globais (limiares, resolução, cores)
│   ├── camera.py                 # Captura assíncrona em thread (Linux & Windows)
│   ├── detector.py               # Detecção e alinhamento facial com MTCNN
│   ├── model.py                  # Extrator FaceNet (InceptionResnetV1) e métricas
│   ├── database.py               # Gerenciamento de embeddings e sincronização
│   └── ui.py                     # Renderização da interface visual (HUD)
│
├── main.py                       # Aplicação principal de reconhecimento ao vivo
├── cadastrar_rosto.py            # Utilitário para cadastrar ou listar pessoas
├── requirements.txt              # Dependências Python universais
│
├── iniciar_reconhecimento.bat    # Iniciar no Windows (1 clique)
├── cadastrar_rosto.bat           # Cadastrar no Windows (1 clique)
├── instalar_dependencias.bat     # Instalar dependências no Windows (1 clique)
│
├── iniciar_reconhecimento.sh     # Iniciar no Linux
├── cadastrar_rosto.sh            # Cadastrar no Linux
└── instalar_dependencias.sh      # Instalar dependências no Linux
```

---

## 🛠️ Instalação e Configuração

### Opção 1: No Windows

1. Certifique-se de ter o **Python 3.10+** instalado (marcando a opção *"Add Python to PATH"* durante a instalação).
2. Dê um duplo-clique no arquivo:
   ```cmd
   instalar_dependencias.bat
   ```
   *Ou execute manualmente no Prompt de Comando (CMD/PowerShell):*
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

> [!TIP]
> **Aceleração por GPU no Windows (NVIDIA):**
> Se você possui uma placa de vídeo NVIDIA, instale o PyTorch com suporte a CUDA antes de instalar as dependências:
> ```cmd
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> pip install -r requirements.txt
> ```

---

### Opção 2: No Linux (Ubuntu, Debian, Fedora, Arch, etc.)

1. Abra o terminal na pasta do projeto e execute:
   ```bash
   ./instalar_dependencias.sh
   ```
   *Ou execute manualmente:*
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🧑‍💻 Como Usar

### 1. Cadastrar um Rosto

Você pode cadastrar novos usuários de duas maneiras:

#### A) Pela Webcam (Recomendado):
Execute o script de cadastro e informe o nome:
- **No Windows**: Dê duplo-clique em `cadastrar_rosto.bat` ou:
  ```cmd
  python cadastrar_rosto.py --name "Seu Nome"
  ```
- **No Linux**:
  ```bash
  ./cadastrar_rosto.sh --name "Seu Nome"
  ```
*Uma janela da câmera se abrirá. Pressione a barra de **[ESPAÇO]** para iniciar a captura de 10 amostras enquanto movimenta levemente a cabeça.*

#### B) Colocando Fotos Existentes:
Crie uma pasta com o nome da pessoa dentro do diretório `dataset/` e coloque fotos nítidas do rosto:
```text
dataset/
└── Carlos/
    ├── foto1.jpg
    └── foto2.png
```
Depois sincronize executando:
```bash
python cadastrar_rosto.py --sync
```

#### C) Listar ou Remover Pessoas:
- Listar cadastrados:
  ```bash
  python cadastrar_rosto.py --list
  ```
- Remover uma pessoa:
  ```bash
  python cadastrar_rosto.py --remove "Nome Da Pessoa"
  ```

---

### 2. Iniciar o Reconhecimento Facial em Tempo Real

Para iniciar a transmissão ao vivo pela câmera:

- **No Windows**: Dê duplo-clique em `iniciar_reconhecimento.bat` ou:
  ```cmd
  python main.py
  ```
- **No Linux**:
  ```bash
  ./iniciar_reconhecimento.sh
  ```

---

## ⌨️ Controles e Atalhos (Durante o Vídeo)

Com a janela de vídeo aberta, você pode interagir utilizando o teclado:

| Tecla | Função |
| :---: | :--- |
| **`Q`** ou **`ESC`** | **Encerra a aplicação e salva automaticamente todos os relatórios e gráficos do TCC**. |
| **`T`** | Ativa ou altera o **Modo de Teste TCC** (define o sujeito Ground Truth ativo para medir Acurácia e F1 ao vivo). |
| **`M`** | **Salva imediatamente os relatórios** e gráficos do TCC em `data/` sem interromper a execução. |
| **`H`** | Alterna a exibição detalhada de **telemetria no topo** (Latência MTCNN/FaceNet, CPU %, RAM MB). |
| **`C`** | Pausa o vídeo para **cadastrar uma nova pessoa** imediatamente. |
| **`R`** | **Recarrega o banco de dados** de rostos (se tiver adicionado fotos na pasta `dataset/`). |
| **`S`** | Tira uma **captura de tela** (screenshot) com as marcações e salva na pasta `data/`. |
| **`Espaço`** | **Pausa / Retoma** o fluxo de vídeo. |

---

## 📊 Geração de Métricas e Triangulação dos Três Ambientes (TCC)

A aplicação integra um motor de telemetria científica que contrasta os resultados com o benchmark anterior:

### 1. Executar com Avaliação em Tempo Real Direta:
```bash
# Executa identificando e calculando Acurácia e F1 para a pessoa na câmera
python main.py --test "Rayssa"
```

### 2. Benchmark Automatizado Cronometrado:
```bash
# Executa coleta rigorosa por 15 segundos e gera tabelas para o Overleaf/artigo
python benchmark_tempo_real.py --duration 15 --ground-truth "Rayssa"
```

### 3. Artefatos Gerados Automaticamente em `data/`:
* **`comparativo_tres_ambientes.md`**: Tabela comparativa consolidando **Ambiente Controlado**, **Ambiente Não-Controlado (LFW)** e **Tempo Real**.
* **`comparativo_tres_ambientes.csv`**: Dados tabulados em CSV.
* **`tabela_tres_ambientes.tex`**: Tabela formatada em LaTeX pronta para inclusão no artigo.
* **`metricas_tempo_real.md`** & **`tabela_tempo_real.tex`**: Relatório aprofundado com decomposição de latência (MTCNN vs FaceNet).
* **`telemetria_tempo_real.csv`**: Série temporal segundo a segundo de FPS, latências, CPU e RAM.
* **`graficos/grafico_triangulacao_ambientes.png`**: Gráfico comparativo dos 3 ambientes em 300 DPI.
* **`graficos/grafico_telemetria_tempo_real.png`**: Painel com evolução de FPS, latência, CPU e RAM ao vivo.

---

## ⚙️ Argumentos Avançados de Linha de Comando

Você pode personalizar o comportamento do reconhecimento passando argumentos para o `main.py`:

```bash
python main.py [opções]
```

- `--test <nome>` ou `--ground-truth <nome>`: Ativa a avaliação em tempo real calculando Acurácia e F1 contra o sujeito na câmera.
- `--cam <indice>`: Especifica o índice da câmera (padrão: `0`). Útil caso use webcam USB externa (ex: `--cam 1`).
- `--threshold <valor>`: Limiar de distância euclidiana para identificação (padrão: `0.85`).
- `--metric <euclidean|cosine>`: Métrica de distância utilizada (`euclidean` por padrão ou `cosine`).
- `--skip <n>`: Intervalo de frames para inferência de embeddings (padrão: `2`).
- `--sync`: Sincroniza fotos da pasta `dataset/` antes de ligar a câmera.
- `--benchmark-csv <caminho>`: Caminho alternativo para o CSV do benchmark prévio.
- `--no-save`: Desativa o salvamento automático de relatórios ao sair.

**Exemplo Completo:**
```bash
python main.py --cam 0 --threshold 0.85 --test "Rayssa"
```

---

## ❓ Resolução de Problemas Comuns

### 1. A câmera não abre ou demora muito para abrir
- **No Windows**: O código já prioriza `cv2.CAP_DSHOW` (DirectShow). Verifique em *Iniciar > Configurações > Privacidade e Segurança > Câmera* se o acesso à câmera por aplicativos está habilitado.
- **No Linux**: Verifique se o seu usuário tem permissão para acessar os dispositivos de vídeo com `sudo usermod -aG video $USER`.

### 2. A taxa de quadros (FPS) está baixa
- No arquivo [src/config.py](file:///home/rayssa/Documentos/Teste/Facenet/src/config.py), você pode ajustar `DETECTION_SCALE = 0.5` ou executar com `--skip 3`.
- Se possuir uma GPU NVIDIA, certifique-se de instalar o PyTorch com suporte a CUDA. O programa detectará e usará a GPU automaticamente.
