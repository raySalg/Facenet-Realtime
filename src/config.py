"""
Configurações Globais do Sistema de Reconhecimento Facial FaceNet.
Projetado para operar de forma transparente no Windows e no Linux.
"""

from pathlib import Path
import torch

# Diretórios base (usando pathlib para garantir compatibilidade Windows e Linux)
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
DATA_DIR = BASE_DIR / "data"
DATABASE_FILE = DATA_DIR / "database.pt"

# Garantir que as pastas existam
DATASET_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dispositivo de processamento (GPU CUDA se disponível, senão CPU)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configurações da Câmera
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# Otimizações de Desempenho em Tempo Real
# Escalar imagem para detecção (0.5 a 0.8 reduz uso de CPU sem perder precisão)
DETECTION_SCALE = 0.6

# Processar reconhecimento a cada N frames (interpola os outros frames para garantir 30+ FPS)
FRAME_SKIP = 2

# Tamanho mínimo do rosto em pixels para ser considerado
MIN_FACE_SIZE = 40

# Modelo FaceNet (InceptionResnetV1)
# Opções: 'vggface2' (recomendado) ou 'casia-webface'
FACENET_PRETRAINED = "vggface2"

# Métrica de Reconhecimento e Limiar (Threshold)
# Distância Euclidiana em vetores normalizados:
# < 0.70: Extremamente seguro / mesma pessoa
# 0.70 - 0.90: Mesma pessoa sob diferentes iluminações/ângulos
# > 0.90 - 1.00: Geralmente pessoas diferentes
DISTANCE_METRIC = "euclidean"  # 'euclidean' ou 'cosine'
RECOGNITION_THRESHOLD = 0.85   # Limiar de corte padrão
COSINE_THRESHOLD = 0.65        # Usado se DISTANCE_METRIC == 'cosine'

# Paleta de Cores da UI (Formato BGR para OpenCV)
COLOR_SUCCESS = (0, 220, 100)    # Verde moderno (Rosto identificado)
COLOR_UNKNOWN = (60, 60, 240)    # Vermelho/Laranja (Desconhecido)
COLOR_PRIMARY = (255, 170, 0)    # Azul elétrico (Destaques / UI)
COLOR_TEXT = (255, 255, 255)       # Branco
COLOR_BG_DARK = (20, 20, 20)      # Fundo escuro para badges
