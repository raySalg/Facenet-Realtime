"""
Módulo de Detecção e Alinhamento Facial com MTCNN.
Otimizado para alto desempenho em tempo real via escala adaptativa.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN

from src import config


class FaceDetector:
    """
    Detector facial baseado em MTCNN (Multi-task Cascaded Convolutional Networks).
    Detecta rostos, gera caixas delimitadoras e alinha os pontos faciais para o FaceNet.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or config.DEVICE
        self.scale = config.DETECTION_SCALE

        # Inicializa o MTCNN
        # keep_all=True permite detectar múltiplos rostos simultaneamente
        self.mtcnn = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=config.MIN_FACE_SIZE,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            select_largest=False,
            keep_all=True,
            device=self.device,
        )

    def detect_and_extract(
        self, bgr_frame: np.ndarray
    ) -> Tuple[List[np.ndarray], Optional[torch.Tensor], List[float]]:
        """
        Detecta rostos no frame OpenCV (BGR), ajusta coordenadas para o frame original
        e extrai tensores de rostos alinhados prontos para o FaceNet.

        Retorna:
            boxes: Lista de caixas delimitadoras [x1, y1, x2, y2]
            face_tensors: Tensor PyTorch (N, 3, 160, 160) ou None
            probs: Lista de probabilidades de confiança da detecção
        """
        if bgr_frame is None:
            return [], None, []

        orig_h, orig_w = bgr_frame.shape[:2]

        # Converte BGR para RGB para o MTCNN
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # Otimização: redimensiona para detecção rápida se escala < 1.0
        if 0.1 < self.scale < 1.0:
            detect_w = max(int(orig_w * self.scale), 80)
            detect_h = max(int(orig_h * self.scale), 60)
            detect_img = pil_img.resize((detect_w, detect_h), Image.Resampling.BILINEAR)
            scale_x = orig_w / detect_w
            scale_y = orig_h / detect_h
        else:
            detect_img = pil_img
            scale_x = scale_y = 1.0

        # Executa detecção
        raw_boxes, raw_probs = self.mtcnn.detect(detect_img)

        if raw_boxes is None or len(raw_boxes) == 0:
            return [], None, []

        # Re-escala as caixas para a resolução original do frame
        adjusted_boxes = []
        valid_boxes = []
        probs = []

        for box, prob in zip(raw_boxes, raw_probs):
            if prob is None or prob < 0.80:
                continue

            x1 = max(0, int(box[0] * scale_x))
            y1 = max(0, int(box[1] * scale_y))
            x2 = min(orig_w, int(box[2] * scale_x))
            y2 = min(orig_h, int(box[3] * scale_y))

            # Valida dimensões mínimas
            if (x2 - x1) < config.MIN_FACE_SIZE or (y2 - y1) < config.MIN_FACE_SIZE:
                continue

            adjusted_boxes.append(np.array([x1, y1, x2, y2]))
            # Para extração, passamos as caixas no espaço original da pil_img
            valid_boxes.append([x1, y1, x2, y2])
            probs.append(float(prob))

        if not valid_boxes:
            return [], None, []

        # Extrai os tensores dos rostos normalizados e alinhados em 160x160 a partir da imagem original
        try:
            face_tensors = self.mtcnn.extract(pil_img, valid_boxes, save_path=None)
            if face_tensors is not None and face_tensors.dim() == 3:
                # Se apenas um rosto foi extraído, adiciona dimensão de batch
                face_tensors = face_tensors.unsqueeze(0)
            return adjusted_boxes, face_tensors, probs
        except Exception:
            return adjusted_boxes, None, probs
