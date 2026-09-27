"""
Módulo do Modelo FaceNet (InceptionResnetV1).
Responsável por extrair embeddings faciais de 512 dimensões e calcular similaridade.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1

from src import config


class FaceNetRecognizer:
    """
    Extrator de embeddings e classificador facial utilizando a arquitetura FaceNet.
    Gera vetores normalizados L2 de 512 dimensões para comparação métrica.
    """

    def __init__(self, device: Optional[torch.device] = None, pretrained: str = config.FACENET_PRETRAINED):
        self.device = device or config.DEVICE
        self.pretrained = pretrained

        print(f"[FaceNet] Carregando modelo pré-treinado '{self.pretrained}' no dispositivo: {self.device}...")
        self.model = InceptionResnetV1(pretrained=self.pretrained).eval().to(self.device)
        print("[FaceNet] Modelo carregado com sucesso.")

    @torch.no_grad()
    def get_embeddings(self, face_tensors: torch.Tensor) -> torch.Tensor:
        """
        Recebe um batch de tensores de rostos (N, 3, 160, 160)
        e retorna os embeddings normalizados de 512 dimensões (N, 512).
        """
        if face_tensors is None or face_tensors.nelement() == 0:
            return torch.empty((0, 512), device=self.device)

        if face_tensors.dim() == 3:
            face_tensors = face_tensors.unsqueeze(0)

        face_tensors = face_tensors.to(self.device)
        raw_embeddings = self.model(face_tensors)
        # Normalização L2 para garantir norma unitária ||x|| = 1
        normalized_embeddings = F.normalize(raw_embeddings, p=2, dim=1)
        return normalized_embeddings

    def match_face(
        self,
        query_embedding: torch.Tensor,
        database: Dict[str, torch.Tensor],
        threshold: float = config.RECOGNITION_THRESHOLD,
        metric: str = config.DISTANCE_METRIC,
    ) -> Tuple[str, float, float]:
        """
        Compara o embedding do rosto com os embeddings armazenados no banco de dados.

        Retorna:
            (nome_identificado, confianca_percentual, distancia_minima)
        """
        if not database:
            return "Desconhecido", 0.0, float("inf")

        query_emb = query_embedding.to(self.device)
        if query_emb.dim() == 1:
            query_emb = query_emb.unsqueeze(0)

        best_name = "Desconhecido"
        best_distance = float("inf")
        best_similarity = -1.0

        for name, db_embeddings in database.items():
            db_emb = db_embeddings.to(self.device)
            if db_emb.dim() == 1:
                db_emb = db_emb.unsqueeze(0)

            if metric == "cosine":
                # Similaridade de cosseno: produto escalar de vetores L2 normalizados
                similarities = torch.mm(query_emb, db_emb.t()).squeeze(0)
                max_sim = torch.max(similarities).item()

                if max_sim > best_similarity:
                    best_similarity = max_sim
                    best_name = name
            else:
                # Distância Euclidiana L2: ||a - b||_2
                # query_emb: (1, 512), db_emb: (K, 512)
                diff = db_emb - query_emb
                distances = torch.norm(diff, p=2, dim=1)
                min_dist = torch.min(distances).item()

                if min_dist < best_distance:
                    best_distance = min_dist
                    best_name = name

        if metric == "cosine":
            is_match = best_similarity >= config.COSINE_THRESHOLD
            score = best_similarity
            # Cálculo de porcentagem de confiança para cosseno
            confidence = max(0.0, min(100.0, score * 100.0))
            if not is_match:
                return "Desconhecido", confidence, (1.0 - score)
            return best_name, confidence, (1.0 - score)
        else:
            is_match = best_distance <= threshold
            # Distância Euclidiana em vetores normalizados varia de 0 a 2.
            # Se dist == 0 -> 100%
            # Se dist == threshold -> ~55%
            # Se dist > threshold -> cai abaixo de 50%
            if best_distance <= threshold:
                confidence = 50.0 + ((threshold - best_distance) / threshold) * 50.0
            else:
                confidence = max(0.0, 50.0 - ((best_distance - threshold) / threshold) * 50.0)

            confidence = max(0.0, min(100.0, confidence))

            if not is_match:
                return "Desconhecido", confidence, best_distance

            return best_name, confidence, best_distance
