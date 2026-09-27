"""
Gerenciador do Banco de Dados de Embeddings e Sincronização com o Dataset.
Compatível com sistemas de arquivos Windows e Linux via pathlib.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import torch

from src import config
from src.detector import FaceDetector
from src.model import FaceNetRecognizer


class FaceDatabase:
    """
    Armazena e gerencia os vetores de características (embeddings) dos rostos conhecidos.
    Permite sincronização automática a partir de pastas de fotos ou registros pela webcam.
    """

    def __init__(self, db_path: Optional[Path] = None, dataset_dir: Optional[Path] = None):
        self.db_path = db_path or config.DATABASE_FILE
        self.dataset_dir = dataset_dir or config.DATASET_DIR
        self.database: Dict[str, torch.Tensor] = {}
        self.load()

    def load(self) -> Dict[str, torch.Tensor]:
        """Carrega os embeddings salvos no disco."""
        if self.db_path.exists():
            try:
                # torch.load seguro com suporte a CPU e GPU
                data = torch.load(self.db_path, map_location="cpu", weights_only=False)
                if isinstance(data, dict):
                    self.database = data
                    print(f"[Database] Carregadas {len(self.database)} pessoas registradas: {list(self.database.keys())}")
                else:
                    self.database = {}
            except Exception as e:
                print(f"[Database] Aviso ao carregar {self.db_path}: {e}. Iniciando banco vazio.")
                self.database = {}
        else:
            self.database = {}
        return self.database

    def save(self):
        """Salva o dicionário de embeddings no disco."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Move tensores para CPU antes de salvar para garantir compatibilidade
        cpu_database = {k: v.cpu() for k, v in self.database.items()}
        torch.save(cpu_database, self.db_path)
        print(f"[Database] Banco de dados salvo em: {self.db_path}")

    def add_person_embeddings(self, name: str, new_embeddings: torch.Tensor):
        """Adiciona ou concatena novos embeddings para uma pessoa."""
        new_embeddings = new_embeddings.cpu()
        if new_embeddings.dim() == 1:
            new_embeddings = new_embeddings.unsqueeze(0)

        if name in self.database:
            existing = self.database[name].cpu()
            combined = torch.cat([existing, new_embeddings], dim=0)
            self.database[name] = combined
        else:
            self.database[name] = new_embeddings

        self.save()

    def remove_person(self, name: str) -> bool:
        """Remove uma pessoa do banco de dados de embeddings."""
        if name in self.database:
            del self.database[name]
            self.save()
            print(f"[Database] '{name}' foi removido do banco de dados.")
            return True
        return False

    def list_persons(self) -> Dict[str, int]:
        """Retorna os nomes cadastrados e a quantidade de amostras faciais de cada um."""
        return {name: emb.shape[0] for name, emb in self.database.items()}

    def sync_from_dataset_folder(
        self,
        detector: FaceDetector,
        recognizer: FaceNetRecognizer,
        save_crops: bool = True,
    ) -> int:
        """
        Varre a pasta dataset/ onde cada subpasta representa o nome de uma pessoa.
        Extrai os rostos com MTCNN e calcula os embeddings FaceNet.
        Exemplo de estrutura:
          dataset/
            ├── Maria/
            │   ├── foto1.jpg
            │   └── foto2.png
            └── João/
                └── foto1.jpg
        """
        print(f"\n[Database] Sincronizando imagens a partir de: {self.dataset_dir} ...")
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        new_db: Dict[str, List[torch.Tensor]] = {}
        total_faces_processed = 0

        if not self.dataset_dir.exists():
            print("[Database] Pasta dataset não encontrada.")
            return 0

        # Iterar sobre subpastas de pessoas
        for person_dir in self.dataset_dir.iterdir():
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name
            embeddings_list = []

            for img_path in person_dir.iterdir():
                if img_path.suffix.lower() not in valid_extensions:
                    continue

                try:
                    # Carregar imagem com PIL para garantir compatibilidade de canais e orientação EXIF
                    pil_img = Image.open(img_path).convert("RGB")
                    # Detectar e extrair rosto alinhado
                    # MTCNN pode extrair diretamente
                    face_tensor = detector.mtcnn(pil_img)
                    if face_tensor is not None:
                        if face_tensor.dim() == 4:
                            face_tensor = face_tensor[0]  # Pega o rosto principal
                        face_tensor = face_tensor.unsqueeze(0)
                        emb = recognizer.get_embeddings(face_tensor)
                        embeddings_list.append(emb.cpu())
                        total_faces_processed += 1
                    else:
                        print(f"[Database] Nenhum rosto detectado em: {img_path.name} ({person_name})")
                except Exception as e:
                    print(f"[Database] Erro ao processar {img_path}: {e}")

            if embeddings_list:
                new_db[person_name] = torch.cat(embeddings_list, dim=0)
                print(f" -> '{person_name}': {len(embeddings_list)} amostra(s) processada(s).")

        if new_db:
            # Atualiza o banco atual
            self.database.update(new_db)
            self.save()
            print(f"[Database] Sincronização concluída! {total_faces_processed} fotos processadas.")
        else:
            print("[Database] Nenhuma imagem válida encontrada nas subpastas de dataset/.")

        return total_faces_processed
