"""
Script Dedicado para Cadastro de Rostos via Webcam ou Sincronização de Imagens.
Permite registrar novas pessoas de forma rápida, salvando fotos e gerando embeddings FaceNet.
"""

import argparse
import sys
import time
from pathlib import Path
import cv2
import torch
from PIL import Image

from src import config
from src.camera import ThreadedCamera
from src.database import FaceDatabase
from src.detector import FaceDetector
from src.model import FaceNetRecognizer
from src.ui import draw_registration_overlay


def register_via_webcam(
    name: str,
    target_samples: int = 10,
    camera_index: int = config.CAMERA_INDEX,
):
    """Captura amostras do rosto da pessoa diretamente pela webcam e atualiza o banco."""
    name = name.strip()
    if not name:
        print("[Erro] O nome da pessoa não pode ser vazio.")
        return

    # Pasta para salvar as fotos da pessoa no dataset
    person_dir = config.DATASET_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Cadastro] Iniciando cadastro para: '{name}'")
    print(f"[Cadastro] Abrindo câmera {camera_index}...")

    detector = FaceDetector()
    recognizer = FaceNetRecognizer()
    database = FaceDatabase()

    try:
        cam = ThreadedCamera(
            camera_index=camera_index,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            fps=config.CAMERA_FPS,
        ).start()
    except Exception as e:
        print(f"[Erro] Falha ao iniciar câmera: {e}")
        return

    window_name = f"Cadastro FaceNet - {name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    captured_tensors = []
    captured_count = 0
    is_capturing = False
    last_capture_time = 0.0
    capture_interval = 0.25  # segundos entre cada captura para variar expressão/ângulo

    print("\n" + "=" * 50)
    print(f"  CADASTRO DE: {name}")
    print("  [ESPAÇO] : Iniciar a captura de fotos")
    print("  [Q] ou [ESC] : Cancelar e sair")
    print("=" * 50 + "\n")

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            display_frame = frame.copy()

            # Detecção de rosto
            boxes, face_tensors, probs = detector.detect_and_extract(frame)

            # Se encontrou ao menos um rosto
            has_single_face = len(boxes) == 1
            instruction = ""

            if not boxes:
                instruction = "Nenhum rosto detectado. Aproxime-se da camera."
            elif len(boxes) > 1:
                instruction = "Multiplos rostos na camera. Apenas uma pessoa deve aparecer."
            else:
                box = boxes[0]
                x1, y1, x2, y2 = box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), config.COLOR_PRIMARY, 2)

                if not is_capturing:
                    instruction = "Pressione [ESPACO] para iniciar a gravacao."
                else:
                    instruction = "Gire levemente o rosto para diferentes angulos..."

                    # Se estiver no modo captura e passou o intervalo
                    now = time.time()
                    if now - last_capture_time >= capture_interval:
                        if face_tensors is not None and face_tensors.shape[0] > 0:
                            # Pega o primeiro tensor de rosto
                            face_crop_tensor = face_tensors[0].unsqueeze(0)
                            captured_tensors.append(face_crop_tensor)

                            # Salva imagem física no dataset
                            sample_filename = person_dir / f"amostra_{captured_count + 1:02d}.jpg"
                            # Corta e salva a região do rosto com margem
                            pad = 20
                            cx1 = max(0, x1 - pad)
                            cy1 = max(0, y1 - pad)
                            cx2 = min(w, x2 + pad)
                            cy2 = min(h, y2 + pad)
                            crop_bgr = frame[cy1:cy2, cx1:cx2]
                            if crop_bgr.size > 0:
                                cv2.imwrite(str(sample_filename), crop_bgr)

                            captured_count += 1
                            last_capture_time = now

            # Renderiza interface de overlay
            display_frame = draw_registration_overlay(
                display_frame,
                person_name=name,
                captured_count=captured_count,
                target_count=target_samples,
                instruction=instruction,
            )

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):  # Q ou ESC
                print("[Cadastro] Operação cancelada pelo usuário.")
                break
            elif key == 32:  # Barra de espaço
                if not is_capturing and has_single_face:
                    is_capturing = True
                    last_capture_time = time.time()

            # Checar se finalizou as capturas
            if captured_count >= target_samples:
                print(f"[Cadastro] Amostras coletadas com sucesso ({captured_count}/{target_samples})!")
                # Processar embeddings
                all_tensors = torch.cat(captured_tensors, dim=0)
                embeddings = recognizer.get_embeddings(all_tensors)
                database.add_person_embeddings(name, embeddings)
                print(f"[Cadastro] Rosto de '{name}' cadastrado e pronto para reconhecimento!")
                time.sleep(1.0)
                break

    finally:
        cam.stop()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Cadastro de faces para FaceNet")
    parser.add_argument("--name", "-n", type=str, default="", help="Nome da pessoa a cadastrar")
    parser.add_argument("--samples", "-s", type=int, default=10, help="Quantidade de amostras a capturar (padrão: 10)")
    parser.add_argument("--cam", "-c", type=int, default=config.CAMERA_INDEX, help="Índice da câmera")
    parser.add_argument("--sync", action="store_true", help="Sincronizar fotos existentes na pasta dataset/")
    parser.add_argument("--list", "-l", action="store_true", help="Listar pessoas atualmente cadastradas")
    parser.add_argument("--remove", "-r", type=str, default="", help="Remover uma pessoa do banco de dados")

    args = parser.parse_args()

    database = FaceDatabase()

    if args.list:
        persons = database.list_persons()
        print("\n--- Pessoas Cadastradas no Banco ---")
        if not persons:
            print("Nenhuma pessoa cadastrada ainda.")
        else:
            for p, count in persons.items():
                print(f" • {p}: {count} amostras de embedding")
        print("------------------------------------\n")
        return

    if args.remove:
        removed = database.remove_person(args.remove)
        if not removed:
            print(f"Pessoa '{args.remove}' não foi encontrada no banco.")
        return

    if args.sync:
        detector = FaceDetector()
        recognizer = FaceNetRecognizer()
        database.sync_from_dataset_folder(detector, recognizer)
        return

    name = args.name
    if not name:
        try:
            name = input("Digite o nome da pessoa para cadastrar: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nOperação cancelada.")
            return

    if not name:
        print("Nome inválido.")
        return

    register_via_webcam(name=name, target_samples=args.samples, camera_index=args.cam)


if __name__ == "__main__":
    main()
