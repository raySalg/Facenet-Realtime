"""
Aplicação Principal de Reconhecimento Facial em Tempo Real com FaceNet.
Totalmente compatível com Windows e Linux.
"""

import argparse
import os
import sys
import time
from pathlib import Path
import cv2
import torch

from src import config
from src.camera import ThreadedCamera
from src.database import FaceDatabase
from src.detector import FaceDetector
from src.model import FaceNetRecognizer
from src.ui import draw_hud
import cadastrar_rosto


def run_recognition_app(
    camera_index: int = config.CAMERA_INDEX,
    threshold: float = config.RECOGNITION_THRESHOLD,
    frame_skip: int = config.FRAME_SKIP,
    metric: str = config.DISTANCE_METRIC,
):
    print("=" * 60)
    print("   SISTEMA DE RECONHECIMENTO FACIAL FACENET (TEMPO REAL)")
    print("=" * 60)
    print(f" • Plataforma: {sys.platform}")
    print(f" • Dispositivo de Computação: {config.DEVICE}")
    print(f" • Limiar de Reconhecimento (Threshold): {threshold}")
    print(f" • Métrica: {metric}")
    print(f" • Índice da Câmera: {camera_index}")
    print("=" * 60 + "\n")

    # Inicializa o banco de dados
    database = FaceDatabase()
    registered_persons = database.list_persons()
    if not registered_persons:
        print("[Aviso] Nenhum rosto cadastrado no momento.")
        print(" -> Você pode cadastrar pressionando a tecla [C] na janela do vídeo")
        print(" -> Ou executar: python cadastrar_rosto.py --name \"Seu Nome\"\n")
    else:
        print(f"[Sistema] {len(registered_persons)} pessoas cadastradas no banco: {list(registered_persons.keys())}\n")

    # Inicializa detector e modelo
    detector = FaceDetector()
    recognizer = FaceNetRecognizer()

    # Inicializa a câmera em thread assíncrona para eliminar atrasos
    print("[Sistema] Conectando à câmera...")
    try:
        camera = ThreadedCamera(
            camera_index=camera_index,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            fps=config.CAMERA_FPS,
        ).start()
    except Exception as e:
        print(f"\n[Erro Fatal] Falha ao inicializar a câmera: {e}")
        print("\nDicas para solução:")
        print(" 1. Verifique se outra aplicação não está usando a webcam (Zoom, Teams, Navegador, etc.).")
        print(" 2. No Linux: certifique-se de ter permissões no /dev/video*.")
        print(" 3. No Windows: verifique em 'Configurações de Privacidade do Windows' se a Câmera está autorizada.")
        print(" 4. Tente outro índice com a opção: python main.py --cam 1")
        return

    window_name = "FaceNet Realtime Face Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

    # Variáveis de controle de FPS e fluxo
    fps = 0.0
    frame_count = 0
    fps_start_time = time.time()
    cached_detections = []
    is_paused = False

    device_str = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"

    print("\n[Sistema] Sistema iniciado com sucesso!")
    print("Controles na janela de vídeo:")
    print("  [Q] ou [ESC] : Sair da aplicação")
    print("  [C]          : Cadastrar um novo rosto agora")
    print("  [R]          : Recarregar banco de dados de rostos")
    print("  [S]          : Tirar foto e salvar em data/")
    print("  [Espaço]     : Pausar / Continuar exibição\n")

    try:
        first_frame_shown = False
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            # Cálculo de FPS
            frame_count += 1
            now = time.time()
            elapsed = now - fps_start_time
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = now

            if not is_paused:
                # Otimização de processamento com Frame Skip
                should_detect = (frame_count % frame_skip == 0) or (len(cached_detections) == 0)

                if should_detect:
                    boxes, face_tensors, probs = detector.detect_and_extract(frame)

                    new_detections = []
                    if face_tensors is not None and face_tensors.shape[0] > 0:
                        # Extrai embeddings para todas as faces detectadas no frame em uma única inferência
                        embeddings = recognizer.get_embeddings(face_tensors)

                        for i, box in enumerate(boxes):
                            emb = embeddings[i]
                            name, confidence, distance = recognizer.match_face(
                                query_embedding=emb,
                                database=database.database,
                                threshold=threshold,
                                metric=metric,
                            )
                            new_detections.append({
                                "box": box,
                                "name": name,
                                "confidence": confidence,
                                "distance": distance,
                            })

                    cached_detections = new_detections

            # Renderiza a HUD moderna
            display_frame = draw_hud(
                frame=frame,
                detections=cached_detections,
                fps=fps,
                device_name=device_str,
                total_registered=len(database.database),
                paused=is_paused,
            )

            cv2.imshow(window_name, display_frame)

            # Leitura de teclas de atalho
            key = cv2.waitKey(1) & 0xFF

            # Checar se a janela foi fechada pelo botão 'X' (Qt / Win32)
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                pass

            # [Q] ou [ESC] - Sair
            if key in (ord("q"), ord("Q"), 27):
                print("[Sistema] Encerrando aplicação...")
                break

            # [Espaço] - Pausar
            elif key == 32:
                is_paused = not is_paused

            # [R] - Recarregar banco
            elif key in (ord("r"), ord("R")):
                print("[Sistema] Recarregando banco de dados...")
                database.load()

            # [S] - Capturar screenshot
            elif key in (ord("s"), ord("S")):
                shot_path = config.DATA_DIR / f"captura_{int(time.time())}.jpg"
                cv2.imwrite(str(shot_path), display_frame)
                print(f"[Sistema] Captura salva em: {shot_path}")

            # [C] - Cadastrar novo rosto
            elif key in (ord("c"), ord("C")):
                print("\n[Cadastro] Pausando reconhecimento para cadastro de novo usuário...")
                cv2.destroyWindow(window_name)
                camera.stop()

                try:
                    novo_nome = input("\n>> Digite o nome da nova pessoa para cadastrar: ").strip()
                except (KeyboardInterrupt, EOFError):
                    novo_nome = ""

                if novo_nome:
                    cadastrar_rosto.register_via_webcam(
                        name=novo_nome,
                        target_samples=10,
                        camera_index=camera_index,
                    )
                    database.load()

                # Reinicia a câmera e a janela principal
                camera = ThreadedCamera(
                    camera_index=camera_index,
                    width=config.CAMERA_WIDTH,
                    height=config.CAMERA_HEIGHT,
                    fps=config.CAMERA_FPS,
                ).start()
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

    except KeyboardInterrupt:
        print("\n[Sistema] Interrupção do teclado detectada.")
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("[Sistema] Câmera e janelas liberadas.")


def main():
    parser = argparse.ArgumentParser(description="Reconhecimento Facial em Tempo Real com FaceNet")
    parser.add_argument("--cam", "-c", type=int, default=config.CAMERA_INDEX, help="Índice da câmera (padrão: 0)")
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=config.RECOGNITION_THRESHOLD,
        help="Limiar de distância Euclidiana (padrão: 0.85). Valores menores = mais rigoroso.",
    )
    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        default=config.DISTANCE_METRIC,
        choices=["euclidean", "cosine"],
        help="Métrica de comparação: euclidean ou cosine (padrão: euclidean)",
    )
    parser.add_argument(
        "--skip",
        "-s",
        type=int,
        default=config.FRAME_SKIP,
        help="Taxa de frame skip para MTCNN/FaceNet (padrão: 2).",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Sincroniza imagens da pasta dataset/ antes de iniciar o reconhecimento.",
    )

    args = parser.parse_args()

    if args.sync:
        print("[Sistema] Sincronizando imagens de dataset/ antes de iniciar...")
        db = FaceDatabase()
        det = FaceDetector()
        rec = FaceNetRecognizer()
        db.sync_from_dataset_folder(det, rec)

    run_recognition_app(
        camera_index=args.cam,
        threshold=args.threshold,
        frame_skip=args.skip,
        metric=args.metric,
    )


if __name__ == "__main__":
    main()
