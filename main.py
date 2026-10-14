"""
Aplicação Principal de Reconhecimento Facial em Tempo Real com FaceNet.
Totalmente compatível com Windows e Linux.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional
import cv2
import torch

from src import config
from src.camera import ThreadedCamera
from src.database import FaceDatabase
from src.detector import FaceDetector
from src.model import FaceNetRecognizer
from src.realtime_metrics import RealtimeBenchmarkTracker
from src.ui import draw_hud
import cadastrar_rosto


def run_recognition_app(
    camera_index: int = config.CAMERA_INDEX,
    threshold: float = config.RECOGNITION_THRESHOLD,
    frame_skip: int = config.FRAME_SKIP,
    metric: str = config.DISTANCE_METRIC,
    ground_truth: Optional[str] = None,
    benchmark_csv: Optional[Path] = None,
    auto_save_metrics: bool = True,
):
    print("=" * 60)
    print("   SISTEMA DE RECONHECIMENTO FACIAL FACENET (TEMPO REAL)")
    print("=" * 60)
    print(f" • Plataforma: {sys.platform}")
    print(f" • Dispositivo de Computação: {config.DEVICE}")
    print(f" • Limiar de Reconhecimento (Threshold): {threshold}")
    print(f" • Métrica: {metric}")
    print(f" • Índice da Câmera: {camera_index}")
    if ground_truth:
        print(f" • Modo de Teste TCC: ATIVO (Sujeito Ground Truth = '{ground_truth}')")
    print("=" * 60 + "\n")

    # Localizar CSV do benchmark do TCC caso não tenha sido passado
    if benchmark_csv is None:
        default_bench_csv = config.BASE_DIR.parent / "Benchmark-de-Modelos-de-Reconhecimento-Facial" / "results" / "metrics_comparison.csv"
        if default_bench_csv.exists():
            benchmark_csv = default_bench_csv

    # Inicializa rastreador de telemetria científica em tempo real
    tracker = RealtimeBenchmarkTracker(ground_truth_subject=ground_truth)

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
    show_telemetry = True

    device_str = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"

    print("\n[Sistema] Sistema iniciado com sucesso!")
    print("Controles na janela de vídeo:")
    print("  [Q] ou [ESC] : Sair da aplicação e salvar relatórios do TCC")
    print("  [T]          : Iniciar/Alterar Modo de Teste TCC (Avaliação de Acurácia)")
    print("  [M]          : Salvar Relatórios e Gráficos do TCC agora")
    print("  [H]          : Alternar exibição de telemetria no topo")
    print("  [C]          : Cadastrar um novo rosto agora")
    print("  [R]          : Recarregar banco de dados de rostos")
    print("  [S]          : Tirar foto e salvar em data/")
    print("  [Espaço]     : Pausar / Continuar exibição\n")

    try:
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
                    # Medição rigorosa de Latência da Detecção (MTCNN)
                    t0_det = time.perf_counter()
                    boxes, face_tensors, probs = detector.detect_and_extract(frame)
                    t_detect_ms = (time.perf_counter() - t0_det) * 1000.0

                    new_detections = []
                    t_extract_ms = 0.0
                    t_match_ms = 0.0

                    if face_tensors is not None and face_tensors.shape[0] > 0:
                        # Medição rigorosa da Extração de Embeddings (FaceNet InceptionResnetV1)
                        t0_ext = time.perf_counter()
                        embeddings = recognizer.get_embeddings(face_tensors)
                        t_extract_ms = (time.perf_counter() - t0_ext) * 1000.0

                        # Medição da Busca Métrica Vetorial
                        t0_match = time.perf_counter()
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
                        t_match_ms = (time.perf_counter() - t0_match) * 1000.0

                    cached_detections = new_detections

                    # Registro de telemetria científica no tracker
                    tracker.record_frame(
                        t_detect_ms=t_detect_ms,
                        t_extract_ms=t_extract_ms,
                        t_match_ms=t_match_ms,
                        detections=cached_detections,
                        instant_fps=fps,
                    )

            # Obter snapshot da telemetria em tempo real
            live_telemetry = tracker.get_live_metrics(fps)

            # Renderiza a HUD moderna com telemetria
            display_frame = draw_hud(
                frame=frame,
                detections=cached_detections,
                fps=fps,
                device_name=device_str,
                total_registered=len(database.database),
                paused=is_paused,
                telemetry=live_telemetry,
                show_telemetry=show_telemetry,
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
                print("[Sistema] Encerrando aplicação e compilando relatórios...")
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

            # [H] - Alternar telemetria HUD
            elif key in (ord("h"), ord("H")):
                show_telemetry = not show_telemetry

            # [M] - Salvar relatórios imediatamente
            elif key in (ord("m"), ord("M")):
                print("\n[Métricas TCC] Exportando relatórios e gráficos parciais...")
                tracker.export_all_tcc_reports(output_dir=config.DATA_DIR, controlled_benchmark_csv=benchmark_csv)

            # [T] - Alternar / Definir Modo de Teste TCC (Ground Truth)
            elif key in (ord("t"), ord("T")):
                if tracker.test_mode_enabled:
                    tracker.set_ground_truth(None)
                else:
                    registered_keys = list(database.database.keys())
                    print("\n" + "=" * 55)
                    print("  CONFIGURAR MODO DE TESTE TCC (GROUND TRUTH)")
                    print("=" * 55)
                    print(f"Pessoas cadastradas: {registered_keys}")
                    print("Digite o nome da pessoa que está diante da câmera.")
                    print("(Ou pressione ENTER para testar 'Desconhecido')")
                    try:
                        target_name = input(">> Nome do Sujeito Ativo: ").strip()
                    except (KeyboardInterrupt, EOFError):
                        target_name = ""

                    if not target_name:
                        target_name = "Desconhecido"
                    tracker.set_ground_truth(target_name)
                print("=" * 55 + "\n")

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

        if auto_save_metrics:
            print("\n" + "=" * 70)
            print("  CONSOLIDANDO MÉTRICAS EXPERIMENTAIS DO TCC")
            print("=" * 70)
            tracker.export_all_tcc_reports(
                output_dir=config.DATA_DIR,
                controlled_benchmark_csv=benchmark_csv,
            )
            final_m = tracker.compute_final_metrics()
            print("\n--- RESUMO DO AMBIENTE EM TEMPO REAL ---")
            print(f" • Acurácia: {final_m['Acuracia']*100:.2f}% | F1-Score: {final_m['F1-Score']*100:.2f}%")
            print(f" • Latência Média por Frame: {final_m['Latencia_Media_ms']:.2f} ms")
            print(f" • Decomposição: MTCNN={final_m['Latencia_MTCNN_ms']:.2f}ms | FaceNet={final_m['Latencia_FaceNet_ms']:.2f}ms | Match={final_m['Latencia_Matching_ms']:.2f}ms")
            print(f" • Throughput Efetivo: {final_m['Throughput_FPS']:.2f} FPS")
            print(f" • Consumo Médio de CPU: {final_m['Avg_CPU_Percent']:.2f}% | Pico RAM: {final_m['Peak_RAM_MB']:.2f} MB")
            print(f" • Combined Efficiency Score (CES): {final_m['CES_Benchmark']:.4f}")
            print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Reconhecimento Facial em Tempo Real com FaceNet e Telemetria de TCC")
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
    parser.add_argument(
        "--test",
        "--ground-truth",
        type=str,
        default=None,
        dest="ground_truth",
        help="Nome do sujeito diante da câmera para cálculo contínuo de Acurácia e F1 no TCC (ex: --test 'Rayssa').",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Desativa o salvamento automático de relatórios e gráficos de telemetria ao fechar.",
    )
    parser.add_argument(
        "--benchmark-csv",
        type=str,
        default=None,
        help="Caminho personalizado para o CSV de métricas do Benchmark prévio.",
    )

    args = parser.parse_args()

    if args.sync:
        print("[Sistema] Sincronizando imagens de dataset/ antes de iniciar...")
        db = FaceDatabase()
        det = FaceDetector()
        rec = FaceNetRecognizer()
        db.sync_from_dataset_folder(det, rec)

    bench_csv = Path(args.benchmark_csv) if args.benchmark_csv else None

    run_recognition_app(
        camera_index=args.cam,
        threshold=args.threshold,
        frame_skip=args.skip,
        metric=args.metric,
        ground_truth=args.ground_truth,
        benchmark_csv=bench_csv,
        auto_save_metrics=not args.no_save,
    )


if __name__ == "__main__":
    main()
