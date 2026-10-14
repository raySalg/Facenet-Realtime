"""
Benchmark Automatizado em Tempo Real para o TCC.
Executa uma sessão cronometrada de reconhecimento facial na webcam com amostragem
contínua de hardware (CPU, RAM, Latência decomposta MTCNN/FaceNet e Vazão FPS),
computando a Acurácia, F1-Score e o Score Combinado de Eficiência (CES).

Consolida automaticamente a Triangulação dos Três Ambientes:
1. Ambiente Controlado
2. Ambiente Não-Controlado (LFW)
3. Ambiente em Tempo Real (Webcam ao Vivo)

Gera tabelas em Markdown/LaTeX e gráficos em 300 DPI para o artigo de TCC.
"""

import argparse
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
from src.realtime_metrics import RealtimeBenchmarkTracker
from src.ui import draw_hud


def run_automated_realtime_benchmark(
    duration_seconds: float = 12.0,
    ground_truth: str = "Rayssa",
    camera_index: int = config.CAMERA_INDEX,
    threshold: float = config.RECOGNITION_THRESHOLD,
    frame_skip: int = config.FRAME_SKIP,
    headless: bool = False,
):
    print("=" * 70)
    print("   BENCHMARK AUTOMATIZADO DE TEMPO REAL - ARTIGO DE TCC")
    print("=" * 70)
    print(f" • Plataforma: {sys.platform}")
    print(f" • Dispositivo de Computação: {config.DEVICE}")
    print(f" • Sujeito Alvo (Ground Truth): '{ground_truth}'")
    print(f" • Duração da Coleta: {duration_seconds} segundos")
    print(f" • Limiar de Decisão (Threshold): {threshold}")
    print("=" * 70 + "\n")

    # Localizar CSV do benchmark prévio
    default_bench_csv = config.BASE_DIR.parent / "Benchmark-de-Modelos-de-Reconhecimento-Facial" / "results" / "metrics_comparison.csv"
    bench_csv = default_bench_csv if default_bench_csv.exists() else None

    # Inicializa rastreador de telemetria
    tracker = RealtimeBenchmarkTracker(ground_truth_subject=ground_truth)

    # Inicializa banco de dados
    database = FaceDatabase()
    persons = database.list_persons()
    print(f"[Sistema] {len(persons)} pessoas cadastradas: {list(persons.keys())}")

    # Inicializa modelos
    detector = FaceDetector()
    recognizer = FaceNetRecognizer()

    # Inicializa câmera
    print("[Sistema] Conectando à câmera para medição...")
    try:
        camera = ThreadedCamera(
            camera_index=camera_index,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            fps=config.CAMERA_FPS,
        ).start()
    except Exception as e:
        print(f"[Erro] Falha ao acessar a câmera: {e}")
        return None

    window_name = "Benchmark Tempo Real - TCC (Gravando Metricas)"
    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)

    fps = 0.0
    frame_count = 0
    fps_start_time = time.time()
    cached_detections = []
    device_str = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"

    start_bench_time = time.time()
    print(f"\n[Telemetria] Coletando métricas em tempo real por {duration_seconds:.1f}s...")

    try:
        while True:
            elapsed_total = time.time() - start_bench_time
            if elapsed_total >= duration_seconds:
                print(f"[Telemetria] Tempo estipulado atingido ({elapsed_total:.1f}s). Finalizando...")
                break

            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            frame_count += 1
            now = time.time()
            elapsed_fps = now - fps_start_time
            if elapsed_fps >= 0.5:
                fps = frame_count / elapsed_fps
                frame_count = 0
                fps_start_time = now

            should_detect = (frame_count % frame_skip == 0) or (len(cached_detections) == 0)

            if should_detect:
                # 1. MTCNN
                t0_det = time.perf_counter()
                boxes, face_tensors, probs = detector.detect_and_extract(frame)
                t_detect_ms = (time.perf_counter() - t0_det) * 1000.0

                new_detections = []
                t_extract_ms = 0.0
                t_match_ms = 0.0

                if face_tensors is not None and face_tensors.shape[0] > 0:
                    # 2. FaceNet Embeddings
                    t0_ext = time.perf_counter()
                    embeddings = recognizer.get_embeddings(face_tensors)
                    t_extract_ms = (time.perf_counter() - t0_ext) * 1000.0

                    # 3. Matching
                    t0_match = time.perf_counter()
                    for i, box in enumerate(boxes):
                        emb = embeddings[i]
                        name, confidence, distance = recognizer.match_face(
                            query_embedding=emb,
                            database=database.database,
                            threshold=threshold,
                            metric="euclidean",
                        )
                        new_detections.append({
                            "box": box,
                            "name": name,
                            "confidence": confidence,
                            "distance": distance,
                        })
                    t_match_ms = (time.perf_counter() - t0_match) * 1000.0

                cached_detections = new_detections

                tracker.record_frame(
                    t_detect_ms=t_detect_ms,
                    t_extract_ms=t_extract_ms,
                    t_match_ms=t_match_ms,
                    detections=cached_detections,
                    instant_fps=fps,
                )

            if not headless:
                live_telemetry = tracker.get_live_metrics(fps)
                display_frame = draw_hud(
                    frame=frame,
                    detections=cached_detections,
                    fps=fps,
                    device_name=device_str,
                    total_registered=len(database.database),
                    paused=False,
                    telemetry=live_telemetry,
                    show_telemetry=True,
                )

                # Desenhar barra de contagem regressiva
                h, w = display_frame.shape[:2]
                progress = min(1.0, elapsed_total / duration_seconds)
                bar_w = int(w * progress)
                cv2.rectangle(display_frame, (0, h - 8), (bar_w, h), (0, 255, 255), -1)

                cv2.imshow(window_name, display_frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    print("[Aviso] Interrompido pelo usuário.")
                    break

    finally:
        camera.stop()
        if not headless:
            cv2.destroyAllWindows()

    # Exportar todos os relatórios e tabelas
    print("\n[Métricas TCC] Consolidando artefatos e gerando gráficos acadêmicos...")
    tracker.export_all_tcc_reports(
        output_dir=config.DATA_DIR,
        controlled_benchmark_csv=bench_csv,
    )

    final_m = tracker.compute_final_metrics()
    print("\n" + "=" * 80)
    print(" RESULTADOS CONSOLIDADOS DO AMBIENTE EM TEMPO REAL PARA O TCC")
    print("=" * 80)
    print(f" • Acurácia de Reconhecimento: {final_m['Acuracia']*100:.2f}%")
    print(f" • F1-Score (Macro):          {final_m['F1-Score']*100:.2f}%")
    print(f" • Latência Média por Frame:  {final_m['Latencia_Media_ms']:.2f} ms")
    print(f"   - Detecção MTCNN:          {final_m['Latencia_MTCNN_ms']:.2f} ms")
    print(f"   - Extração FaceNet 512D:   {final_m['Latencia_FaceNet_ms']:.2f} ms")
    print(f"   - Busca Métrica Vetorial:  {final_m['Latencia_Matching_ms']:.2f} ms")
    print(f" • Vazão Efetiva (Throughput): {final_m['Throughput_FPS']:.2f} FPS")
    print(f" • Uso Médio de CPU:          {final_m['Avg_CPU_Percent']:.2f}%")
    print(f" • Pico de Memória RAM:       {final_m['Peak_RAM_MB']:.2f} MB")
    print(f" • Score Combinado (CES):     {final_m['CES_Benchmark']:.4f}")
    print("=" * 80)
    print(f"\nArquivos gerados em: {config.DATA_DIR}")
    print(" • metricas_tempo_real.md          -> Relatório detalhado em Markdown")
    print(" • tabela_tempo_real.tex           -> Tabela LaTeX pronta para o Overleaf")
    print(" • comparativo_tres_ambientes.md   -> Triangulação dos 3 Ambientes")
    print(" • comparativo_tres_ambientes.csv  -> Dados em formato CSV")
    print(" • tabela_tres_ambientes.tex       -> Tabela LaTeX dos 3 Ambientes")
    print(" • graficos/grafico_telemetria_tempo_real.png")
    print(" • graficos/grafico_triangulacao_ambientes.png\n")

    return final_m


def main():
    parser = argparse.ArgumentParser(description="Benchmark Automatizado do FaceNet em Tempo Real")
    parser.add_argument("--duration", "-d", type=float, default=10.0, help="Duração da coleta em segundos (padrão: 10s)")
    parser.add_argument("--ground-truth", "-g", type=str, default="Rayssa", help="Sujeito ativo diante da câmera (padrão: Rayssa)")
    parser.add_argument("--cam", "-c", type=int, default=config.CAMERA_INDEX, help="Índice da câmera (padrão: 0)")
    parser.add_argument("--headless", action="store_true", help="Executar sem abrir janela gráfica OpenCV")

    args = parser.parse_args()

    run_automated_realtime_benchmark(
        duration_seconds=args.duration,
        ground_truth=args.ground_truth,
        camera_index=args.cam,
        headless=args.headless,
    )


if __name__ == "__main__":
    main()
