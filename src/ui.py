"""
Módulo de Interface e Renderização Visual (HUD) em OpenCV.
Proporciona uma estética moderna, elegante e informativa sobre os frames da câmera.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from src import config


def draw_hud(
    frame: np.ndarray,
    detections: List[dict],
    fps: float,
    device_name: str,
    total_registered: int,
    paused: bool = False,
) -> np.ndarray:
    """
    Desenha os elementos de interface sobre o frame:
    - Bounding boxes com cantos estilizados (Corner Brackets HUD)
    - Badges com nome e porcentagem de confiança
    - Barra de status superior com FPS, Dispositivo e Atalhos
    """
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # 1. Desenhar a barra superior semi-transparente (Dashboard)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 42), config.COLOR_BG_DARK, -1)
    # Mescla com 75% de opacidade
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # Informações da barra superior
    fps_text = f"FPS: {fps:4.1f}"
    device_text = f"Device: {device_name.upper()}"
    registered_text = f"Cadastrados: {total_registered}"
    controls_text = "[Q] Sair  |  [C] Cadastrar  |  [R] Recarregar"

    cv2.putText(canvas, "FACENET AI", (14, 26), cv2.FONT_HERSHEY_DUPLEX, 0.65, config.COLOR_PRIMARY, 2, cv2.LINE_AA)
    cv2.putText(canvas, fps_text, (150, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(canvas, device_text, (260, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(canvas, registered_text, (410, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLOR_TEXT, 1, cv2.LINE_AA)

    # Atalhos alinhados à direita
    (cw, _), _ = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(canvas, controls_text, (max(10, w - cw - 14), 26), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # 2. Desenhar as detecções faciais
    for det in detections:
        box = det["box"]
        name = det["name"]
        conf = det["confidence"]
        dist = det.get("distance", 0.0)

        x1, y1, x2, y2 = box
        is_known = name != "Desconhecido"
        color = config.COLOR_SUCCESS if is_known else config.COLOR_UNKNOWN

        # Desenhar caixa delimitadora fina
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        # Desenhar cantos reforçados (Bracket corners estilo HUD)
        corner_len = max(10, min(int((x2 - x1) * 0.2), 22))
        thickness = 3

        # Superior esquerdo
        cv2.line(canvas, (x1, y1), (x1 + corner_len, y1), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x1, y1), (x1, y1 + corner_len), color, thickness, cv2.LINE_AA)
        # Superior direito
        cv2.line(canvas, (x2, y1), (x2 - corner_len, y1), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x2, y1), (x2, y1 + corner_len), color, thickness, cv2.LINE_AA)
        # Inferior esquerdo
        cv2.line(canvas, (x1, y2), (x1 + corner_len, y2), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x1, y2), (x1, y2 - corner_len), color, thickness, cv2.LINE_AA)
        # Inferior direito
        cv2.line(canvas, (x2, y2), (x2 - corner_len, y2), color, thickness, cv2.LINE_AA)
        cv2.line(canvas, (x2, y2), (x2, y2 - corner_len), color, thickness, cv2.LINE_AA)

        # Texto do Badge
        if is_known:
            label_text = f"{name} ({conf:.0f}%)"
        else:
            label_text = f"{name}"

        # Fundo do Badge acima da caixa
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        font_thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

        badge_y1 = max(45, y1 - label_h - 10)
        badge_y2 = badge_y1 + label_h + 8
        badge_x1 = x1
        badge_x2 = x1 + label_w + 14

        # Desenhar fundo do badge
        cv2.rectangle(canvas, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
        # Texto do badge
        text_color = (0, 0, 0) if is_known else (255, 255, 255)
        cv2.putText(
            canvas,
            label_text,
            (badge_x1 + 7, badge_y2 - 6),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

    # 3. Aviso se o vídeo estiver pausado
    if paused:
        pause_overlay = canvas.copy()
        cv2.rectangle(pause_overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(pause_overlay, 0.45, canvas, 0.55, 0, canvas)
        msg = "PAUSADO [Espaço para continuar]"
        (mw, mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)
        cv2.putText(canvas, msg, (int((w - mw) / 2), int((h + mh) / 2)), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


def draw_registration_overlay(
    frame: np.ndarray,
    person_name: str,
    captured_count: int,
    target_count: int,
    instruction: str = "Olhe para a camera e mova levemente o rosto",
) -> np.ndarray:
    """Renderiza a interface durante o processo de captura e cadastro de novo rosto."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]

    # Barra inferior com informações do cadastro
    bar_height = 80
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, h - bar_height), (w, h), config.COLOR_BG_DARK, -1)
    cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0, canvas)

    # Título do cadastro
    title = f"CADASTRANDO: {person_name.upper()}  [{captured_count}/{target_count}]"
    cv2.putText(canvas, title, (20, h - bar_height + 28), cv2.FONT_HERSHEY_DUPLEX, 0.65, config.COLOR_PRIMARY, 2, cv2.LINE_AA)

    # Barra de Progresso
    progress = captured_count / max(1, target_count)
    bar_x = 20
    bar_y = h - 35
    bar_w = w - 40
    bar_h = 10

    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill_w = int(bar_w * progress)
    if fill_w > 0:
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), config.COLOR_SUCCESS, -1)

    # Instrução ao usuário
    cv2.putText(canvas, instruction, (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return canvas
