"""
Gerenciador de Câmera Cross-Platform com suporte a Threading.
Garante inicialização rápida e sem atrasos (lag) de buffer no Windows e Linux.
"""

import platform
import threading
import time
import cv2
from typing import Optional, Tuple


class ThreadedCamera:
    """
    Captura de vídeo em thread dedicada para evitar latência acumulada
    no buffer do OpenCV durante o processamento do FaceNet.
    """

    def __init__(self, camera_index: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        self.cap: Optional[cv2.VideoCapture] = None
        self.current_frame = None
        self.is_running = False
        self.lock = threading.Lock()
        self.thread: Optional[threading.Thread] = None

        self._init_camera()

    def _init_camera(self):
        """Inicializa a câmera testando backends adequados para o sistema operacional."""
        system = platform.system()
        
        # Ordem de preferência de backends por OS
        if system == "Windows":
            # cv2.CAP_DSHOW (DirectShow) é crucial no Windows para evitar atrasos e travamentos
            candidate_backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        elif system == "Linux":
            # cv2.CAP_V4L2 (Video4Linux2) é o padrão mais estável no Linux
            candidate_backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        else:
            candidate_backends = [cv2.CAP_ANY]

        opened = False
        for backend in candidate_backends:
            try:
                cap = cv2.VideoCapture(self.camera_index, backend)
                if cap.isOpened():
                    # Testar leitura de um frame inicial
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        self.cap = cap
                        opened = True
                        break
                    else:
                        cap.release()
            except Exception:
                continue

        # Se não abriu com os backends específicos, tenta o padrão sem especificar backend
        if not opened:
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(
                f"Não foi possível abrir a câmera (índice {self.camera_index}).\n"
                f"Verifique se ela está conectada e se você concedeu permissão de acesso à câmera no seu sistema ({system})."
            )

        # Configurar resolução e taxa de quadros desejada
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Ler primeiro frame para validação
        ret, frame = self.cap.read()
        if ret and frame is not None:
            self.current_frame = frame

    def start(self) -> "ThreadedCamera":
        """Inicia a thread de leitura contínua."""
        if self.is_running:
            return self

        self.is_running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        # Aguarda breve momento para o primeiro frame estar disponível
        time.sleep(0.05)
        return self

    def _capture_loop(self):
        """Loop executado em segundo plano para consumir frames continuamente."""
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.current_frame = frame
            else:
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[cv2.typing.MatLike]]:
        """Retorna o frame mais recente capturado (sem atraso de buffer)."""
        with self.lock:
            if self.current_frame is None:
                return False, None
            return True, self.current_frame.copy()

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def stop(self):
        """Para a thread e libera a câmera."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
