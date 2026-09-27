"""
Módulo de Métricas de Avaliação e Eficiência Computacional (CES).
Calcula Acurácia, F1-Score Macro, Latência, Throughput, Memória RAM, CPU e CES.
"""

import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import psutil
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculate_classification_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """
    Calcula as métricas de eficácia de modelagem:
    - Acurácia global
    - F1-Score Macro (balanceado entre todas as identidades)
    - Precisão Macro e Revocação Macro
    """
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    return {
        "acuracia": acc,
        "f1_macro": f1_macro,
        "precisao_macro": prec_macro,
        "revocacao_macro": rec_macro,
    }


def calculate_ces(
    acuracia: float,
    f1_macro: float,
    latencia_ms: float,
    ram_mb: float = 150.0,
    tamanho_disco_mb: float = 100.0,
    cpu_percent: float = 35.0,
    mode: str = "artigo",
) -> Dict[str, float]:
    """
    Calcula o Score Combinado de Eficiência (CES - Combined Efficiency Score).

    O uso de logaritmos de base 10 no denominador atua como amortecedor essencial:
    Comprime a diferença de ordens de magnitude dos custos computacionais, evitando
    que redes profundas sejam desproporcionalmente penalizadas em relação a métodos triviais.

    Exemplo prático:
    - Latência de 1 ms   -> penalidade ~ 1.00
    - Latência de 100 ms -> penalidade ~ 2.03 (100x mais lento gera penalidade apenas 2x maior)
    """
    # Eficácia composta (média harmônica ou aritmética entre Acurácia e F1-Score Macro)
    # Valores normalizados no intervalo [0.0, 1.0]
    eficacia = (acuracia + f1_macro) / 2.0

    # Amortecedor logarítmico para Latência
    # f(L) = log10(L + 9): para L=1ms => log10(10)=1.0; para L=100ms => log10(109)≈2.03
    penalidade_latencia = math.log10(max(1.0, latencia_ms) + 9.0)

    # Amortecedores para Recursos Físicos (RAM, Disco e CPU)
    penalidade_ram = math.log10(max(1.0, ram_mb) + 9.0)
    penalidade_disco = math.log10(max(0.1, tamanho_disco_mb) + 9.0)
    penalidade_cpu = math.log10(max(1.0, cpu_percent) + 9.0)

    # 1. CES Latência Direta (fórmula canônica do artigo)
    # CES_lat = Eficácia / log10(L + 9)
    ces_latencia = eficacia / penalidade_latencia

    # 2. CES Holístico Global (Pondera Latência, Memória RAM, Tamanho em Disco e CPU)
    # Pesos padrão: 50% Latência, 20% RAM, 15% Tamanho, 15% CPU
    custo_ponderado = (
        0.50 * penalidade_latencia
        + 0.20 * penalidade_ram
        + 0.15 * penalidade_disco
        + 0.15 * penalidade_cpu
    )
    ces_global = eficacia / custo_ponderado

    return {
        "eficacia": eficacia,
        "penalidade_latencia": penalidade_latencia,
        "ces_latencia": ces_latencia,
        "ces_global": ces_global,
        "ces_normalizado_100": ces_global * 100.0,
    }


class ResourceMonitor:
    """Monitor de consumo de recursos de hardware em tempo real."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def get_ram_usage_mb(self) -> float:
        """Retorna o consumo de memória RAM do processo atual em MB."""
        return self.process.memory_info().rss / (1024.0 * 1024.0)

    def get_cpu_percent(self) -> float:
        """Retorna o uso percentual de CPU do processo."""
        try:
            return self.process.cpu_percent(interval=0.05)
        except Exception:
            return 0.0
