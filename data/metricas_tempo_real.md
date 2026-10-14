# Métricas de Reconhecimento Facial em Tempo Real (FaceNet)

Este relatório consolida a avaliação experimental do modelo **FaceNet (InceptionResnetV1 + MTCNN)** em execução contínua no ambiente em **Tempo Real (Webcam Streaming)**.

## 1. Resumo Executivo das Métricas

| Métrica | Valor Obtido | Descrição Técnica |
| :--- | :---: | :--- |
| **Acurácia Global** | **30.37%** | Fração de predições corretas durante o streaming |
| **F1-Score (Macro)** | **11.65%** | Média harmônica balanceada entre Precisão e Revocação |
| **Precisão (Macro)** | **25.00%** | Confiabilidade das identificações positivas |
| **Revocação (Macro)** | **7.59%** | Capacidade de identificar sujeitos cadastrados |
| **Latência Média por Frame** | **46.02 ms** | Tempo total do ciclo de processamento (MTCNN + FaceNet + Match) |
| **Latência P50 (Mediana)** | **57.77 ms** | 50% dos frames foram processados abaixo deste tempo |
| **Latência P95 (Percentil 95)** | **79.68 ms** | Limite superior para 95% das inferências |
| **Decomposição: MTCNN** | **24.86 ms** | Detecção, alinhamento facial e extração de tensor 160x160 |
| **Decomposição: FaceNet** | **21.09 ms** | Forward pass da CNN gerando embeddings de 512D |
| **Decomposição: Matching** | **0.07 ms** | Busca vetorial Euclidiana/Cosseno no banco |
| **Vazão Real (Throughput)** | **7.35 FPS** | Taxa efetiva de quadros por segundo em regime contínuo |
| **Uso Médio de CPU** | **51.96%** | Processamento computacional normalizado por núcleo |
| **Trabalho de CPU (CPU Work)** | **65.4320 %-s** | Esforço acumulado de CPU durante a sessão de teste |
| **Pico de Memória RAM** | **847.81 MB** | Consumo físico máximo alocado (Resident Set Size) |
| **Tamanho do Modelo em Disco** | **106.7 MB** | Espaço físico dos pesos das redes (InceptionResnetV1 + MTCNN) |
| **Score de Eficiência (CES)** | **0.0620** | $CES = \frac{Acurácia}{\log_{10}(Latência + 1) \times \log_{10}(RAM + 1)}$ |
| **Duração da Sessão** | **125.9 s** | Tempo total de streaming analisado |
| **Total de Frames Analisados** | **925 frames** | Quantidade total de quadros avaliados |
| **Amostras com Ground Truth** | **438 faces** | Predições comparadas diretamente contra sujeito ativo |

---

## 2. Decomposição da Latência por Etapa do Pipeline

* **Detecção e Alinhamento Facial (MTCNN)**: 24.86 ms (54.0% do tempo)
* **Extração de Características (FaceNet 512D)**: 21.09 ms (45.8% do tempo)
* **Comparação Métrica Vetorial**: 0.07 ms (0.2% do tempo)
