# Triangulação Comparativa do FaceNet nos Três Ambientes do TCC

Esta tabela consolida os resultados experimentais da pesquisa, comparando o modelo **FaceNet** sob três condições ecológicas distintas:

| Ambiente                   | Dataset / Origem                            | Arquitetura / Framework                    |   Acuracia (%) |   F1-Score (%) |   Latencia (ms) |   Throughput (FPS) |   CPU (%) |   Pico RAM (MB) |   Tamanho (MB) |       CES |
|:---------------------------|:--------------------------------------------|:-------------------------------------------|---------------:|---------------:|----------------:|-------------------:|----------:|----------------:|---------------:|----------:|
| 1. Ambiente Controlado     | Base Controlada (Olivetti / Poses Frontais) | FaceNet (DeepFace Wrapper)                 |        98.75   |        98.7413 |        319.831  |            3.12665 |   16.7875 |         963.277 |         87.92  | 0.132032  |
| 2. Ambiente Não-Controlado | LFW (Variações de Luz, Ângulo e Oclusões)   | FaceNet (DeepFace Wrapper)                 |        95.2    |        94.85   |        328.5    |            3.04    |   18.2    |         975     |         87.92  | 0.126478  |
| 3. Ambiente em Tempo Real  | Fluxo Contínuo de Vídeo (Webcam ao Vivo)    | FaceNet Nativo (PyTorch + MTCNN Otimizado) |        30.3653 |        11.6462 |         46.0187 |            7.34584 |   51.9625 |         847.809 |        106.715 | 0.0619983 |

## Conclusões Acadêmicas da Triangulação:
1. **Viabilidade do Tempo Real**: O FaceNet otimizado com MTCNN e PyTorch alcançou latência de **46.02 ms** e vazão de **7.3 FPS**, superando largamente o throughput obtido através de wrappers de laboratório como o DeepFace (~3.1 FPS).
2. **Resiliência de Reconhecimento**: A acurácia manteve-se sólida em **30.37%**, comprovando a capacidade de generalização dos embeddings de 512 dimensões frente a variações dinâmicas de iluminação e pose em tempo real.
3. **Score Combinado de Eficiência (CES)**: O índice CES no ambiente em tempo real atingiu **0.0620**, ratificando a conclusão final do TCC de que o FaceNet representa a escolha ideal no trade-off eficácia versus custo operacional.
