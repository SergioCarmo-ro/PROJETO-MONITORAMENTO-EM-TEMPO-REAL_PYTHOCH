# 🧠 PROJETO DE MONITORAMENTO EM TEMPO REAL — PyTorch

Este projeto tem como objetivo implementar um **sistema de monitoramento em tempo real** utilizando **redes neurais profundas (Deep Learning)** com **PyTorch**, integrando visão computacional, aprendizado supervisionado e técnicas de otimização modernas para reconhecimento e classificação de imagens.

---

## 🏗️ Arquitetura Resumida

O sistema segue uma **arquitetura modular** baseada em PyTorch, composta por:

1. **Pré-processamento de Imagens:**  
   Utiliza `OpenCV` e `Pillow (PIL)` para leitura e transformação das imagens.

2. **Dataset Personalizado (Custom Dataset):**  
   Classe `Dataset` customizada para carregar dados de imagens, aplicar transformações (`torchvision.transforms.functional`) e dividir em lotes (`DataLoader`).

3. **Modelos de Deep Learning:**  
   - **ResNet-18** (pré-treinada com `ResNet18_Weights.IMAGENET1K_V1`)  
   - **VGG-16** (pré-treinada com `VGG16_Weights.IMAGENET1K_V1`)

4. **Camadas de Treinamento e Validação:**  
   Implementação de funções de perda (`nn.CrossEntropyLoss`), otimizadores (`optim.Adam`, `optim.SGD`), e ajuste dinâmico de taxa de aprendizado (`ReduceLROnPlateau`).

5. **Visualização dos Resultados:**  
   Utilização de `matplotlib` para gerar gráficos de precisão, perda e resultados qualitativos de inferência.

---

## 🧩 Estrutura Simplificada do Projeto

