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

PROJETO-MONITORAMENTO-EM-TEMPO-REAL_PYTHOCH/

│

├── dataset/

│ ├── train/

│ ├── val/

│ └── test/

│

├── models/

│ ├── resnet_model.py

│ ├── vgg_model.py

│ └── init.py

│

├── utils/

│ ├── preprocessing.py

│ ├── visualization.py

│ └── dataset_loader.py

│

├── main.py

├── requirements.txt

└── README.md


---

## ⚙️ Tecnologias Utilizadas

- 🐍 **Python 3.10+**
- 🧠 **PyTorch**
- 🧩 **Torchvision**
- 🧮 **NumPy**
- 🖼️ **OpenCV**
- 🖌️ **Pillow (PIL)**
- 📊 **Matplotlib**
- 🔁 **TQDM** (para barras de progresso)
- ☁️ **Google Colab** (ambiente de execução recomendado)

---

## 🚀 Funcionalidades do Protótipo

- 🔹 Carregamento e pré-processamento automático de imagens.  
- 🔹 Treinamento com **ResNet18** ou **VGG16** pré-treinadas.  
- 🔹 Visualização gráfica da evolução do treinamento.  
- 🔹 Ajuste automático da **taxa de aprendizado** (scheduler).  
- 🔹 Detecção e classificação em tempo real.  
- 🔹 Modularização do código para fácil manutenção e expansão.  

---

## 🧠 Como Executar o Sistema

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/SergioCarmo-ro/PROJETO-MONITORAMENTO-EM-TEMPO-REAL_PYTHOCH.git
   cd PROJETO-MONITORAMENTO-EM-TEMPO-REAL_PYTHOCH


Crie o ambiente virtual e instale as dependências:

python -m venv venv
source venv/bin/activate     # (Linux/macOS)
venv\Scripts\activate        # (Windows)
pip install -r requirements.txt


Execute o script principal:

python main.py


(Opcional) Execute no Google Colab:

Faça upload do repositório no Colab.

Ajuste o caminho dos datasets.

Execute as células sequencialmente.

📦 Exemplo de Importação das Principais Bibliotecas
import os
import random
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, vgg16
from torchvision.models.resnet import ResNet18_Weights
from torchvision.models.vgg import VGG16_Weights
import copy
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

👨‍💻 Autor

Sérgio Ademir Rocha do Carmo

📍 Amazonas – Brasil

💡 Desenvolvedor e Pesquisador em Sistemas Inteligentes e Visão Computacional.

🧾 Licença

Este projeto é de uso acadêmico e experimental.

© 2025 Sérgio Ademir Rocha do Carmo – Todos os direitos reservados.
