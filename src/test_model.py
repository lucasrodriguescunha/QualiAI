import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import os

# Carregar o modelo treinado
model = keras.models.load_model("apple_defect_model.h5")

# Caminho para a imagem de teste
img_path = "QualiAi/test3.png"  # Atualize com o caminho correto

# Carregar a imagem e redimensioná-la
img = keras.utils.load_img(img_path, target_size=(256, 256))

# Converter a imagem em um array
img_array = keras.utils.img_to_array(img)

# Adicionar uma dimensão extra para criar um lote de tamanho 1
img_array = np.expand_dims(img_array, axis=0)

# Normalizar os pixels se necessário (por exemplo, dividir por 255)
img_array /= 255.0

# Fazer a previsão
prediction = model.predict(img_array)

# Interpretar a previsão
if prediction[0] > 0.5:
    print("A imagem é classificada como: Classe 1: Não defeituosa")
else:
    print("A imagem é classificada como: Classe 0: Defeituosa")