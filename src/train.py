import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import os

# Definir caminhos para os dados
data_dir = "QualiAi/dataset"  # Atualize com o caminho correto
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

# Parâmetros do modelo
img_size = (256, 256)
batch_size = 32

# Carregar os datasets usando image_dataset_from_directory
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="training",
    seed=123
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset="validation",
    seed=123
)

# Normalização e aumentação dos dados
data_augmentation = keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Aplicar a normalização ao dataset
autotune = tf.data.AUTOTUNE
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y)).cache().shuffle(1000).prefetch(buffer_size=autotune)
val_dataset = val_dataset.cache().prefetch(buffer_size=autotune)

# Construção da CNN
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Treinar o modelo
epochs = 10
history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

# Avaliação do modelo
plt.plot(history.history['accuracy'], label='Acurácia Treino')
plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()

# Salvar o modelo treinado
model.save("apple_defect_model.h5")

print("Modelo treinado e salvo com sucesso!")
