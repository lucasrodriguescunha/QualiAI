import numpy as np
from tensorflow import keras
from PIL import Image
from datetime import datetime
import io

# Carrega o modelo treinado
model = keras.models.load_model('C:\\Users\\Lucas\\Documents\\QualiAI\\src\\apple_defect_model.h5')

def predict_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((256, 256))
        img_array = keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        score = float(np.squeeze(prediction))
        resultado = "Não defeituosa" if prediction[0] > 0.5 else "Defeituosa"
        data_analise = datetime.now().strftime('%d/%m/%Y %H:%M')

        return {
            'resultado': resultado,
            'data_analise': data_analise,
            'confianca': round(score * 100, 2)
        }
    except Exception as e:
        raise RuntimeError(f"Erro ao processar a imagem: {str(e)}")
