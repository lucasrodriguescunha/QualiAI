import numpy as np
from tensorflow import keras
from PIL import Image
from datetime import datetime
import io
import os

def predict_image(image_bytes, tipo_fruta):
    try:
        # Decide o modelo baseado na fruta selecionada
        if tipo_fruta == "Maçãs":
            model_path = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'model', 'modelo_maca.h5'))
        elif tipo_fruta == "Mangas":
            model_path = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'model', 'modelo_manga.h5'))
        else:
            raise ValueError("Fruta inválida ou modelo não encontrado.")

        model = keras.models.load_model(model_path)

        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((256, 256))
        img_array = keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        score = float(np.squeeze(prediction))
        is_not_defective = score > 0.5

        resultado = "Não defeituosa" if is_not_defective else "Defeituosa"
        confianca = round(score * 100, 2) if is_not_defective else round((1 - score) * 100, 2)
        data_analise = datetime.now()

        return {
            'resultado': resultado,
            'data_analise': data_analise,
            'confianca': confianca
        }
    except Exception as e:
        raise RuntimeError(f"Erro ao processar a imagem: {str(e)}")
