import numpy as np
from tensorflow import keras
from PIL import Image
from datetime import datetime
import io
import os
from app.utils.background_removal import remove_background_and_center
from app.utils.object_detection import detect_and_crop_fruit

def save_preprocessed_image(image: Image.Image):
    """
    Salva a imagem preprocessada em uma pasta local chamada 'processed_images'.
    Retorna o caminho salvo.
    """
    output_dir = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'processed_images'))
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    filename = f"imagemprocessada{timestamp}.jpg"
    path_salvo = os.path.join(output_dir, filename)

    image.save(path_salvo, format='JPEG')
    return path_salvo

def predict_image(image_bytes, tipo_fruta):
    try:
        if tipo_fruta == "macas":
            model_path = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'model', 'modelo_maca.h5'))
        elif tipo_fruta == "mangas":
            model_path = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'model', 'modelo_manga.h5'))
        else:
            raise ValueError("Fruta inválida ou modelo não encontrado.")

        model = keras.models.load_model(model_path)

        # Caminho para o modelo YOLOv8
        yolo_model_path = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'model', 'yolov8n.pt'))

        # Abrir imagem original
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Detectar e recortar a fruta
        cropped_image = detect_and_crop_fruit(original_image, tipo_fruta, yolo_model_path)

        # Aplicar remoção de fundo + centralização na imagem recortada
        preprocessed_img = remove_background_and_center(cropped_image)

        # Salvar imagem processada
        # save_preprocessed_image(preprocessed_img)

        # Preparar imagem para predição
        img_array = keras.utils.img_to_array(preprocessed_img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

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