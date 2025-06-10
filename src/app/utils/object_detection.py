from ultralytics import YOLO
import os
from PIL import Image
import numpy as np

def detect_and_crop_fruit(image: Image.Image, tipo_fruta: str, model_path: str):
    """
    Detecta a fruta na imagem usando YOLOv8 e recorta a região correspondente.
    Retorna a imagem recortada ou a imagem original se nenhuma fruta for detectada.
    
    Args:
        image (PIL.Image): Imagem de entrada.
        tipo_fruta (str): Tipo de fruta ('macas' ou 'mangas').
        model_path (str): Caminho para o modelo YOLOv8.
    
    Returns:
        PIL.Image: Imagem recortada da fruta ou imagem original se não detectada.
    """
    try:
        # Mapear tipo_fruta para classes do YOLO (baseado no COCO dataset)
        class_name = "apple" if tipo_fruta == "macas" else "mango"
        
        # Carregar o modelo YOLOv8
        model = YOLO(model_path)
        
        # Converter imagem PIL para formato compatível com YOLO
        img_array = np.array(image)
        
        # Executar detecção
        results = model(img_array, verbose=False)[0]
        
        # Filtrar detecções para a classe desejada
        detections = []
        for box in results.boxes:
            if results.names[int(box.cls)] == class_name:
                detections.append({
                    'conf': box.conf.cpu().numpy()[0],
                    'xyxy': box.xyxy.cpu().numpy()[0]
                })
        
        if not detections:
            print(f"Nenhuma {class_name} detectada na imagem. Usando imagem original.")
            return image
        
        # Selecionar a detecção com maior confiança
        best_detection = max(detections, key=lambda x: x['conf'])
        x1, y1, x2, y2 = map(int, best_detection['xyxy'])
        
        # Recortar a imagem
        cropped_image = image.crop((x1, y1, x2, y2))
        
        return cropped_image
    
    except Exception as e:
        print(f"Erro ao detectar e recortar a fruta: {str(e)}")
        return image  # Retorna a imagem original em caso de erro