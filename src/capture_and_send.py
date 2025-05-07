import cv2
import requests
import io

# Dados fixos
GRUPO_ID = "grupo_teste"
TIPO_FRUTA = "mangas"  # ou "macas"
API_URL = "http://127.0.0.1:5000/api/images"

# Captura imagem da webcam
cap = cv2.VideoCapture(0)
print("Pressione 's' para capturar a imagem ou 'q' para sair")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar a webcam")
        break

    cv2.imshow("Webcam - Pressione 's' para capturar", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = io.BytesIO(buffer).getvalue()

        # Envia a imagem para a API
        files = {'file': ('imagem_webcam.jpg', image_bytes, 'image/jpeg')}
        data = {'grupo_id': GRUPO_ID, 'tipo_fruta': TIPO_FRUTA}

        response = requests.post(API_URL, files=files, data=data)
        print("Resposta da API:", response.json())
        break

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
