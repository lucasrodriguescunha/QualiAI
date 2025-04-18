from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

model = keras.models.load_model('QualiAI\src\\apple_defect_model.h5')

@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nome do arquivo vazio'}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((256, 256))
        img_array = keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        prediction = model.predict(img_array)
        result = "Não defeituosa" if prediction[0] > 0.5 else "Defeituosa"

        return jsonify({'resultado': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
