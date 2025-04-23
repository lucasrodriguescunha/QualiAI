from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

from db import collection  

from app import predict_image

app = Flask(__name__)
CORS(app)

@app.route('/api/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nome do arquivo vazio'}), 400

    try:
        image_bytes = file.read()
        result = predict_image(image_bytes)

        collection.insert_one({
            'nome_arquivo': file.filename,
            'resultado': result['resultado'],
            'confianca': result['confianca'],
            'data_analise': result['data_analise']
        })

        return jsonify({'relatorio': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/listar', methods=['GET'])
def listar_resultados():
    resultados = list(collection.find({}, {'_id': 0}))
    return jsonify(resultados)

if __name__ == '__main__':
    app.run(debug=True)
