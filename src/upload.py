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
        response = predict_image(file.read())

        # collection.insert_one({
        #     'resultado': response['resultado'],
        #     'confianca': response['confianca'],
        #     'data_analise': response.get('data_analise', datetime.now().isoformat())
        # })

        return jsonify({'resultado': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/listar', methods=['GET'])
def listar_resultados():
    resultados = list(collection.find({}, {'_id': 0}))
    return jsonify(resultados)

if __name__ == '__main__':
    app.run(debug=True)
