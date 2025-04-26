from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import uuid

from db import collection  
from app import predict_image

app = Flask(__name__)
CORS(app)

# Upload de imagem
@app.route('/api/images', methods=['POST'])
def upload_image():
    if 'file' not in request.files or 'grupo_id' not in request.form:
        return jsonify({'error': 'Arquivo ou grupo_id ausente'}), 400

    file = request.files['file']
    grupo_id = request.form['grupo_id']

    if file.filename == '':
        return jsonify({'error': 'Nome do arquivo vazio'}), 400

    try:
        image_bytes = file.read()
        result = predict_image(image_bytes)

        registro = {
            'grupo_id': grupo_id,
            'nome_arquivo': file.filename,
            'resultado': result['resultado'],
            'confianca': result['confianca'],
            'data_analise': result['data_analise']
        }

        collection.insert_one(registro)

        return jsonify({'resultado': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Listagem e filtro de imagens
@app.route('/api/images', methods=['GET'])
def listar_ou_filtrar_imagens():
    try:
        filtro = request.args.get('resultado')  # Pega query param /api/image?resultado=Defeituosa
        query = {}

        if filtro:
            query['resultado'] = filtro

        registros = list(collection.find(query, {'_id': 0}))
        agrupados = {}

        for r in registros:
            grupo_id = r.get('grupo_id', 'sem_grupo')
            if grupo_id not in agrupados:
                agrupados[grupo_id] = []
            agrupados[grupo_id].append(r)

        return jsonify(agrupados)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
