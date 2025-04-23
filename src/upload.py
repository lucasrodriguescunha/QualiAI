from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Conexão com MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['quali_ai']
collection = db['resultados']

# Rota de upload
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

        # Salvar no MongoDB
        collection.insert_one({
            'nome_arquivo': file.filename,
            'resultado': result['resultado'],
            'confianca': result['confianca'],
            'data_analise': result['data_analise']
        })

        return jsonify({'resultado': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Rota para listar os resultados
@app.route('/api/listar', methods=['GET'])
def listar_resultados():
    resultados = list(collection.find({}, {'_id': 0}))
    return jsonify(resultados)

if __name__ == '__main__':
    app.run(debug=True)
