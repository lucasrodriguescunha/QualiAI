from flask import Blueprint, request, jsonify
from app.services.predict_service import predict_image
from app.db.mongo_connection import collection

bp = Blueprint('routes', __name__)

@bp.route('/api/images', methods=['POST'])
def upload_image():
    if 'file' not in request.files or 'grupo_id' not in request.form or 'tipo_fruta' not in request.form:
        return jsonify({'error': 'Arquivo, grupo_id ou tipo_fruta ausente'}), 400

    file = request.files['file']
    grupo_id = request.form['grupo_id']
    tipo_fruta = request.form['tipo_fruta']

    if file.filename == '':
        return jsonify({'error': 'Nome do arquivo vazio'}), 400

    try:
        image_bytes = file.read()
        result = predict_image(image_bytes, tipo_fruta)

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

@bp.route('/api/images', methods=['GET'])
def listar_ou_filtrar_imagens():
    try:
        filtro = request.args.get('resultado')
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
