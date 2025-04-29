from flask import Blueprint, request, jsonify
from app.services.predict_service import predict_image
from app.db.mongo_connection import collection
from app.utils.utils import padronizar_resultado  # <- importamos a função
from datetime import datetime, timedelta

bp = Blueprint('routes', __name__)

# Upload da imagem
@bp.route('/api/images', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files or 'grupo_id' not in request.form or 'tipo_fruta' not in request.form:
            return jsonify({'error': 'Arquivo, grupo_id ou tipo_fruta ausente'}), 400

        file = request.files['file']
        grupo_id = request.form['grupo_id']
        tipo_fruta = request.form['tipo_fruta']

        if file.filename == '':
            return jsonify({'error': 'Nome do arquivo vazio'}), 400

        image_bytes = file.read()
        result = predict_image(image_bytes, tipo_fruta)

        resultado_padronizado = padronizar_resultado(result['resultado'])

        registro = {
            'grupo_id': grupo_id,
            'nome_arquivo': file.filename,
            'resultado': resultado_padronizado,
            'confianca': result['confianca'],
            'data_analise': result['data_analise']
        }

        collection.insert_one(registro)

        return jsonify({'message': 'Imagem processada com sucesso', 'data': result}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Consulta de imagens (com filtros)
@bp.route('/api/images', methods=['GET'])
def get_images():
    try:
        filtro_resultado = request.args.get('resultado')
        filtro_data = request.args.get('data')

        query = {}

        if filtro_resultado:
            if filtro_resultado in ['defeituosa', 'nao_defeituosa']:
                query['resultado'] = filtro_resultado
            else:
                return jsonify({'error': 'Filtro de resultado inválido'}), 400

        if filtro_data:
            hoje = datetime.utcnow()
            if filtro_data == '30dias':
                data_limite = hoje - timedelta(days=30)
            elif filtro_data == '7dias':
                data_limite = hoje - timedelta(days=7)
            else:
                return jsonify({'error': 'Filtro de data inválido'}), 400

            query['data_analise'] = {'$gte': data_limite.isoformat()}

        registros = list(collection.find(query, {'_id': 0}))
        agrupados = {}

        for r in registros:
            grupo_id = r.get('grupo_id', 'sem_grupo')
            agrupados.setdefault(grupo_id, []).append(r)

        return jsonify({'grupos': agrupados}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
