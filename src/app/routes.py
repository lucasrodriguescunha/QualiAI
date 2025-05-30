from flask import Blueprint, request, jsonify
from app.services.predict_service import predict_image
from app.db.mongo_connection import collection
from app.utils.utils import padronizar_resultado, padronizar_nome_arquivo
from datetime import datetime, timedelta

bp = Blueprint('routes', __name__)

@bp.route('/api/images', methods=['POST'])
def upload_image_by_file():
    try:
        if 'file' not in request.files or 'grupo_id' not in request.form or 'tipo_fruta' not in request.form:
            return jsonify({'error': 'Arquivo, grupo_id ou tipo_fruta ausente'}), 400

        file = request.files['file']
        grupo_id = request.form['grupo_id']
        tipo_fruta = padronizar_resultado(request.form['tipo_fruta']) 
        id_usuario = request.form.get('id_usuario')

        if file.filename == '':
            return jsonify({'error': 'Nome do arquivo vazio'}), 400

        nome_padronizado = padronizar_nome_arquivo(file.filename)
        image_bytes = file.read()
        result = predict_image(image_bytes, tipo_fruta)

        resultado_padronizado = padronizar_resultado(result['resultado'])

        registro = {
            'grupo_id': grupo_id,
            'nome_arquivo': nome_padronizado,
            'resultado': resultado_padronizado,
            'confianca': result['confianca'],
            'data_analise': result['data_analise'],
            'tipo_fruta': tipo_fruta,
            'id_usuario': id_usuario
        }

        collection.insert_one(registro)

        return jsonify({'message': 'Imagem processada com sucesso', 'data': result}), 201

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/images', methods=['GET'])
def get_images():
    try:
        filtro_resultado = request.args.get('resultado')
        filtro_data = request.args.get('data')
        filtro_produto = request.args.get('tipo_fruta')

        query = {}

        if filtro_resultado and filtro_resultado != 'todas':
            resultado_padronizado = padronizar_resultado(filtro_resultado)
            if resultado_padronizado in ['defeituosa', 'nao_defeituosa']:
                query['resultado'] = resultado_padronizado
            else:
                return jsonify({'error': 'Filtro de resultado inválido'}), 400

        if filtro_data and filtro_data != 'todas':
            hoje = datetime.now()
            if filtro_data == '30dias':
                data_limite = hoje - timedelta(days=30)
            elif filtro_data == '7dias':
                data_limite = hoje - timedelta(days=7)
            elif filtro_data == '24horas':
                data_limite = hoje - timedelta(hours=24)
            else:
                return jsonify({'error': 'Filtro de data inválido'}), 400

            query['data_analise'] = {'$gte': data_limite}

        if filtro_produto and filtro_produto != 'todas':
            query['tipo_fruta'] = padronizar_resultado(filtro_produto)

        registros = list(collection.find(query, {'_id': 0}))
        agrupados = {}

        for r in registros:
            grupo_id = r.get('grupo_id', 'sem_grupo')
            r.pop('grupo_id', None) 
            agrupados.setdefault(grupo_id, []).append(r)

        return jsonify({'grupos': agrupados}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
