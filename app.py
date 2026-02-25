from flask import Flask, render_template, request, jsonify, Response, flash, redirect, url_for
from flask_cors import CORS
import cv2
import face_recognition
import numpy as np
import base64
from datetime import datetime
import os
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from io import BytesIO, StringIO
import mysql.connector
from dotenv import load_dotenv
import time
import logging
import csv
import requests
from threading import Lock
import io


# Importe o Swagger
from flasgger import Swagger 

app = Flask(__name__)
# Inicialize o Swagger
swagger = Swagger(app, template={
    "info": {
        "title": "API de Reconhecimento Facial (Panorama)",
        "description": "API para cadastro e autenticação de usuários via biometria facial.",
        "version": "1.0.0"
    }
})


# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
CORS(app)
app.secret_key = 'sua_chave_secreta_muito_longa_e_aleatoria'

# Configuração do banco de dados
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'reconhecimento_facial'),
    'port': int(os.getenv('DB_PORT', 3306))
}

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'fotos')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cache para encodings do banco (melhora performance)
encodings_cache = {}
last_cache_update = 0
CACHE_DURATION = 300  # 5 minutos

def get_db_connection():
    try:
        connection = mysql.connector.connect(**db_config)
        return connection
    except mysql.connector.Error as e:
        logger.error(f"Erro ao conectar ao banco: {e}")
        return None

def load_encodings_cache():
    """Carrega encodings do banco em cache para melhor performance"""
    global encodings_cache, last_cache_update
    
    current_time = time.time()
    if current_time - last_cache_update < CACHE_DURATION and encodings_cache:
        return encodings_cache
    
    logger.info("Atualizando cache de encodings...")
    start_time = time.time()
    
    conn = get_db_connection()
    if not conn:
        return {}
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT u.id, u.nome, f.caminho 
            FROM usuario u 
            INNER JOIN fotos_usuario f ON u.id = f.usuario_id
        """)
        usuarios = cursor.fetchall()
        
        new_cache = {}
        
        for user_id, nome, caminho in usuarios:
            img_path = os.path.join(os.getcwd(), caminho.lstrip('/'))
            
            if not os.path.exists(img_path):
                logger.warning(f"Arquivo não encontrado: {img_path}")
                continue
            
            try:
                # Carregar e processar imagem
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # Redimensionar para acelerar processamento
                height, width = image.shape[:2]
                if width > 800:
                    scale = 800 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    image = cv2.resize(image, (new_width, new_height))
                
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Extrair encoding com configurações otimizadas
                face_locations = face_recognition.face_locations(
                    rgb_image, 
                    model="hog",  # HOG é mais rápido que CNN
                    number_of_times_to_upsample=1  # Reduzir para acelerar
                )
                
                if face_locations:
                    encodings = face_recognition.face_encodings(
                        rgb_image, 
                        face_locations,
                        num_jitters=1,  # Reduzir para acelerar
                        model="small"   # Modelo menor e mais rápido
                    )
                    
                    if encodings:
                        new_cache[user_id] = {
                            'nome': nome,
                            'encoding': encodings[0],
                            'path': caminho
                        }
                        logger.info(f"Encoding carregado para {nome}")
                
            except Exception as e:
                logger.error(f"Erro ao processar {nome}: {e}")
                continue
        
        encodings_cache = new_cache
        last_cache_update = current_time
        
        load_time = time.time() - start_time
        logger.info(f"Cache atualizado com {len(new_cache)} usuários em {load_time:.2f}s")
        
        return encodings_cache
        
    finally:
        cursor.close()
        conn.close()

def extract_face_encoding_fast(image):
    """Versão otimizada para extrair encoding rapidamente"""
    try:
        start_time = time.time()
        
        # Converter para RGB se necessário
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Redimensionar para acelerar (máximo 640px de largura)
        height, width = rgb_image.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            rgb_image = cv2.resize(rgb_image, (new_width, new_height))
        
        # Detectar faces com configurações rápidas
        face_locations = face_recognition.face_locations(
            rgb_image,
            model="hog",  # Mais rápido
            number_of_times_to_upsample=1  # Menos upsampling = mais rápido
        )
        
        if not face_locations:
            return None, "Nenhum rosto detectado"
        
        # Pegar a maior face
        largest_face = max(face_locations, key=lambda face: (face[2] - face[0]) * (face[1] - face[3]))
        top, right, bottom, left = largest_face
        
        # Verificar tamanho mínimo
        face_width = right - left
        face_height = bottom - top
        
        if face_width < 30 or face_height < 30:
            return None, "Face muito pequena"
        
        # Extrair encoding com configurações rápidas
        encodings = face_recognition.face_encodings(
            rgb_image,
            [largest_face],
            num_jitters=1,  # Menos jitters = mais rápido
            model="small"   # Modelo menor = mais rápido
        )
        
        if not encodings:
            return None, "Não foi possível extrair características"
        
        processing_time = time.time() - start_time
        logger.info(f"Encoding extraído em {processing_time:.2f}s")
        
        return encodings[0], "Sucesso"
        
    except Exception as e:
        logger.error(f"Erro ao extrair encoding: {e}")
        return None, f"Erro: {str(e)}"

    cache_lock = Lock()
    
    # Tenta usar o modelo CNN se o HOG falhar
    def find_face_encodings(image):
        # Tenta primeiro com o HOG, que é mais rápido
        face_locations = face_recognition.face_locations(image, model='hog')
        
        # Se o HOG não encontrar rostos, tenta com o CNN (mais lento, porém mais preciso)
        if not face_locations:
            print("Modelo HOG falhou. Tentando com o modelo CNN...")
            face_locations = face_recognition.face_locations(image, model='cnn')
            
        # Se encontrou com algum dos modelos, extrai os encodings
        if face_locations:
            return face_recognition.face_encodings(image, known_face_locations=face_locations)
        
        # Se nenhum modelo encontrou, retorna lista vazia
        return []


def find_face_locations_robust(image):
    """Tenta detectar faces com HOG e faz fallback para CNN se falhar."""
    # Tenta com HOG primeiro por ser mais rápido
    face_locations = face_recognition.face_locations(image, model='hog')
    if not face_locations:
        logger.info("Detecção HOG falhou, tentando com CNN (mais lento)...")
        face_locations = face_recognition.face_locations(image, model='cnn')
    return face_locations


# Rota principal - serve o menu
@app.route('/')
def index():
    return render_template('index.html')


def enhance_face_image_for_save(image):
    """
    Processa a imagem para salvar, focando em normalização e qualidade.
    """
    try:
        # Converter para PIL se necessário
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image
        
        rgb_image = np.array(pil_image.convert('RGB'))
        
        # Usar a nova função robusta de detecção
        face_locations = find_face_locations_robust(rgb_image)
        
        if not face_locations:
            return None, False, "Nenhuma face detectada na imagem para salvar."
            
        # Pega a maior face
        top, right, bottom, left = max(face_locations, key=lambda x: (x[2]-x[0])*(x[1]-x[3]))
        
        # Cropar a face diretamente da imagem original
        face_image = pil_image.crop((left, top, right, bottom))
        
        # Redimensionar para um tamanho padrão para consistência
        face_image = face_image.resize((300, 300), Image.Resampling.LANCZOS)
        
        # Normalização de contraste com equalização de histograma
        gray_face = face_image.convert('L')
        equalized_face = ImageOps.equalize(gray_face)
        
        # Converter de volta para RGB para salvar como JPEG
        final_image = equalized_face.convert('RGB')
        
        return final_image, True, "Imagem processada e normalizada com sucesso."
        
    except Exception as e:
        logger.error(f"Erro em enhance_face_image_for_save: {e}")
        return None, False, f"Erro no processamento da imagem: {str(e)}"
        
        
@app.route('/cadastro', methods=['GET', 'POST'])
def cadastro():
    if request.method == 'POST':
        try:
            # Verificar se é uma requisição JSON (do PHP) ou form (do frontend)
            if request.is_json:
                data = request.get_json()
                nome = data.get('nome')
                cpf = data.get('cpf') or ''
                foto_url = data.get('foto_url')
                id_usuario_php = data.get('id_usuario_php')

                if not foto_url:
                    return jsonify({'success': False, 'message': 'Foto não fornecida'}), 400

                if not id_usuario_php:
                    return jsonify({'success': False, 'message': 'ID do usuário PHP não fornecido'}), 400

            else:
                nome = request.form.get('nome')
                cpf = request.form.get('cpf') or ''
                foto_url = request.form.get('foto_url')
                id_usuario_php = request.form.get('id_usuario_php')

                if not foto_url:
                    flash('Por favor, selecione um aluno com foto cadastrada.', 'error')
                    return redirect(url_for('cadastro'))

                if not id_usuario_php:
                    flash('ID do usuário PHP não fornecido.', 'error')
                    return redirect(url_for('cadastro'))

            # Baixar e processar a foto
            try:
                # Fazer o download da imagem do servidor PHP
                response = requests.get(foto_url, verify=False)  # verify=False para evitar problemas com SSL
                if response.status_code != 200:
                    if request.is_json:
                        return jsonify({'success': False, 'message': 'Erro ao baixar a foto'}), 400
                    else:
                        flash('Erro ao baixar a foto do aluno. Por favor, tente novamente.', 'error')
                        return redirect(url_for('cadastro'))

                # Converter a resposta em imagem
                image = Image.open(BytesIO(response.content))
                
                # Processar e extrair face
                face_image, success, message = enhance_face_image_for_save(image)
                
                if not success:
                    if request.is_json:
                        return jsonify({'success': False, 'message': message}), 400
                    else:
                        flash(f'Não foi possível processar a imagem do aluno: {message}', 'error')
                        return redirect(url_for('cadastro'))

                # Primeiro inserir o usuário
                conn = get_db_connection()
                cursor = conn.cursor()
                
                try:
                    # Iniciar transação
                    cursor.execute("START TRANSACTION")
                    
                    # Inserir usuário com id_usuario_php
                    cursor.execute("""
                        INSERT INTO usuario (nome, cpf, id_usuario_php)
                        VALUES (%s, %s, %s)
                    """, (nome, cpf, id_usuario_php))
                    
                    # Pegar o ID do usuário inserido
                    usuario_id = cursor.lastrowid

                    # Gerar nome único para o arquivo
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"{nome.replace(' ', '_')}_{timestamp}.jpg"
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    
                    # Salvar a imagem processada com o rosto
                    face_image.save(filepath, 'JPEG', quality=95)
                    
                    # Salvar o caminho no banco
                    db_filepath = f"/static/fotos/{filename}"
                    cursor.execute("""
                        INSERT INTO fotos_usuario (usuario_id, caminho)
                        VALUES (%s, %s)
                    """, (usuario_id, db_filepath))
                    
                    # Commit da transação
                    cursor.execute("COMMIT")

                    if request.is_json:
                        return jsonify({'success': True, 'message': 'Usuário cadastrado com sucesso!'})
                    else:
                        flash('Usuário cadastrado com sucesso!', 'success')
                        return redirect(url_for('cadastro'))
                    
                except Exception as e:
                    # Rollback em caso de erro
                    cursor.execute("ROLLBACK")
                    if os.path.exists(filepath):
                        os.remove(filepath)  # Remove o arquivo se foi criado
                    raise e

            except requests.exceptions.RequestException as e:
                if request.is_json:
                    return jsonify({'success': False, 'message': f'Erro ao baixar a foto: {str(e)}'}), 500
                else:
                    flash(f'Erro ao baixar a foto do aluno: {str(e)}', 'error')
                    return redirect(url_for('cadastro'))
            except Exception as e:
                if request.is_json:
                    return jsonify({'success': False, 'message': f'Erro ao processar a foto: {str(e)}'}), 500
                else:
                    flash(f'Erro ao processar a foto: {str(e)}', 'error')
                    return redirect(url_for('cadastro'))
            
        except Exception as e:
            if request.is_json:
                return jsonify({'success': False, 'message': f'Erro ao cadastrar usuário: {str(e)}'}), 500
            else:
                flash(f'Erro ao cadastrar usuário: {str(e)}', 'error')
                return redirect(url_for('cadastro'))
            
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
    
    return render_template('cadastro.html')
    
    
@app.route('/api/alunos_php')
def get_alunos_php():
    """
    Obter Alunos sem Foto Registrada
    Retorna alunos da API PHP externa que ainda não têm biometria cadastrada.
    ---
    tags:
      - Integração Externa
    responses:
      200:
        description: Lista de alunos.
        schema:
          type: array
          items:
            type: object
            properties:
              id:
                type: integer
              nome:
                type: string
      500:
        description: Erro ao consultar a API externa.
    """
    try:
        # 1. Obter todos os alunos da API PHP
        url_php = "https://apppanorama.jsatecsistemas.com.br/api_alunos_para_reconhecimento.php"
        response = requests.get(url_php, timeout=15, verify=False)
        response.raise_for_status()
        alunos_php = response.json()

        # 2. Obter IDs dos usuários que já possuem foto no banco de dados local
        ids_com_foto = set()
        conn = get_db_connection()
        if conn:
            try:
                cursor = conn.cursor()
                # Seleciona os IDs de usuário do PHP que estão na tabela de usuários e têm uma foto associada
                cursor.execute("""
                    SELECT u.id_usuario_php 
                    FROM usuario u
                    INNER JOIN fotos_usuario f ON u.id = f.usuario_id
                    WHERE u.id_usuario_php IS NOT NULL
                """)
                # Adiciona os IDs a um conjunto para uma verificação rápida
                ids_com_foto = {row[0] for row in cursor.fetchall()}
            except mysql.connector.Error as db_error:
                logger.error(f"Erro ao consultar o banco de dados local: {db_error}")
            finally:
                if 'cursor' in locals() and cursor:
                    cursor.close()
                conn.close()

        # 3. Filtrar a lista de alunos para remover aqueles que já têm foto
        # O ID do aluno do PHP é um inteiro, então garantimos a conversão para comparação.
        alunos_sem_foto = [
            aluno for aluno in alunos_php 
            if aluno.get('id') and int(aluno.get('id')) not in ids_com_foto
        ]
        
        # 4. Retornar a lista filtrada como JSON
        return jsonify(alunos_sem_foto)

    except requests.exceptions.RequestException as e:
        # Captura erros de conexão, timeout, etc.
        logger.error(f"!!! ERRO [RequestException] ao chamar API PHP: {e}")
        return jsonify({'success': False, 'message': f'Erro ao acessar a API do PHP: {str(e)}'}), 500
    except ValueError as e:
        # Captura erros de decodificação do JSON (ex: se o PHP retornar um erro HTML)
        logger.error(f"!!! ERRO [ValueError] de JSON na API PHP: {e}")
        return jsonify({'success': False, 'message': f'A API do PHP não retornou um JSON válido.'}), 500
    except Exception as e:
        logger.error(f"!!! ERRO [Geral] na rota /api/alunos_php: {e}")
        return jsonify({'success': False, 'message': f'Ocorreu um erro inesperado: {str(e)}'}), 500


# Rota para captura de fotos
@app.route('/captura_fotos')
def captura_fotos():
    return render_template('captura_fotos.html')

# Rota para salvar foto
@app.route('/salvar_foto', methods=['POST'])
def salvar_foto():
    """
    Salvar Biometria de Aluno
    Recebe os dados do usuário e sua foto em Base64, extrai o rosto e cadastra no banco.
    ---
    tags:
      - Cadastro de Biometria
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            usuario_id_php:
              type: integer
              description: ID do usuário na API PHP externa
            nome:
              type: string
              description: Nome completo
            cpf:
              type: string
              description: CPF (opcional)
            image:
              type: string
              description: Foto em formato Base64
    responses:
      200:
        description: Biometria cadastrada com sucesso.
      400:
        description: Erro nos dados enviados (ex. dados faltando, qualidade da foto ruim).
      500:
        description: Erro interno do servidor ou de banco de dados.
    """
    if request.method == 'POST':
        try:
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)

            data = request.json
            usuario_id_php = data.get('usuario_id_php')
            nome_usuario = data.get('nome')
            cpf_usuario = data.get('cpf') or ''
            foto_base64 = data.get('image')

            if not all([usuario_id_php, nome_usuario, foto_base64]):
                return jsonify({'success': False, 'message': 'Dados incompletos (ID PHP, Nome e Foto são obrigatórios)'}), 400

            if ',' in foto_base64:
                foto_base64 = foto_base64.split(',')[1]

            try:
                foto_bytes = base64.b64decode(foto_base64)
            except Exception as e:
                return jsonify({'success': False, 'message': f'Erro ao decodificar imagem base64: {str(e)}'}), 400

            conn = get_db_connection()
            cursor = conn.cursor()
            
            usuario_id_local = None
            try:
                cursor.execute("SELECT id FROM usuario WHERE id_usuario_php = %s", (usuario_id_php,))
                result = cursor.fetchone()
                
                if result:
                    usuario_id_local = result[0]
                else:
                    cursor.execute(
                        "INSERT INTO usuario (nome, cpf, id_usuario_php) VALUES (%s, %s, %s)",
                        (nome_usuario, cpf_usuario, usuario_id_php)
                    )
                    usuario_id_local = cursor.lastrowid

                image = Image.open(BytesIO(foto_bytes))
                face_image, success, message = enhance_face_image_for_save(image)
                
                if not success:
                    return jsonify({'success': False, 'message': message}), 400
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{nome_usuario.replace(' ', '_')}_{timestamp}.jpg"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                
                face_image.save(filepath, 'JPEG', quality=95)
                
                db_filepath = f"/static/fotos/{filename}"
                
                cursor.execute("SELECT id FROM fotos_usuario WHERE usuario_id = %s", (usuario_id_local,))
                existing_photo = cursor.fetchone()
                
                if existing_photo:
                    cursor.execute(
                        "UPDATE fotos_usuario SET caminho = %s, data_captura = CURRENT_TIMESTAMP WHERE usuario_id = %s",
                        (db_filepath, usuario_id_local)
                    )
                else:
                    cursor.execute(
                        "INSERT INTO fotos_usuario (usuario_id, caminho) VALUES (%s, %s)",
                        (usuario_id_local, db_filepath)
                    )
                
                conn.commit()
                
                return jsonify({
                    'success': True, 
                    'message': 'Foto salva com sucesso!',
                    'filepath': db_filepath
                })
                
            except Exception as e:
                if conn:
                    conn.rollback()
                return jsonify({'success': False, 'message': f'Erro no banco de dados ou processamento: {str(e)}'}), 500

        except Exception as e:
            return jsonify({'success': False, 'message': f'Erro geral no servidor: {str(e)}'}), 500

        finally:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'conn' in locals() and conn:
                conn.close()
            
            
            

# Rota adicional para verificar qualidade da foto antes de salvar
@app.route('/verificar_qualidade_foto', methods=['POST'])
def verificar_qualidade_foto():
    """
    Verificar Qualidade de Foto
    Verifica se a foto enviada é elegível para extração de rosto e biometria.
    ---
    tags:
      - Cadastro de Biometria
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            image:
              type: string
              description: Imagem em Base64
    responses:
      200:
        description: Retorna se a qualidade da foto está adequada ou inadequada.
    """
    try:
        data = request.json
        foto_base64 = data.get('image')
        
        if not foto_base64:
            return jsonify({'success': False, 'message': 'Nenhuma imagem fornecida'})
        
        # Remover cabeçalho
        if ',' in foto_base64:
            foto_base64 = foto_base64.split(',')[1]
        
        # Decodificar
        foto_bytes = base64.b64decode(foto_base64)
        image = Image.open(BytesIO(foto_bytes))
        
        # Verificar qualidade
        processed_image, success, message = enhance_face_image_for_save(image)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Foto adequada para processamento',
                'quality': 'Boa',
                'details': message
            })
        else:
            return jsonify({
                'success': False,
                'message': message,
                'quality': 'Inadequada',
                'suggestions': [
                    'Melhore a iluminação',
                    'Posicione o rosto centralmente',
                    'Aproxime-se mais da câmera',
                    'Evite sombras no rosto'
                ]
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Erro na verificação: {str(e)}'
        })

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    """
    Login via Reconhecimento Facial
    Versão otimizada do processamento de imagem
    ---
    tags:
      - Autenticação Facial
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            image:
              type: string
              description: Imagem do usuário em Base64
            threshold:
              type: number
              description: Grau de tolerância para o reconhecimento (padrão 0.5)
              example: 0.5
    responses:
      200:
        description: Usuário Autenticado com Sucesso.
      400:
        description: Nenhum rosto encontrado, ou base64 inválido.
      500:
        description: Erro interno do servidor.
    """
    start_time = time.time()
    
    try:
        # Validar dados
        if not request.json or 'image' not in request.json:
            return jsonify({
                'success': False,
                'message': 'Nenhuma imagem fornecida'
            }), 400
        
        # Decodificar imagem
        image_data = request.json['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        try:
            # Converte a imagem de base64 para um array numpy
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            logger.error(f"Erro ao decodificar imagem: {e}")
            return jsonify({
                'success': False,
                'message': 'Formato de imagem inválido ou dados corrompidos.'
            }), 400

        # Pré-processamento: Aplica realces de contraste e nitidez na imagem do login
        # para que ela se pareça mais com a foto de referência salva.
        enhancer_contrast = ImageEnhance.Contrast(image)
        image = enhancer_contrast.enhance(1.4)
        enhancer_sharpness = ImageEnhance.Sharpness(image)
        image = enhancer_sharpness.enhance(1.3)
        
        np_image = np.array(image.convert('RGB'))

        # Encontra os encodings na imagem recebida usando a nova função
        face_encodings_in_image = find_face_encodings(np_image)
        
        if not face_encodings_in_image:
            return jsonify({
                'success': False,
                'message': 'Nenhum rosto detectado na imagem.',
                'processing_time': round(time.time() - start_time, 2)
            })
        
        # Carregar cache de encodings
        cache = load_encodings_cache()
        
        if not cache:
            return jsonify({
                'success': False,
                'message': 'Nenhum usuário cadastrado ou erro no cache',
                'processing_time': round(time.time() - start_time, 2)
            })
        
        # Comparar com encodings do cache
        best_match = None
        best_distance = 1.0
        best_user_id = None
        
        comparison_start = time.time()
        
        for user_id, user_data in cache.items():
            try:
                # Calcular distância
                distance = face_recognition.face_distance(
                    [user_data['encoding']], 
                    face_encodings_in_image[0] # Usar o primeiro encoding encontrado
                )[0]
                
                logger.info(f"Distância para {user_data['nome']}: {distance:.3f}")
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = user_data['nome']
                    best_user_id = user_id
                    
            except Exception as e:
                logger.error(f"Erro ao comparar com {user_data['nome']}: {e}")
                continue
        
        comparison_time = time.time() - comparison_start
        logger.info(f"Comparação concluída em {comparison_time:.2f}s")
        
        # Threshold mais permissivo para melhor reconhecimento
        threshold = float(request.json.get('threshold', 0.5))  # Aumentado de 0.45 para 0.5
        
        total_time = time.time() - start_time
        
        if best_match and best_distance <= threshold: # Changed from < to <=
            # Registrar login
            try:
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO login (usuario_id, data_login, hora_login)
                        VALUES (%s, CURDATE(), CURTIME())
                    """, (best_user_id,))
                    conn.commit()
                    cursor.close()
                    conn.close()
            except Exception as e:
                logger.error(f"Erro ao registrar login: {e}")
            
            return jsonify({
                'success': True,
                'name': best_match,
                'message': f'Bem-vindo, {best_match}!',
                'confidence': round((1 - best_distance) * 100, 1),
                'distance': round(best_distance, 3),
                'processing_time': round(total_time, 2),
                'users_checked': len(cache)
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Usuário não reconhecido (melhor match: {round(best_distance, 3)})',
                'debug': {
                    'best_distance': round(best_distance, 3),
                    'threshold': threshold,
                    'users_checked': len(cache),
                    'processing_time': round(total_time, 2),
                    'best_match_name': best_match if best_match else 'Nenhum'
                }
            })
            
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(f"Erro geral no processamento: {e}")
        return jsonify({
            'success': False,
            'message': f'Erro interno: {str(e)}',
            'processing_time': round(total_time, 2)
        }), 500

@app.route('/update_cache', methods=['POST'])
def update_cache():
    """Força atualização do cache"""
    global last_cache_update
    last_cache_update = 0  # Força atualização
    cache = load_encodings_cache()
    
    return jsonify({
        'success': True,
        'message': f'Cache atualizado com {len(cache)} usuários',
        'users': list(cache.values())
    })

@app.route('/health')
def health_check():
    """
    Verificação de Integridade da API
    ---
    tags:
      - Monitoramento
    responses:
      200:
        description: Retorna o status da API, Banco e Cache.
        schema:
          type: object
          properties:
            status:
              type: string
              example: OK
            database:
              type: string
              example: Connected
            users_in_db:
              type: integer
              example: 15
      500:
        description: Erro de conexão com o banco ou cache.
    """
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM usuario")
            user_count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            cache_info = {
                'cached_users': len(encodings_cache),
                'last_update': datetime.fromtimestamp(last_cache_update).isoformat() if last_cache_update else 'Never'
            }
            
            return jsonify({
                'status': 'OK',
                'database': 'Connected',
                'users_in_db': user_count,
                'cache': cache_info
            })
        else:
            return jsonify({
                'status': 'OK',
                'database': 'Error'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'Error',
            'message': str(e)
        }), 500


@app.route('/exportar_login')
def exportar_login():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Buscar todos os registros de login com nome do usuário
        cursor.execute("""
            SELECT u.nome, l.data_login, l.hora_login 
            FROM login l
            JOIN usuario u ON l.usuario_id = u.id
            ORDER BY l.data_login DESC, l.hora_login DESC
        """)
        
        registros = cursor.fetchall()
        
        # Criar arquivo CSV em memória
        si = StringIO()
        cw = csv.writer(si, delimiter=';')
        
        # Escrever cabeçalho
        cw.writerow(['Nome', 'Data', 'Hora'])
        
        # Escrever registros
        for registro in registros:
            cw.writerow(registro)
        
        # Preparar o arquivo para download
        output = si.getvalue()
        si.close()
        
        # Gerar nome do arquivo com data atual
        data_atual = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"registros_login_{data_atual}.csv"
        
        # Retornar o arquivo CSV
        return Response(
            output,
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename={filename}',
                'Content-Type': 'text/csv'
            }
        )
        
    except Exception as e:
        flash(f'Erro ao exportar registros: {str(e)}', 'error')
        return redirect(url_for('index'))
        
    finally:
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == '__main__':
    # Remover validação SSL caso consuma APIs externas
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    app.run(host='0.0.0.0', port=8090, debug=True)
