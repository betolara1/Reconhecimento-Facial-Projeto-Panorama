from flask import Flask, render_template, request, jsonify, Response, flash, redirect, url_for
from flask_cors import CORS
import cv2
import face_recognition
import numpy as np
import base64
from datetime import datetime
import os
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO, StringIO
import mysql.connector
from dotenv import load_dotenv
import time
import logging
import csv

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

@app.route('/')
def index():
    return render_template('index.html')


def enhance_face_image_for_save(image):
    """
    Aplica o mesmo processamento avançado do reprocess_database_photos.py
    para novas fotos sendo salvas
    """
    try:
        # Converter para PIL se necessário
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Converter para numpy para detecção de face
        rgb_image = np.array(pil_image)
        
        # Detectar faces com alta precisão
        face_locations = face_recognition.face_locations(rgb_image, model="hog", number_of_times_to_upsample=2)
        
        if not face_locations:
            return None, False, "Nenhuma face detectada"
        
        # Pegar a maior face
        top, right, bottom, left = max(face_locations, key=lambda x: (x[2]-x[0])*(x[1]-x[3]))
        
        # Verificar se a face tem tamanho adequado
        face_width = right - left
        face_height = bottom - top
        
        if face_width < 50 or face_height < 50:
            return None, False, "Face muito pequena para processamento"
        
        # Expandir a região da face (margem de 30% para capturar mais contexto)
        height, width = rgb_image.shape[:2]
        
        margin_h = int(face_height * 0.3)
        margin_w = int(face_width * 0.3)
        
        top = max(0, top - margin_h)
        bottom = min(height, bottom + margin_h)
        left = max(0, left - margin_w)
        right = min(width, right + margin_w)
        
        # Extrair região da face expandida
        face_image = pil_image.crop((left, top, right, bottom))
        
        # Redimensionar para tamanho padrão (300x300 para consistência)
        face_image = face_image.resize((300, 300), Image.Resampling.LANCZOS)
        
        # === MELHORIAS DE QUALIDADE ===
        
        # 1. Melhorar contraste
        enhancer = ImageEnhance.Contrast(face_image)
        face_image = enhancer.enhance(1.4)
        
        # 2. Melhorar brilho
        enhancer = ImageEnhance.Brightness(face_image)
        face_image = enhancer.enhance(1.1)
        
        # 3. Melhorar nitidez
        enhancer = ImageEnhance.Sharpness(face_image)
        face_image = enhancer.enhance(1.3)
        
        # 4. Aplicar filtro de nitidez avançado
        face_image = face_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
        # === NORMALIZAÇÃO DE ILUMINAÇÃO ===
        
        # Converter de volta para OpenCV para aplicar CLAHE
        cv_face = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
        
        # Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(cv_face, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Converter de volta para PIL para retorno
        final_image = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        
        # Verificar se ainda conseguimos detectar face na imagem processada
        test_rgb = np.array(final_image)
        test_locations = face_recognition.face_locations(test_rgb)
        
        if not test_locations:
            return None, False, "Face perdida durante processamento"
        
        # Testar se conseguimos extrair encoding
        try:
            encodings = face_recognition.face_encodings(test_rgb, test_locations)
            if not encodings:
                return None, False, "Não foi possível extrair características faciais"
        except:
            return None, False, "Erro ao extrair características faciais"
        
        return final_image, True, "Processamento concluído com sucesso"
        
    except Exception as e:
        return None, False, f"Erro no processamento: {str(e)}"

# Rota para captura de fotos
@app.route('/captura_fotos')
def captura_fotos():
    try:
        conn = get_db_connection()
        if not conn:
            flash('Erro de conexão com banco de dados', 'error')
            return redirect(url_for('index'))
        
        cursor = conn.cursor()
        
        # Buscar usuários sem fotos
        cursor.execute("""
            SELECT u.id, u.nome 
            FROM usuario u 
            LEFT JOIN fotos_usuario f ON u.id = f.usuario_id 
            WHERE f.id IS NULL
        """)
        
        usuarios = cursor.fetchall()
        
        return render_template('captura_fotos.html', usuarios=usuarios)
        
    except Exception as e:
        flash(f'Erro ao carregar usuários: {str(e)}', 'error')
        return redirect(url_for('index'))
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# Rota para salvar foto com processamento avançado
@app.route('/salvar_foto', methods=['POST'])
def salvar_foto():
    """
    Salva foto aplicando processamento avançado automaticamente
    """
    try:
        # Verificar se o diretório existe
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER)
        
        # Obter dados da requisição
        data = request.json if request.is_json else request.form
        usuario_id = data.get('usuario_id')
        foto_base64 = data.get('image') or data.get('foto')
        
        # Validar dados
        if not usuario_id or not foto_base64:
            return jsonify({
                'success': False, 
                'message': 'Dados incompletos - usuário e foto são obrigatórios'
            }), 400
        
        # Remover cabeçalho base64
        if ',' in foto_base64:
            foto_base64 = foto_base64.split(',')[1]
        
        # Decodificar imagem
        try:
            foto_bytes = base64.b64decode(foto_base64)
            original_image = Image.open(BytesIO(foto_bytes))
        except Exception as e:
            return jsonify({
                'success': False, 
                'message': f'Erro ao decodificar imagem: {str(e)}'
            }), 400
        
        # Conectar ao banco
        conn = get_db_connection()
        if not conn:
            return jsonify({
                'success': False, 
                'message': 'Erro de conexão com banco de dados'
            }), 500
        
        cursor = conn.cursor()
        
        # Buscar usuário
        try:
            cursor.execute("SELECT nome FROM usuario WHERE id = %s", (usuario_id,))
            result = cursor.fetchone()
            
            if not result:
                return jsonify({
                    'success': False, 
                    'message': 'Usuário não encontrado'
                }), 404
            
            nome_usuario = result[0]
            
        except Exception as e:
            return jsonify({
                'success': False, 
                'message': f'Erro ao buscar usuário: {str(e)}'
            }), 500
        
        # === PROCESSAMENTO AVANÇADO DA IMAGEM ===
        print(f"🔍 Processando foto de {nome_usuario} com algoritmo avançado...")
        
        processed_image, success, message = enhance_face_image_for_save(original_image)
        
        if not success:
            return jsonify({
                'success': False,
                'message': f'Erro no processamento da face: {message}',
                'details': 'Certifique-se de que há boa iluminação e que seu rosto está bem visível'
            }), 400
        
        print(f"✅ Processamento concluído: {message}")
        
        # Gerar nome único para o arquivo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{nome_usuario.replace(' ', '_')}_{timestamp}_processed.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # Salvar imagem processada com alta qualidade
        try:
            processed_image.save(filepath, 'JPEG', quality=95, optimize=True)
            print(f"💾 Imagem salva: {filepath}")
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Erro ao salvar imagem: {str(e)}'
            }), 500
        
        # Salvar no banco de dados
        db_filepath = f"/static/fotos/{filename}"
        
        try:
            # Verificar se já existe foto para este usuário
            cursor.execute("SELECT id FROM fotos_usuario WHERE usuario_id = %s", (usuario_id,))
            existing_photo = cursor.fetchone()
            
            if existing_photo:
                # Atualizar registro existente
                cursor.execute("""
                    UPDATE fotos_usuario 
                    SET caminho = %s, data_captura = NOW() 
                    WHERE usuario_id = %s
                """, (db_filepath, usuario_id))
                print(f"🔄 Foto atualizada no banco para {nome_usuario}")
            else:
                # Inserir novo registro
                cursor.execute("""
                    INSERT INTO fotos_usuario (usuario_id, caminho, data_captura)
                    VALUES (%s, %s, NOW())
                """, (usuario_id, db_filepath))
                print(f"➕ Nova foto inserida no banco para {nome_usuario}")
            
            conn.commit()
            
            # Verificar se a foto foi salva corretamente testando encoding
            try:
                test_image = cv2.imread(filepath)
                test_rgb = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
                test_locations = face_recognition.face_locations(test_rgb)
                test_encodings = face_recognition.face_encodings(test_rgb, test_locations)
                
                encoding_quality = "✅ Excelente" if test_encodings else "⚠️ Limitada"
                print(f"🎯 Qualidade do encoding: {encoding_quality}")
                
            except Exception as e:
                print(f"⚠️ Erro no teste de encoding: {e}")
            
            return jsonify({
                'success': True,
                'message': f'Foto de {nome_usuario} processada e salva com sucesso!',
                'details': {
                    'processing': message,
                    'filename': filename,
                    'filepath': db_filepath,
                    'size': '300x300px (otimizado)',
                    'quality': 'Alta precisão para reconhecimento'
                }
            })
            
        except Exception as e:
            conn.rollback()
            # Remover arquivo se houve erro no banco
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({
                'success': False,
                'message': f'Erro ao salvar no banco: {str(e)}'
            }), 500
        
    except Exception as e:
        print(f"❌ Erro geral: {e}")
        return jsonify({
            'success': False,
            'message': f'Erro interno: {str(e)}'
        }), 500
        
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

# Rota adicional para verificar qualidade da foto antes de salvar
@app.route('/verificar_qualidade_foto', methods=['POST'])
def verificar_qualidade_foto():
    """
    Verifica a qualidade da foto antes de salvar (opcional)
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
    """Versão otimizada do processamento de imagem"""
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
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Erro ao decodificar imagem: {str(e)}'
            }), 400
        
        decode_time = time.time() - start_time
        logger.info(f"Imagem decodificada em {decode_time:.2f}s")
        
        # Extrair encoding da imagem capturada
        captured_encoding, message = extract_face_encoding_fast(cv_image)
        
        if captured_encoding is None:
            return jsonify({
                'success': False,
                'message': message,
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
                    captured_encoding
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
        
        if best_match and best_distance < threshold:
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
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()