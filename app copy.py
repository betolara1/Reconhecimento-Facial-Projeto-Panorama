from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify
from flask_cors import CORS
import cv2
import face_recognition
import numpy as np
from db import get_db_connection
import base64
from datetime import datetime
import os
from PIL import Image
from io import BytesIO
import time
from threading import Timer
import csv
from io import StringIO
import requests

app = Flask(__name__)
CORS(app)
app.secret_key = 'sua_chave_secreta_muito_longa_e_aleatoria'


# Variável global para armazenar o último usuário reconhecido e as imagens
last_recognized_user = None
captured_image = None
matched_image = None

# Variáveis globais para controle do reconhecimento
face_detection_start_time = None
face_detected = False
recognition_complete = False
recognition_result = None

# Definir o diretório de fotos
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'fotos')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def process_face_image(image_data, mode='PIL'):
    """
    Processa uma imagem para extrair e normalizar a face.
    
    Args:
        image_data: Pode ser um objeto PIL.Image ou um array numpy
        mode: 'PIL' se a entrada é uma imagem PIL, 'CV2' se é um array numpy/opencv
    
    Returns:
        face_image: Imagem PIL contendo apenas o rosto normalizado
        success: Boolean indicando se uma face foi encontrada e processada
    """
    try:
        # Converter para formato adequado para face_recognition
        if mode == 'PIL':
            if image_data.mode != 'RGB':
                image_data = image_data.convert('RGB')
            np_image = np.array(image_data)
        else:  # mode == 'CV2'
            np_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        
        # Melhorar a qualidade da imagem
        # Normalizar o brilho e contraste
        np_image = cv2.normalize(np_image, None, 0, 255, cv2.NORM_MINMAX)
        
        # Detectar faces na imagem usando HOG (mais preciso que o padrão)
        face_locations = face_recognition.face_locations(np_image, model="hog", number_of_times_to_upsample=2)
        
        if not face_locations:
            return None, False
        
        # Pegar a primeira face encontrada
        top, right, bottom, left = face_locations[0]
        
        # Adicionar margem ao redor do rosto (15% de cada lado - reduzido para maior precisão)
        height = bottom - top
        width = right - left
        margin_v = int(height * 0.15)
        margin_h = int(width * 0.15)
        
        # Ajustar as coordenadas com a margem
        top = max(0, top - margin_v)
        bottom = min(np_image.shape[0], bottom + margin_v)
        left = max(0, left - margin_h)
        right = min(np_image.shape[1], right + margin_h)
        
        # Recortar a região do rosto
        face_image = np_image[top:bottom, left:right]
        
        # Converter de volta para PIL e redimensionar
        face_pil = Image.fromarray(face_image)
        face_pil.thumbnail((200, 200), Image.Resampling.LANCZOS)
        
        return face_pil, True
        
    except Exception as e:
        print(f"Erro ao processar face: {str(e)}")
        return None, False

# Rota principal
@app.route('/')
def index():
    return render_template('index.html')

# Rota para API de cadastro
@app.route('/cadastro', methods=['GET', 'POST'])
def cadastro():
    if request.method == 'POST':
        try:
            nome = request.form.get('nome')
            cpf = request.form.get('cpf')
            foto_url = request.form.get('foto_url')

            if not foto_url:
                flash('Por favor, selecione um aluno com foto cadastrada.', 'error')
                return redirect(url_for('cadastro'))

            # Baixar e processar a foto
            try:
                # Fazer o download da imagem do servidor PHP
                response = requests.get(foto_url, verify=False)  # verify=False para evitar problemas com SSL
                if response.status_code != 200:
                    flash('Erro ao baixar a foto do aluno. Por favor, tente novamente.', 'error')
                    return redirect(url_for('cadastro'))

                # Converter a resposta em imagem
                image = Image.open(BytesIO(response.content))
                
                # Processar e extrair face
                face_image, success = process_face_image(image, mode='PIL')
                
                if not success:
                    flash('Nenhum rosto detectado na imagem do aluno. Por favor, contate o administrador.', 'error')
                    return redirect(url_for('cadastro'))

                # Primeiro inserir o usuário
                conn = get_db_connection()
                cursor = conn.cursor()
                
                try:
                    # Iniciar transação
                    cursor.execute("START TRANSACTION")
                    
                    # Inserir usuário
                    cursor.execute("""
                        INSERT INTO usuario (nome, cpf)
                        VALUES (%s, %s)
                    """, (nome, cpf))
                    
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
                    flash('Usuário cadastrado com sucesso!', 'success')
                    return redirect(url_for('cadastro'))
                    
                except Exception as e:
                    # Rollback em caso de erro
                    cursor.execute("ROLLBACK")
                    if os.path.exists(filepath):
                        os.remove(filepath)  # Remove o arquivo se foi criado
                    raise e

            except requests.exceptions.RequestException as e:
                flash(f'Erro ao baixar a foto do aluno: {str(e)}', 'error')
                return redirect(url_for('cadastro'))
            except Exception as e:
                flash(f'Erro ao processar a foto: {str(e)}', 'error')
                return redirect(url_for('cadastro'))
            
        except Exception as e:
            flash(f'Erro ao cadastrar usuário: {str(e)}', 'error')
            return redirect(url_for('cadastro')) 
            
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
    
    return render_template('cadastro.html')

# Rota para captura de fotos
@app.route('/captura_fotos')
def captura_fotos():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Modificar a consulta para retornar apenas usuários sem fotos
    cursor.execute("""
        SELECT u.id, u.nome 
        FROM usuario u 
        LEFT JOIN fotos_usuario f ON u.id = f.usuario_id 
        WHERE f.id IS NULL
    """)
    
    usuarios = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('captura_fotos.html', usuarios=usuarios)

# Rota para salvar foto
@app.route('/salvar_foto', methods=['POST'])
def salvar_foto():
    if request.method == 'POST':
        try:
            # Verificar se o diretório existe, se não, criar
            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)

            data = request.json if request.is_json else request.form
            usuario_id = data.get('usuario_id')
            foto_base64 = data.get('image') or data.get('foto')

            # Debug
            print("Usuario ID:", usuario_id)
            print("Foto recebida:", bool(foto_base64))

            if not usuario_id or not foto_base64:
                return jsonify({'success': False, 'message': 'Dados incompletos'}), 400

            # Remover o cabeçalho da string base64 se existir
            if ',' in foto_base64:
                foto_base64 = foto_base64.split(',')[1]

            try:
                foto_bytes = base64.b64decode(foto_base64)
            except Exception as e:
                return jsonify({'success': False, 'message': f'Erro ao decodificar imagem: {str(e)}'}), 400

            # Conectar ao banco e buscar usuário
            conn = get_db_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute("SELECT nome FROM usuario WHERE id = %s", (usuario_id,))
                result = cursor.fetchone()
                if not result:
                    return jsonify({'success': False, 'message': 'Usuário não encontrado'}), 404
                nome_usuario = result[0]
            except Exception as e:
                return jsonify({'success': False, 'message': f'Erro ao buscar usuário: {str(e)}'}), 500

            try:
                # Processar imagem
                image = Image.open(BytesIO(foto_bytes))
                
                # Processar e extrair face
                face_image, success = process_face_image(image, mode='PIL')
                
                if not success:
                    return jsonify({'success': False, 'message': 'Nenhum rosto detectado na imagem'}), 400
                
                # Gerar nome único para o arquivo
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{nome_usuario.replace(' ', '_')}_{timestamp}.jpg"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                
                # Salvar imagem
                face_image.save(filepath, 'JPEG', quality=95)
                
                # Verificar se já existe uma foto para este usuário
                cursor.execute("SELECT * FROM fotos_usuario WHERE usuario_id = %s", (usuario_id,))
                existing_photo = cursor.fetchone()
                
                db_filepath = f"/static/fotos/{filename}"
                
                if existing_photo:
                    # Atualizar o registro existente
                    cursor.execute("""
                        UPDATE fotos_usuario 
                        SET caminho = %s, data_cadastro = CURDATE() 
                        WHERE usuario_id = %s
                    """, (db_filepath, usuario_id))
                else:
                    # Inserir novo registro
                    cursor.execute("""
                        INSERT INTO fotos_usuario (usuario_id, caminho)
                        VALUES (%s, %s)
                    """, (usuario_id, db_filepath))
                
                conn.commit()
                
                return jsonify({
                    'success': True, 
                    'message': 'Foto salva com sucesso!',
                    'filepath': db_filepath
                })
                
            except Exception as e:
                if 'conn' in locals():
                    conn.rollback()
                return jsonify({'success': False, 'message': f'Erro ao processar imagem: {str(e)}'}), 500

        except Exception as e:
            return jsonify({'success': False, 'message': f'Erro geral: {str(e)}'}), 500

        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

# Rota para login facial
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/save_login', methods=['POST'])
def save_login():
    data = request.json
    usuario_id = data.get('usuario_id')
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Inserir registro de login
        cursor.execute("""
            INSERT INTO login (usuario_id, data_login, hora_login) 
            VALUES (%s, CURDATE(), CURTIME())
        """, (usuario_id,))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def gerar_frames():
    global face_detection_start_time, face_detected, recognition_complete, recognition_result
    
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if not face_locations:
            face_detected = False
            face_detection_start_time = None
            recognition_complete = False
        
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            if not face_detected:
                face_detected = True
                face_detection_start_time = time.time()
                recognition_complete = False
            
            if face_detection_start_time and not recognition_complete:
                elapsed_time = time.time() - face_detection_start_time
                remaining_time = max(3 - int(elapsed_time), 0)
                
                cv2.putText(frame, f"Aguarde: {remaining_time}s", 
                            (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.75, (0, 255, 0), 2)
                
                if elapsed_time >= 3 and not recognition_complete:
                    face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                    
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT u.id, u.nome, f.caminho FROM usuario u INNER JOIN fotos_usuario f ON u.id = f.usuario_id")
                    usuarios = cursor.fetchall()
                    cursor.close()
                    conn.close()
                    
                    best_match = None
                    best_distance = 1.0
                    best_user_id = None
                    
                    for usuario in usuarios:
                        img_path = os.path.join(os.getcwd(), usuario[2].lstrip('/'))
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        
                        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        db_face_encodings = face_recognition.face_encodings(rgb_img)
                        
                        if not db_face_encodings:
                            continue
                        
                        db_face_encoding = db_face_encodings[0]
                        
                        distance = face_recognition.face_distance([db_face_encoding], face_encoding)[0]
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_match = usuario[1]
                            best_user_id = usuario[0]
                    
                    if best_match and best_distance < 0.6:
                        recognition_result = {
                            'status': 'success',
                            'message': f'Bem vindo {best_match}!',
                            'usuario_id': best_user_id
                        }
                    else:
                        recognition_result = {
                            'status': 'error',
                            'message': 'Usuário não cadastrado'
                        }
                    
                    recognition_complete = True
                    Timer(2.0, reset_recognition_vars).start()
        
        if recognition_result:
            cv2.putText(frame, recognition_result['message'], 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    camera.release()

def reset_recognition_vars():
    global face_detection_start_time, face_detected, recognition_complete, recognition_result
    face_detection_start_time = None
    face_detected = False
    recognition_complete = False
    recognition_result = None

@app.route('/video_feed')
def video_feed():
    return Response(gerar_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_and_compare', methods=['POST'])
def capture_and_compare():
    global last_recognized_user, captured_image, matched_image
    
    # Capturar frame da câmera
    camera = cv2.VideoCapture(-1)
    ret, frame = camera.read()
    camera.release()
    
    if not ret:
        return jsonify({'success': False, 'message': 'Erro ao capturar imagem'})
    
    # Processar imagem capturada
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    
    if not face_locations:
        return jsonify({'success': False, 'message': 'Nenhum rosto detectado'})
    
    face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
    
    # Salvar imagem capturada para exibição
    _, buffer = cv2.imencode('.jpg', frame)
    captured_image = base64.b64encode(buffer).decode('utf-8')
    
    # Buscar faces no banco de dados
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT u.id, u.nome, f.caminho FROM usuario u INNER JOIN fotos_usuario f ON u.id = f.usuario_id")
    usuarios = cursor.fetchall()
    cursor.close()
    conn.close()
    
    best_match = None
    best_distance = 1.0
    
    for usuario in usuarios:
        # Carregar imagem do arquivo
        img_path = os.path.join(os.getcwd(), usuario[2].lstrip('/'))
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Obter encoding da face do banco
        db_face_encodings = face_recognition.face_encodings(rgb_img)
        if not db_face_encodings:
            continue
            
        db_face_encoding = db_face_encodings[0]
        
        # Calcular distância entre as faces
        distance = face_recognition.face_distance([db_face_encoding], face_encoding)[0]
        
        if distance < best_distance:
            best_distance = distance
            best_match = {
                'nome': usuario[1],
                'distance': distance,
                'image': base64.b64encode(usuario[2]).decode('utf-8')
            }
    
    if best_match and best_distance < 0.6:  # Threshold para considerar match
        last_recognized_user = best_match['nome']
        matched_image = best_match['image']
        return jsonify({
            'success': True,
            'name': best_match['nome'],
            'distance': float(best_distance),
            'captured_image': captured_image,
            'matched_image': matched_image
        })
    
    return jsonify({
        'success': False,
        'message': 'Usuário não reconhecido',
        'captured_image': captured_image
    })

@app.route('/check_recognition')
def check_recognition():
    return jsonify({
        'name': last_recognized_user,
        'captured_image': captured_image,
        'matched_image': matched_image
    })

@app.route('/recognition_result')
def get_recognition_result():
    global recognition_result
    if recognition_result:
        result = recognition_result
        recognition_result = None  # Limpar resultado após leitura
        return jsonify(result)
    return jsonify(None)

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

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.json
        image_data = data['image'].split(',')[1]
        image = base64.b64decode(image_data)
        np_image = np.frombuffer(image, np.uint8)
        frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # Processar e extrair face da imagem capturada
        face_image, success = process_face_image(frame, mode='CV2')
        
        if not success:
            return jsonify({'success': False, 'message': 'Nenhum rosto detectado. Por favor, posicione seu rosto adequadamente.'})
        
        # Converter a face processada para array numpy para o face_recognition
        face_array = np.array(face_image)
        face_encodings = face_recognition.face_encodings(face_array, num_jitters=10)  # Aumentar precisão com mais samples
        
        if not face_encodings:
            return jsonify({'success': False, 'message': 'Não foi possível processar as características faciais. Por favor, tente novamente com melhor iluminação.'})
            
        face_encoding = face_encodings[0]
        
        # Buscar faces no banco de dados
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT u.id, u.nome, f.caminho FROM usuario u INNER JOIN fotos_usuario f ON u.id = f.usuario_id")
        usuarios = cursor.fetchall()
        cursor.close()
        conn.close()
        
        best_match = None
        best_distance = 1.0
        best_user_id = None
        
        for usuario in usuarios:
            img_path = os.path.join(os.getcwd(), usuario[2].lstrip('/'))
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Processar imagem do banco da mesma forma
            db_face_image, db_success = process_face_image(img, mode='CV2')
            if not db_success:
                continue
                
            db_face_array = np.array(db_face_image)
            db_face_encodings = face_recognition.face_encodings(db_face_array, num_jitters=10)
            
            if not db_face_encodings:
                continue
                
            db_face_encoding = db_face_encodings[0]
            
            # Calcular distância usando mais samples para maior precisão
            distance = face_recognition.face_distance([db_face_encoding], face_encoding)[0]
            
            if distance < best_distance:
                best_distance = distance
                best_match = usuario[1]
                best_user_id = usuario[0]
        
        # Threshold mais rigoroso (0.4) para maior precisão
        if best_match and best_distance < 0.4:
            # Registrar login
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO login (usuario_id, data_login, hora_login)
                VALUES (%s, CURDATE(), CURTIME())
            """, (best_user_id,))
            conn.commit()
            cursor.close()
            conn.close()
            
            return jsonify({
                'success': True,
                'name': best_match,
                'message': f'Bem vindo {best_match}!',
                'distance': float(best_distance),
                'countdown_complete': True
            })
        
        return jsonify({
            'success': False,
            'message': 'Usuário não reconhecido. Por favor, tente novamente.',
            'countdown_complete': True
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

