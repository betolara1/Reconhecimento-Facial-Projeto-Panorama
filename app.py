from flask import Flask, render_template, request, redirect, url_for, flash, Response, jsonify
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

app = Flask(__name__)
app.secret_key = 'sua_chave_secreta'

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
UPLOAD_FOLDER = 'static/fotos'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Rota principal
@app.route('/')
def index():
    return render_template('index.html')

# Rota para cadastro de usuário
@app.route('/cadastro', methods=['GET', 'POST'])
def cadastro():
    if request.method == 'POST':
        nome = request.form['nome']
        telefone = request.form['telefone']
        cep = request.form['cep']
        rua = request.form['rua']
        numero = request.form['numero']
        bairro = request.form['bairro']
        cidade = request.form['cidade']
        uf = request.form['uf']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO usuario (nome, telefone, cep, rua, numero, bairro, cidade, uf)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (nome, telefone, cep, rua, numero, bairro, cidade, uf))
            
            conn.commit()
            flash('Usuário cadastrado com sucesso!', 'success')
            return redirect(url_for('cadastro'))
        except Exception as e:
            flash(f'Erro ao cadastrar usuário: {str(e)}', 'error')
            return redirect(url_for('cadastro'))
        finally:
            cursor.close()
            conn.close()
            
    return render_template('cadastro.html')

# Rota para captura de fotos
@app.route('/captura_fotos')
def captura_fotos():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Modificar a consulta para retornar apenas usuários sem fotos
    cursor.execute("""
        SELECT u.cod, u.nome 
        FROM usuario u 
        LEFT JOIN fotos_usuario f ON u.cod = f.usuario_id 
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

            usuario_id = request.form.get('usuario_id')
            foto_base64 = request.form.get('foto')

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
                cursor.execute("SELECT nome FROM usuario WHERE cod = %s", (usuario_id,))
                result = cursor.fetchone()
                if not result:
                    return jsonify({'success': False, 'message': 'Usuário não encontrado'}), 404
                nome_usuario = result[0]
            except Exception as e:
                return jsonify({'success': False, 'message': f'Erro ao buscar usuário: {str(e)}'}), 500

            try:
                # Processar imagem
                image = Image.open(BytesIO(foto_bytes))
                
                # Converter para RGB
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Redimensionar
                image.thumbnail((400, 400), Image.Resampling.LANCZOS)
                
                # Gerar nome único para o arquivo
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{nome_usuario.replace(' ', '_')}_{timestamp}.jpg"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                
                # Salvar imagem
                image.save(filepath, 'JPEG', quality=95)
                
                # Salvar no banco
                db_filepath = f"/static/fotos/{filename}"
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

# Rota para processar o vídeo do login facial
def gerar_frames():
    global face_detection_start_time, face_detected, recognition_complete, recognition_result
    
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Converter frame para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        
        # Se não detectar rosto, resetar variáveis
        if not face_locations:
            face_detected = False
            face_detection_start_time = None
            recognition_complete = False  # Resetar para permitir novo reconhecimento
            
        # Desenhar retângulo e informações na tela
        for (top, right, bottom, left) in face_locations:
            # Desenhar retângulo verde ao redor do rosto
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            if not face_detected:
                face_detected = True
                face_detection_start_time = time.time()
                recognition_complete = False  # Resetar ao detectar novo rosto
            
            # Calcular tempo restante
            if face_detection_start_time and not recognition_complete:
                elapsed_time = time.time() - face_detection_start_time
                remaining_time = max(3 - int(elapsed_time), 0)
                
                # Mostrar contagem regressiva
                cv2.putText(frame, f"Aguarde: {remaining_time}s", 
                          (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.75, (0, 255, 0), 2)
                
                # Após 5 segundos, realizar o reconhecimento
                if elapsed_time >= 5 and not recognition_complete:
                    # Realizar reconhecimento facial
                    face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]
                    
                    # Buscar faces no banco de dados
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT u.cod, u.nome, f.caminho FROM usuario u INNER JOIN fotos_usuario f ON u.cod = f.usuario_id")
                    usuarios = cursor.fetchall()
                    cursor.close()
                    conn.close()
                    
                    best_match = None
                    best_distance = 1.0
                    best_user_id = None  # Adicionar variável para armazenar ID do usuário
                    
                    for usuario in usuarios:
                        img_path = os.path.join(os.getcwd(), usuario[2].lstrip('/'))
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        
                        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        db_face_locations = face_recognition.face_locations(rgb_img)
                        if not db_face_locations:
                            continue
                        
                        db_face_encoding = face_recognition.face_encodings(rgb_img)[0]
                        distance = face_recognition.face_distance([db_face_encoding], face_encoding)[0]
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_match = usuario[1]
                            best_user_id = usuario[0]  # Armazenar ID do usuário
                    
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
                    # Resetar após 2 segundos
                    Timer(2.0, reset_recognition_vars).start()
        
        # Mostrar resultado do reconhecimento
        if recognition_result:
            cv2.putText(frame, recognition_result['message'], 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, (0, 255, 0), 2)
        
        # Converter frame para jpg
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
    camera = cv2.VideoCapture(0)
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
    cursor.execute("SELECT u.cod, u.nome, f.caminho FROM usuario u INNER JOIN fotos_usuario f ON u.cod = f.usuario_id")
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
        db_face_locations = face_recognition.face_locations(rgb_img)
        if not db_face_locations:
            continue
            
        db_face_encoding = face_recognition.face_encodings(rgb_img)[0]
        
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
            JOIN usuario u ON l.usuario_id = u.cod
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

if __name__ == '__main__':
    app.run(debug=True) 