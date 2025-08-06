import cv2
import face_recognition
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import base64
from io import BytesIO
import mysql.connector
import os
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

class ImprovedFaceRecognition:
    def __init__(self, db_config):
        self.db_config = db_config
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def get_db_connection(self):
        return mysql.connector.connect(**self.db_config)
    
    def enhance_image_quality(self, image):
        """
        Melhora a qualidade da imagem para melhor reconhecimento facial
        """
        # Converter para PIL se necessário
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Melhorar contraste
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Melhorar brilho
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.1)
        
        # Aplicar filtro de nitidez suave
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        
        return image
    
    def normalize_lighting(self, image):
        """
        Normaliza a iluminação da imagem usando CLAHE
        """
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Converter para LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Aplicar CLAHE no canal L
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Recombinar canais
        lab = cv2.merge([l, a, b])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return image
    
    def detect_face_quality(self, image, face_location):
        """
        Avalia a qualidade da face detectada
        """
        top, right, bottom, left = face_location
        face_width = right - left
        face_height = bottom - top
        
        # Verificar tamanho mínimo da face
        min_face_size = 80
        if face_width < min_face_size or face_height < min_face_size:
            return False, "Face muito pequena"
        
        # Verificar se a face não está muito próxima das bordas
        img_height, img_width = image.shape[:2]
        margin = 20
        if (left < margin or top < margin or 
            right > img_width - margin or bottom > img_height - margin):
            return False, "Face muito próxima da borda"
        
        # Verificar proporção da face
        aspect_ratio = face_width / face_height
        if aspect_ratio < 0.7 or aspect_ratio > 1.4:
            return False, "Proporção da face inadequada"
        
        return True, "Face adequada"
    
    def extract_multiple_encodings(self, image, num_jitters=10):
        """
        Extrai múltiplos encodings da mesma face para maior robustez
        """
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Normalizar iluminação
        image = self.normalize_lighting(image)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detectar faces com diferentes modelos
        face_locations_hog = face_recognition.face_locations(rgb_image, model="hog", number_of_times_to_upsample=2)
        face_locations_cnn = face_recognition.face_locations(rgb_image, model="cnn", number_of_times_to_upsample=1)
        
        all_encodings = []
        
        # Processar detecções HOG
        for face_location in face_locations_hog:
            quality_ok, quality_msg = self.detect_face_quality(rgb_image, face_location)
            if quality_ok:
                encodings = face_recognition.face_encodings(
                    rgb_image, 
                    [face_location], 
                    num_jitters=num_jitters,
                    model="large"
                )
                if encodings:
                    all_encodings.extend(encodings)
        
        # Processar detecções CNN (se disponível)
        try:
            for face_location in face_locations_cnn:
                quality_ok, quality_msg = self.detect_face_quality(rgb_image, face_location)
                if quality_ok:
                    encodings = face_recognition.face_encodings(
                        rgb_image, 
                        [face_location], 
                        num_jitters=num_jitters,
                        model="large"
                    )
                    if encodings:
                        all_encodings.extend(encodings)
        except:
            pass  # CNN pode não estar disponível
        
        return all_encodings
    
    def calculate_similarity_score(self, encoding1, encoding2):
        """
        Calcula score de similaridade usando múltiplas métricas
        """
        # Distância euclidiana (face_recognition padrão)
        euclidean_distance = face_recognition.face_distance([encoding1], encoding2)[0]
        
        # Similaridade coseno
        cosine_sim = cosine_similarity([encoding1], [encoding2])[0][0]
        
        # Score combinado (quanto menor, mais similar)
        combined_score = (1 - cosine_sim) * 0.3 + euclidean_distance * 0.7
        
        return combined_score, euclidean_distance, cosine_sim
    
    def process_and_recognize_face(self, image_data, confidence_threshold=0.45):
        """
        Processa imagem e realiza reconhecimento facial com alta precisão
        """
        try:
            # Decodificar imagem
            if isinstance(image_data, str):
                if ',' in image_data:
                    image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = Image.open(BytesIO(image_bytes))
            else:
                image = image_data
            
            # Melhorar qualidade da imagem
            enhanced_image = self.enhance_image_quality(image)
            
            # Extrair múltiplos encodings
            captured_encodings = self.extract_multiple_encodings(enhanced_image)
            
            if not captured_encodings:
                return {
                    'success': False, 
                    'message': 'Nenhum rosto detectado com qualidade adequada. Tente melhorar a iluminação e posicionamento.'
                }
            
            # Buscar usuários no banco
            conn = self.get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT u.id, u.nome, f.caminho 
                FROM usuario u 
                INNER JOIN fotos_usuario f ON u.id = f.usuario_id
            """)
            usuarios = cursor.fetchall()
            cursor.close()
            conn.close()
            
            best_matches = []
            
            for usuario in usuarios:
                user_id, nome, caminho = usuario
                
                # Carregar e processar imagem do banco
                img_path = os.path.join(os.getcwd(), caminho.lstrip('/'))
                if not os.path.exists(img_path):
                    continue
                
                db_image = Image.open(img_path)
                db_encodings = self.extract_multiple_encodings(db_image)
                
                if not db_encodings:
                    continue
                
                # Comparar todos os encodings capturados com todos do banco
                best_score = float('inf')
                best_details = None
                
                for captured_encoding in captured_encodings:
                    for db_encoding in db_encodings:
                        score, euclidean, cosine = self.calculate_similarity_score(
                            captured_encoding, db_encoding
                        )
                        
                        if score < best_score:
                            best_score = score
                            best_details = {
                                'euclidean': euclidean,
                                'cosine': cosine,
                                'combined': score
                            }
                
                if best_score < confidence_threshold:
                    best_matches.append({
                        'user_id': user_id,
                        'nome': nome,
                        'score': best_score,
                        'details': best_details
                    })
            
            # Ordenar por melhor score
            best_matches.sort(key=lambda x: x['score'])
            
            if best_matches:
                best_match = best_matches[0]
                
                # Verificar se o melhor match é significativamente melhor que o segundo
                confidence_gap = 0.1
                if len(best_matches) > 1:
                    second_best = best_matches[1]
                    if best_match['score'] + confidence_gap > second_best['score']:
                        return {
                            'success': False,
                            'message': 'Reconhecimento ambíguo. Tente novamente com melhor posicionamento.',
                            'debug': {
                                'best_score': best_match['score'],
                                'second_score': second_best['score']
                            }
                        }
                
                # Registrar login
                self.register_login(best_match['user_id'])
                
                return {
                    'success': True,
                    'name': best_match['nome'],
                    'message': f'Bem-vindo, {best_match["nome"]}!',
                    'confidence': 1 - best_match['score'],
                    'details': best_match['details']
                }
            
            return {
                'success': False,
                'message': 'Usuário não reconhecido. Verifique se você está cadastrado no sistema.'
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Erro no processamento: {str(e)}'
            }
    
    def register_login(self, user_id):
        """
        Registra o login no banco de dados
        """
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO login (usuario_id, data_login, hora_login)
                VALUES (%s, CURDATE(), CURTIME())
            """, (user_id,))
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Erro ao registrar login: {e}") 

# Exemplo de uso
if __name__ == "__main__":
    # Configuração do banco
    db_config = {
        'host': 'localhost',
        'user': 'useradmin',
        'password': 'c4dHJnYj3Sm6DVUjbFVN',
        'database': 'api-panorama',
        'port': 3306
    }
    
    # Inicializar sistema
    face_system = ImprovedFaceRecognition(db_config)
    
    # Exemplo de reconhecimento
    # result = face_system.process_and_recognize_face(image_data)
    # print(result)
