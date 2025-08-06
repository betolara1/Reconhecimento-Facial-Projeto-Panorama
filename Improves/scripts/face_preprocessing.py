import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import face_recognition

class FacePreprocessor:
    """
    Classe especializada em pré-processamento de imagens para reconhecimento facial
    """
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    def detect_and_align_face(self, image):
        """
        Detecta e alinha a face baseado na posição dos olhos
        """
        if isinstance(image, Image.Image):
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            cv_image = image.copy()
        
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        # Pegar a maior face
        face = max(faces, key=lambda x: x[2] * x[3])
        x, y, w, h = face
        
        # Região da face
        face_gray = gray[y:y+h, x:x+w]
        face_color = cv_image[y:y+h, x:x+w]
        
        # Detectar olhos na região da face
        eyes = self.eye_cascade.detectMultiScale(face_gray)
        
        if len(eyes) >= 2:
            # Ordenar olhos por posição x
            eyes = sorted(eyes, key=lambda x: x[0])
            
            # Calcular centros dos olhos
            eye1_center = (eyes[0][0] + eyes[0][2]//2, eyes[0][1] + eyes[0][3]//2)
            eye2_center = (eyes[1][0] + eyes[1][2]//2, eyes[1][1] + eyes[1][3]//2)
            
            # Calcular ângulo de rotação
            dy = eye2_center[1] - eye1_center[1]
            dx = eye2_center[0] - eye1_center[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Rotacionar imagem para alinhar olhos
            center = (face_color.shape[1]//2, face_color.shape[0]//2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            aligned_face = cv2.warpAffine(face_color, rotation_matrix, (face_color.shape[1], face_color.shape[0]))
            
            return aligned_face
        
        return face_color
    
    def enhance_for_recognition(self, image):
        """
        Aplica melhorias específicas para reconhecimento facial
        """
        # Normalização de histograma
        if len(image.shape) == 3:
            # Para imagem colorida, aplicar em cada canal
            enhanced = np.zeros_like(image)
            for i in range(3):
                enhanced[:,:,i] = cv2.equalizeHist(image[:,:,i])
        else:
            enhanced = cv2.equalizeHist(image)
        
        # Redução de ruído
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Melhoria de contraste local
        if len(enhanced.shape) == 3:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def standardize_face_size(self, face_image, target_size=(200, 200)):
        """
        Padroniza o tamanho da face para comparação consistente
        """
        if isinstance(face_image, np.ndarray):
            face_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        
        # Redimensionar mantendo proporção
        face_image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Criar imagem com fundo branco do tamanho alvo
        standardized = Image.new('RGB', target_size, (255, 255, 255))
        
        # Centralizar a face
        x = (target_size[0] - face_image.width) // 2
        y = (target_size[1] - face_image.height) // 2
        standardized.paste(face_image, (x, y))
        
        return standardized
    
    def process_face_for_database(self, image):
        """
        Pipeline completo de processamento para armazenar no banco
        """
        # Detectar e alinhar face
        aligned_face = self.detect_and_align_face(image)
        if aligned_face is None:
            return None, False
        
        # Melhorar qualidade
        enhanced_face = self.enhance_for_recognition(aligned_face)
        
        # Padronizar tamanho
        standardized_face = self.standardize_face_size(enhanced_face)
        
        return standardized_face, True
    
    def extract_robust_encoding(self, image, num_samples=15):
        """
        Extrai encoding robusto usando múltiplas amostras
        """
        if isinstance(image, Image.Image):
            np_image = np.array(image)
        else:
            np_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        encodings = []
        
        # Extrair múltiplos encodings com diferentes parâmetros
        for jitter in range(5, num_samples + 5, 5):
            try:
                face_encodings = face_recognition.face_encodings(
                    np_image, 
                    num_jitters=jitter,
                    model="large"
                )
                if face_encodings:
                    encodings.extend(face_encodings)
            except:
                continue
        
        if not encodings:
            return None
        
        # Calcular encoding médio para maior estabilidade
        mean_encoding = np.mean(encodings, axis=0)
        
        return mean_encoding

# Exemplo de uso do preprocessador
if __name__ == "__main__":
    preprocessor = FacePreprocessor()
    
    # Carregar imagem
    image = cv2.imread("caminho/para/imagem.jpg")
    
    # Processar para banco de dados
    processed_face, success = preprocessor.process_face_for_database(image)
    
    if success:
        # Extrair encoding robusto
        encoding = preprocessor.extract_robust_encoding(processed_face)
        print("Processamento concluído com sucesso!")
    else:
        print("Falha no processamento da face")
