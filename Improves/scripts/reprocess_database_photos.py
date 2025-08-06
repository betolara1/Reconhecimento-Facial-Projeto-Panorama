import cv2
import face_recognition
import numpy as np
import os
import mysql.connector
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageFilter
import logging

load_dotenv()

# Configuração do banco
db_config = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'reconhecimento_facial'),
    'port': int(os.getenv('DB_PORT', 3306))
}

def enhance_face_image_advanced(image_path, output_path):
    """Reprocessa uma foto para melhor qualidade de reconhecimento"""
    try:
        # Carregar imagem
        image = cv2.imread(image_path)
        if image is None:
            return False, "Não foi possível carregar a imagem"
        
        # Converter para PIL para melhor processamento
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Detectar e extrair apenas a região da face
        rgb_image = np.array(pil_image)
        face_locations = face_recognition.face_locations(rgb_image, model="hog")
        
        if not face_locations:
            return False, "Nenhuma face detectada"
        
        # Pegar a maior face
        top, right, bottom, left = max(face_locations, key=lambda x: (x[2]-x[0])*(x[1]-x[3]))
        
        # Expandir a região da face (margem de 30%)
        height, width = rgb_image.shape[:2]
        face_height = bottom - top
        face_width = right - left
        
        margin_h = int(face_height * 0.3)
        margin_w = int(face_width * 0.3)
        
        top = max(0, top - margin_h)
        bottom = min(height, bottom + margin_h)
        left = max(0, left - margin_w)
        right = min(width, right + margin_w)
        
        # Extrair região da face
        face_image = pil_image.crop((left, top, right, bottom))
        
        # Redimensionar para tamanho padrão
        face_image = face_image.resize((300, 300), Image.Resampling.LANCZOS)
        
        # Melhorar qualidade
        # Contraste
        enhancer = ImageEnhance.Contrast(face_image)
        face_image = enhancer.enhance(1.4)
        
        # Brilho
        enhancer = ImageEnhance.Brightness(face_image)
        face_image = enhancer.enhance(1.1)
        
        # Nitidez
        enhancer = ImageEnhance.Sharpness(face_image)
        face_image = enhancer.enhance(1.3)
        
        # Filtro de nitidez
        face_image = face_image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
        # Converter de volta para OpenCV
        cv_face = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
        
        # Aplicar CLAHE
        lab = cv2.cvtColor(cv_face, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Salvar imagem melhorada
        cv2.imwrite(output_path, enhanced, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return True, "Imagem processada com sucesso"
        
    except Exception as e:
        return False, f"Erro: {str(e)}"

def reprocess_all_photos():
    """Reprocessa todas as fotos do banco de dados"""
    print("🔄 Reprocessando fotos do banco de dados para melhor precisão...")
    
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Buscar todas as fotos
        cursor.execute("""
            SELECT u.id, u.nome, f.caminho 
            FROM usuario u 
            INNER JOIN fotos_usuario f ON u.id = f.usuario_id
        """)
        
        usuarios = cursor.fetchall()
        
        if not usuarios:
            print("❌ Nenhum usuário com foto encontrado")
            return
        
        print(f"📸 Encontradas {len(usuarios)} fotos para reprocessar")
        
        success_count = 0
        
        for user_id, nome, caminho in usuarios:
            print(f"\n🔍 Processando {nome}...")
            
            # Caminho da imagem original
            original_path = os.path.join(os.getcwd(), caminho.lstrip('/'))
            
            if not os.path.exists(original_path):
                print(f"   ❌ Arquivo não encontrado: {original_path}")
                continue
            
            # Criar backup da imagem original
            backup_path = original_path.replace('.jpg', '_backup.jpg')
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(original_path, backup_path)
                print(f"   💾 Backup criado: {backup_path}")
            
            # Reprocessar imagem
            success, message = enhance_face_image_advanced(original_path, original_path)
            
            if success:
                print(f"   ✅ {message}")
                success_count += 1
                
                # Testar qualidade do encoding
                try:
                    test_image = cv2.imread(original_path)
                    rgb_test = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_test)
                    
                    if face_locations:
                        encodings = face_recognition.face_encodings(rgb_test, face_locations)
                        if encodings:
                            print(f"   🎯 Encoding extraído com sucesso")
                        else:
                            print(f"   ⚠️  Falha ao extrair encoding")
                    else:
                        print(f"   ⚠️  Nenhuma face detectada na imagem processada")
                        
                except Exception as e:
                    print(f"   ⚠️  Erro no teste de encoding: {e}")
            else:
                print(f"   ❌ {message}")
        
        print(f"\n📊 Resumo:")
        print(f"   • Total de fotos: {len(usuarios)}")
        print(f"   • Processadas com sucesso: {success_count}")
        print(f"   • Falhas: {len(usuarios) - success_count}")
        
        if success_count > 0:
            print(f"\n✅ Reprocessamento concluído!")
            print(f"💡 Reinicie o sistema para atualizar o cache com as novas imagens")
        
    except Exception as e:
        print(f"❌ Erro no reprocessamento: {e}")
    
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    reprocess_all_photos()
