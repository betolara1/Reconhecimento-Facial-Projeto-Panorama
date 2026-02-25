import pytest
import base64
from unittest.mock import patch, MagicMock
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    app.config['WTF_CSRF_ENABLED'] = False
    with app.test_client() as client:
        yield client

def test_index_page(client):
    """Testa se a página inicial carrega corretamente"""
    response = client.get('/')
    assert response.status_code == 200
    assert b'Cadastrar' in response.data or b'Login' in response.data

def test_cadastro_page(client):
    """Testa se a página de cadastro carrega"""
    response = client.get('/cadastro')
    assert response.status_code == 200

def test_login_page(client):
    """Testa se a página de login carrega"""
    response = client.get('/login')
    assert response.status_code == 200

def test_captura_fotos_page(client):
    """Testa se a página de captura de fotos carrega"""
    response = client.get('/captura_fotos')
    assert response.status_code == 200

@patch('app.get_db_connection')
def test_health_check(mock_get_db, client):
    """Testa o endpoint de verificação de saúde da API (Health Check)"""
    # Setup mock para conexão com o banco
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = [5] # Simula 5 usuários no banco
    mock_conn.cursor.return_value = mock_cursor
    mock_get_db.return_value = mock_conn
    
    response = client.get('/health')
    assert response.status_code == 200
    
    data = response.get_json()
    assert data['status'] == 'OK'
    assert data['database'] == 'Connected'
    assert data['users_in_db'] == 5

@patch('app.get_db_connection')
def test_health_check_db_failure(mock_get_db, client):
    """Testa o health check quando o banco de dados falha conectando"""
    mock_get_db.return_value = None
    
    response = client.get('/health')
    assert response.status_code == 500
    
    data = response.get_json()
    assert data['status'] == 'OK'
    assert data['database'] == 'Error'

@patch('requests.get')
@patch('app.get_db_connection')
def test_get_alunos_php(mock_get_db, mock_requests_get, client):
    """Testa a integração mockada com a API PHP de obter alunos"""
    # Mock do banco local (retornando vazio para IDs com foto)
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value = mock_cursor
    mock_get_db.return_value = mock_conn
    
    # Mock da resposta do request ao PHP
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"id": 1, "nome": "Aluno Teste 1"},
        {"id": 2, "nome": "Aluno Teste 2"}
    ]
    mock_requests_get.return_value = mock_response
    
    response = client.get('/api/alunos_php')
    assert response.status_code == 200
    
    data = response.get_json()
    assert len(data) == 2
    assert data[0]['nome'] == 'Aluno Teste 1'
