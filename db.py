# CRIAÇÃO DO BANCO DE DADOS E TABELAS

import mysql.connector
import os
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv()

db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME'),
    'port': os.getenv('DB_PORT')
}

# Função para conectar ao banco de dados
def get_db_connection():
    connection = mysql.connector.connect(**db_config)
    return connection