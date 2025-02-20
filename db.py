# CRIAÇÃO DO BANCO DE DADOS E TABELAS

import mysql.connector

db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '', 
    'database': 'cadastro',
    'port' : '3306'
}

# Função para conectar ao banco de dados
def get_db_connection():
    connection = mysql.connector.connect(**db_config)
    return connection