# Usamos uma imagem base do python oficial com uma versão mais recente (slim para ser mais leve)
FROM python:3.10-slim

# Definir diretório de trabalho no container
WORKDIR /app

# Instalar dependências de sistema necessárias para compilar bibliotecas como dlib e opencv
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    pkg-config \
    default-libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivo de requisitos para a imagem
COPY requirements.txt .

# Atualizar o pip e instalar as dependências
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar todo o código da aplicação para o container
COPY . .

# Expomos a porta definida no gunicorn.conf.py
EXPOSE 8090

# Comando para rodar a aplicação escutando em todos os IPs (em docker isso é necessário para a porta ser acessível externamente)
CMD ["gunicorn", "--workers", "4", "--timeout", "120", "--bind", "0.0.0.0:8090", "wsgi:app"]
