# 1. Offizielles Python-Image nutzen
FROM python:3.10-slim

# 2. Arbeitsverzeichnis definieren
WORKDIR /app

# 3. GCC Compiler installieren, damit 'twofish' gebaut werden kann
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. requirements.txt kopieren und Pip upgraden
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip

# 5. Die Python-Abhängigkeiten installieren
RUN pip install --no-cache-dir -r requirements.txt

# 6. Den gesamten Projektcode in den Container kopieren
COPY . .

# 7. Den Netzwerk-Port für Streamlit freigeben
EXPOSE 8501

# 8. Startbefehl für Streamlit
CMD ["python", "-m", "streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]