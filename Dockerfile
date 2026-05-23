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

# Den Port-Zwang von 8501 entfernen und durch den Hugging Face Standard ersetzen
EXPOSE 7860

# Startbefehl: Streamlit liest den Port nun dynamisch aus der Cloud-Umgebung
CMD ["streamlit", "run", "src/app.py", "--server.port=7860", "--server.address=0.0.0.0"]