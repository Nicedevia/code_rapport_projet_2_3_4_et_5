# Utiliser Python 3.9 comme base
FROM python:3.9

# Désactiver l'interaction pour éviter les interruptions
ENV DEBIAN_FRONTEND=noninteractive

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier tous les fichiers du projet dans le conteneur
COPY ./ /app/

# Mettre à jour pip et installer `certifi` pour corriger les erreurs SSL
RUN python -m pip install --upgrade pip certifi

# Définir la variable d'environnement pour les certificats SSL
ENV SSL_CERT_FILE=/usr/local/lib/python3.9/site-packages/certifi/cacert.pem

# Installer les packages nécessaires, en forçant `pip` à ignorer les erreurs SSL
RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    fastapi uvicorn opencv-python pillow prometheus_client librosa pydub \
    tensorflow==2.15.0 keras==2.15.0 python-multipart

# Exposer le port 8000 pour l'API
EXPOSE 8000
# port promet 
EXPOSE 9090

# Assurer que le dossier des modèles existe
RUN mkdir -p /app/models

# Copier les modèles dans le conteneur
COPY models /app/models

# Lancer l'API avec Uvicorn
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]
