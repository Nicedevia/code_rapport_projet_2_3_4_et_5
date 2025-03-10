# config-3.py
# 📌 Fichier de configuration principal du projet

import os

# Définition des chemins de base
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Configuration des logs
LOG_LEVEL = "INFO"

# Configuration du modèle
MODEL_PATH = os.path.join(BASE_DIR, "models", "image_audio_fusion_new_model_v2.keras")

# Configuration de l'API (si nécessaire)
API_HOST = "0.0.0.0"
API_PORT = 8000

print(f"✅ Configuration chargée depuis {__file__}")
