import sys
import os
# Ajoute le dossier racine du projet dans PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.api import app

client = TestClient(app)

# 📌 Chemins vers les fichiers de test 
TEST_IMAGE_PATH = "data/images/cleaned/test_set/cats/cat.16.jpg"
TEST_AUDIO_PATH = "data/audio/cleaned/train/cats/cat_1.wav"
TEST_INVALID_IMAGE_PATH = "tests/test_invalid.txt"
TEST_INVALID_AUDIO_PATH = "tests/test_invalid.mp3"

# --- 🔎 TESTS POUR L'API DE CLASSIFICATION ---
def test_predict_image():
    """✅ Vérifie la prédiction d'une image valide."""
    with open(TEST_IMAGE_PATH, "rb") as file:
        response = client.post("/predict/image", files={"file": file})
    assert response.status_code == 200
    json_response = response.json()
    assert "prediction" in json_response
    assert "confidence" in json_response

def test_predict_audio():
    """✅ Vérifie la prédiction d'un fichier audio valide."""
    with open(TEST_AUDIO_PATH, "rb") as file:
        response = client.post("/predict/audio", files={"file": file})
    assert response.status_code == 200
    json_response = response.json()
    assert "prediction" in json_response
    assert "confidence" in json_response

def test_predict_multimodal():
    """✅ Vérifie la prédiction avec une image et un audio valides."""
    with open(TEST_IMAGE_PATH, "rb") as image_file, open(TEST_AUDIO_PATH, "rb") as audio_file:
        response = client.post("/predict/multimodal", files={
            "image_file": ("test_image.jpg", image_file, "image/jpeg"),
            "audio_file": ("test_audio.wav", audio_file, "audio/wav")
        })
    assert response.status_code == 200
    json_response = response.json()
    assert "prediction" in json_response
    assert "confidence" in json_response

# --- ❌ TESTS POUR LES CAS D'ERREURS ---
def test_predict_image_invalid_format():
    """🚨 Vérifie le refus d'un fichier image au format invalide."""
    with open(TEST_INVALID_IMAGE_PATH, "rb") as file:
        response = client.post("/predict/image", files={"file": file})
    assert response.status_code == 400
    assert response.json()["detail"] == "Format d'image non supporté"

def test_predict_audio_invalid_format():
    """🚨 Vérifie le refus d'un fichier audio au format invalide."""
    with open(TEST_INVALID_AUDIO_PATH, "rb") as file:
        response = client.post("/predict/audio", files={"file": file})
    assert response.status_code == 400
    assert response.json()["detail"] == "Format audio non supporté"

def test_predict_multimodal_missing_file():
    """🚨 Vérifie l'échec si un fichier est manquant."""
    with open(TEST_IMAGE_PATH, "rb") as image_file:
        response = client.post("/predict/multimodal", files={
            "image_file": ("test_image.jpg", image_file, "image/jpeg")
        })
    assert response.status_code == 422  # FastAPI renvoie 422 pour les paramètres manquants

def test_predict_multimodal_invalid_files():
    """🚨 Vérifie que l'API rejette des fichiers invalides pour l'image et l'audio."""
    with open(TEST_INVALID_IMAGE_PATH, "rb") as image_file, open(TEST_INVALID_AUDIO_PATH, "rb") as audio_file:
        response = client.post("/predict/multimodal", files={
            "image_file": ("test_invalid.txt", image_file, "text/plain"),
            "audio_file": ("test_invalid.mp3", audio_file, "audio/mp3")
        })
    assert response.status_code == 400

# --- 🔐 TESTS D'AUTHENTIFICATION ---
def test_authentication():
    """✅ Vérifie le fonctionnement du login et l'accès protégé."""
    response = client.post("/token", data={"username": "user", "password": "pass"})
    assert response.status_code == 200
    token = response.json().get("access_token")
    
    response = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json().get("message") == "Accès autorisé"

def test_authentication_invalid_token():
    """🚨 Vérifie qu'un mauvais token est refusé."""
    response = client.get("/protected", headers={"Authorization": "Bearer wrong_token"})
    assert response.status_code == 401

# --- 📊 TEST METRICS ---
def test_metrics():
    """✅ Vérifie que les métriques Prometheus sont accessibles."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text  # Vérifier la présence d'une métrique
