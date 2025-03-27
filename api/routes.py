from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import numpy as np
import io
import cv2
import librosa
import tensorflow as tf
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from .model_loader import load_image_model, load_audio_model, load_fusion_model

DEFAULT_THRESHOLD = 0.5
router = APIRouter()

# Charger les modèles
image_model = load_image_model()
audio_model = load_audio_model()
fusion_model = load_fusion_model()

# Forcer l'appel des modèles avec un dummy input pour définir leurs inputs
dummy_image = tf.zeros((1, 64, 64, 1))
dummy_audio = tf.zeros((1, 64, 64, 1))
_ = image_model(dummy_image)
_ = audio_model(dummy_audio)

# Création des extracteurs en utilisant inputs[0]
image_extractor = tf.keras.Model(
    inputs=image_model.inputs[0],
    outputs=image_model.layers[-2].output
)
audio_extractor = tf.keras.Model(
    inputs=audio_model.inputs[0],
    outputs=audio_model.layers[-2].output
)

# ---------------------------
# Définition des métriques Prometheus
# ---------------------------
request_counter = Counter("http_requests_total", "Nombre total de requêtes reçues")
prediction_duration = Histogram("model_prediction_duration_seconds", "Durée des prédictions du modèle en secondes")
prediction_errors = Counter("model_prediction_errors_total", "Nombre total d'erreurs lors des prédictions")

# Endpoint pour exposer les métriques Prometheus
@router.get("/metrics", tags=["Monitoring"])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ---------------------------
# Fonctions de Prétraitement
# ---------------------------
def preprocess_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail="Image invalide")
    img = cv2.resize(img, (64, 64)) / 255.0
    img = img.reshape(1, 64, 64, 1)
    features = image_extractor.predict(img)
    return features

def preprocess_audio_from_bytes(audio_bytes: bytes) -> np.ndarray:
    try:
        audio_stream = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_stream, sr=22050, duration=2)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Audio invalide") from e

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    spec_img = cv2.resize(S_db, (64, 64))
    spec_img = (spec_img - spec_img.min()) / (spec_img.max() - spec_img.min())
    spec_img = spec_img.reshape(1, 64, 64, 1)
    features = audio_extractor.predict(spec_img)
    return features

# ---------------------------
# Endpoints de Prédiction
# ---------------------------

# Endpoint pour la prédiction d'une image seule
@router.post("/predict/image", tags=["Prediction"])
async def predict_image(file: UploadFile = File(..., description="Fichier image (JPEG ou PNG)")):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Format d'image non supporté")
    image_bytes = await file.read()
    try:
        features_image = preprocess_image_from_bytes(image_bytes)
    except Exception as e:
        prediction_errors.inc()
        raise e

    # Appel du modèle d'image (prédiction simplifiée)
    prediction = image_model.predict(features_image)
    label = "Chien" if prediction[0][0] > DEFAULT_THRESHOLD else "Chat"
    confidence = float(prediction[0][0])
    return {"prediction": label, "confidence": confidence, "used_threshold": DEFAULT_THRESHOLD}

# Endpoint pour la prédiction d'un audio seul
@router.post("/predict/audio", tags=["Prediction"])
async def predict_audio(file: UploadFile = File(..., description="Fichier audio (WAV)")):
    if file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Format audio non supporté")
    audio_bytes = await file.read()
    try:
        features_audio = preprocess_audio_from_bytes(audio_bytes)
    except Exception as e:
        prediction_errors.inc()
        raise e

    # Appel du modèle audio (prédiction simplifiée)
    prediction = audio_model.predict(features_audio)
    label = "Chien" if prediction[0][0] > DEFAULT_THRESHOLD else "Chat"
    confidence = float(prediction[0][0])
    return {"prediction": label, "confidence": confidence, "used_threshold": DEFAULT_THRESHOLD}

# Endpoint pour la prédiction multimodale (image et audio)
@router.post("/predict/multimodal", tags=["Prediction"])
async def predict_multimodal(
    image_file: UploadFile = File(..., description="Fichier image (JPEG ou PNG)"),
    audio_file: UploadFile = File(..., description="Fichier audio (WAV)"),
    threshold: float = Query(DEFAULT_THRESHOLD, ge=0, le=1, description="Seuil pour la classification (0 à 1)")
):
    if image_file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Format d'image non supporté")
    if audio_file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Format audio non supporté")
    
    image_bytes = await image_file.read()
    audio_bytes = await audio_file.read()
    
    try:
        features_image = preprocess_image_from_bytes(image_bytes)
        features_audio = preprocess_audio_from_bytes(audio_bytes)
    except Exception as e:
        prediction_errors.inc()
        raise e

    start_time = time.time()
    prediction = fusion_model.predict([features_image, features_audio])
    duration = time.time() - start_time
    prediction_duration.observe(duration)
    
    label = "Chien" if prediction[0][0] > threshold else "Chat"
    confidence = float(prediction[0][0])
    return {"prediction": label, "confidence": confidence, "used_threshold": threshold}

# ---------------------------
# Authentification et Sécurisation
# ---------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.post("/token", tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    return {"access_token": "dummy_token", "token_type": "bearer"}

@router.get("/protected", tags=["Authentication"])
async def protected_route(token: str = Depends(oauth2_scheme)):
    if token != "dummy_token":
        raise HTTPException(status_code=401, detail="Token invalide")
    return {"message": "Accès autorisé"}

from fastapi.responses import PlainTextResponse
import subprocess
import os

@router.get("/force-error", response_class=PlainTextResponse)
def trigger_error_and_show_report():
    # Déclencher volontairement une erreur
    try:
        raise ValueError("Erreur volontaire pour test MCO")
    except Exception as e:
        # Génère le rapport d'incident via le script
        subprocess.run(["python", "logs/incident_report.py"], check=False)

        # Lire et renvoyer le contenu du rapport
        report_path = "incident_report.md"
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                return f.read()
        else:
            return "Aucun rapport généré."
