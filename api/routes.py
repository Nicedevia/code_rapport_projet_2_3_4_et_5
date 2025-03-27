# --- api/routes.py ---
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import Response, PlainTextResponse
import numpy as np
import io
import cv2
import librosa
import tensorflow as tf
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from .model_loader import load_image_model, load_audio_model, load_fusion_model
import subprocess
import os
import logging

DEFAULT_THRESHOLD = 0.5
router = APIRouter()

image_model = load_image_model()
audio_model = load_audio_model()
fusion_model = load_fusion_model()

# Dummy pass
_ = image_model(tf.zeros((1, 64, 64, 1)))
_ = audio_model(tf.zeros((1, 64, 64, 1)))

image_extractor = tf.keras.Model(inputs=image_model.inputs[0], outputs=image_model.layers[-2].output)
audio_extractor = tf.keras.Model(inputs=audio_model.inputs[0], outputs=audio_model.layers[-2].output)

request_counter = Counter("http_requests_total", "Nombre total de requêtes reçues")
prediction_duration = Histogram("model_prediction_duration_seconds", "Durée des prédictions du modèle en secondes")
prediction_errors = Counter("model_prediction_errors_total", "Nombre total d'erreurs lors des prédictions")

logger = logging.getLogger("main_logger")

@router.get("/metrics", tags=["Monitoring"])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

def preprocess_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail="Image invalide")
    img = cv2.resize(img, (64, 64)) / 255.0
    img = img.reshape(1, 64, 64, 1)
    return image_extractor.predict(img)

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
    return audio_extractor.predict(spec_img)

@router.post("/predict/image", tags=["Prediction"])
async def predict_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Format d'image non supporté")
    image_bytes = await file.read()
    try:
        features = preprocess_image_from_bytes(image_bytes)
    except Exception as e:
        prediction_errors.inc()
        raise e
    prediction = image_model.predict(features)
    label = "Chien" if prediction[0][0] > DEFAULT_THRESHOLD else "Chat"
    return {"prediction": label, "confidence": float(prediction[0][0]), "used_threshold": DEFAULT_THRESHOLD}

@router.post("/predict/audio", tags=["Prediction"])
async def predict_audio(file: UploadFile = File(...)):
    if file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Format audio non supporté")
    audio_bytes = await file.read()
    try:
        features = preprocess_audio_from_bytes(audio_bytes)
    except Exception as e:
        prediction_errors.inc()
        raise e
    prediction = audio_model.predict(features)
    label = "Chien" if prediction[0][0] > DEFAULT_THRESHOLD else "Chat"
    return {"prediction": label, "confidence": float(prediction[0][0]), "used_threshold": DEFAULT_THRESHOLD}

@router.post("/predict/multimodal", tags=["Prediction"])
async def predict_multimodal(
    image_file: UploadFile = File(...),
    audio_file: UploadFile = File(...),
    threshold: float = Query(DEFAULT_THRESHOLD, ge=0, le=1)
):
    if image_file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Format d'image non supporté")
    if audio_file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Format audio non supporté")
    image_bytes = await image_file.read()
    audio_bytes = await audio_file.read()
    try:
        f_img = preprocess_image_from_bytes(image_bytes)
        f_audio = preprocess_audio_from_bytes(audio_bytes)
    except Exception as e:
        prediction_errors.inc()
        raise e
    start = time.time()
    prediction = fusion_model.predict([f_img, f_audio])
    prediction_duration.observe(time.time() - start)
    label = "Chien" if prediction[0][0] > threshold else "Chat"
    return {"prediction": label, "confidence": float(prediction[0][0]), "used_threshold": threshold}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.post("/token", tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    return {"access_token": "dummy_token", "token_type": "bearer"}

@router.get("/protected", tags=["Authentication"])
async def protected_route(token: str = Depends(oauth2_scheme)):
    if token != "dummy_token":
        raise HTTPException(status_code=401, detail="Token invalide")
    return {"message": "Accès autorisé"}

@router.get("/force-error", response_class=PlainTextResponse)
def trigger_error_and_show_report():
    raise ValueError("❌ Erreur volontaire pour test MCO - HTTP 500")

