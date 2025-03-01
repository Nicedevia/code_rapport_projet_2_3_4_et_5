from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import numpy as np
import io
import cv2
import librosa
import soundfile as sf
import tensorflow as tf
from PIL import Image
from .model_loader import load_image_model, load_audio_model, load_fusion_model
from config import DEFAULT_THRESHOLD

router = APIRouter()

# Charger les modèles et créer les extracteurs de features à partir des modèles d'image et d'audio
image_model = load_image_model()
audio_model = load_audio_model()
fusion_model = load_fusion_model()

# On crée les extracteurs en utilisant la couche avant-dernière de chaque modèle
image_extractor = tf.keras.Model(inputs=image_model.input, outputs=image_model.layers[-2].output)
audio_extractor = tf.keras.Model(inputs=audio_model.input, outputs=audio_model.layers[-2].output)

# --- Fonctions de Prétraitement ---

def preprocess_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Prétraite une image à partir des bytes :
      - Décode l'image en niveaux de gris.
      - Redimensionne à 64x64 pixels, normalise.
      - Reformate en (1, 64, 64, 1).
      - Extrait les features via l'image_extractor.
    """
    # Convertir les bytes en tableau numpy via OpenCV
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail="Image invalide")
    img = cv2.resize(img, (64, 64)) / 255.0
    img = img.reshape(1, 64, 64, 1)
    features = image_extractor.predict(img)
    return features

def preprocess_audio_from_bytes(audio_bytes: bytes) -> np.ndarray:
    """
    Prétraite un fichier audio à partir des bytes :
      - Charge le signal audio depuis un flux BytesIO avec librosa.
      - Calcule le melspectrogramme (n_mels=128) et convertit en dB.
      - Redimensionne le spectrogramme à 64x64 et normalise.
      - Reformate en (1, 64, 64, 1).
      - Extrait les features via l'audio_extractor.
    """
    try:
        audio_stream = io.BytesIO(audio_bytes)
        y, sr = librosa.load(audio_stream, sr=22050, duration=2)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Audio invalide") from e

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    # Redimensionner à 64x64
    spec_img = cv2.resize(S_db, (64, 64))
    spec_img = (spec_img - spec_img.min()) / (spec_img.max() - spec_img.min())
    spec_img = spec_img.reshape(1, 64, 64, 1)
    features = audio_extractor.predict(spec_img)
    return features

# --- Endpoints de Prédiction ---

@router.post("/predict/multimodal", tags=["Prediction"])
async def predict_multimodal(
    image_file: UploadFile = File(..., description="Fichier image (JPEG ou PNG)"),
    audio_file: UploadFile = File(..., description="Fichier audio (WAV)"),
    threshold: float = Query(DEFAULT_THRESHOLD, ge=0, le=1, description="Seuil pour la classification (0 à 1)")
):
    """
    Prédit la classe à partir d'une image et d'un fichier audio (fusion multimodale).
    """
    if image_file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Format d'image non supporté")
    if audio_file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Format audio non supporté")
    
    image_bytes = await image_file.read()
    audio_bytes = await audio_file.read()
    
    features_image = preprocess_image_from_bytes(image_bytes)
    features_audio = preprocess_audio_from_bytes(audio_bytes)
    
    # Appel du modèle de fusion sur les vecteurs de caractéristiques extraits
    prediction = fusion_model.predict([features_image, features_audio])
    label = "Chien" if prediction[0][0] > threshold else "Chat"
    confidence = float(prediction[0][0])
    return {"prediction": label, "confidence": confidence, "used_threshold": threshold}


# --- Authentification et Sécurisation ---

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@router.post("/token", tags=["Authentication"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Génère un token d'accès.
    Pour la démo, toute paire utilisateur/mot de passe renvoie un token fictif.
    """
    return {"access_token": "dummy_token", "token_type": "bearer"}

@router.get("/protected", tags=["Authentication"])
async def protected_route(token: str = Depends(oauth2_scheme)):
    """
    Endpoint protégé nécessitant une authentification.
    """
    if token != "dummy_token":
        raise HTTPException(status_code=401, detail="Token invalide")
    return {"message": "Accès autorisé"}
