import pickle
import numpy as np
import cv2

# --- Fonctions de prétraitement ---
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64)) / 255.0
    return img.reshape(64, 64, 1)

def preprocess_audio(audio_path):
    spec_path = audio_path.replace("cleaned", "spectrograms").replace(".wav", ".png")
    if not os.path.exists(spec_path):
        print(f"❌ Spectrogramme introuvable pour {audio_path} -> {spec_path}")
        return None
    spec_img = cv2.imread(spec_path, cv2.IMREAD_GRAYSCALE)
    if spec_img is None:
        return None
    spec_img = cv2.resize(spec_img, (64, 64)) / 255.0
    return spec_img.reshape(64, 64, 1)

# --- Chargement du modèle fusionné depuis .pkl ---
with open("models/image_audio_fusion_new_model_v2.pkl", "rb") as f:
    model = pickle.load(f)

print("✅ Modèle fusionné chargé avec succès depuis .pkl !")

# --- Fonction de prédiction ---
def predict(image_path, audio_path):
    img = preprocess_image(image_path)
    aud = preprocess_audio(audio_path)

    if img is None or aud is None:
        raise ValueError("Impossible de charger l'image ou l'audio.")

    prediction_proba = model.predict([np.expand_dims(img, axis=0), np.expand_dims(aud, axis=0)])
    predicted_class = int(np.argmax(prediction_proba, axis=1)[0])

    return predicted_class

# --- Test d'une prédiction ---
image_path = "data/images/cleaned/test_set/cats/cat.16.jpg"
audio_path = "data/audio/cleaned/train/cats/cat_1.wav"

try:
    result = predict(image_path, audio_path)
    print(f"🔍 Prédiction : {result}")
except Exception as e:
    print(f"❌ Erreur : {e}")
