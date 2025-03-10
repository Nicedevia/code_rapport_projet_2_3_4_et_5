import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pytest

# 🔹 Définition du chemin du modèle (assurez-vous qu'il est bien généré par l'entraînement)
MODEL_PATH = "models/image_audio_fusion_new_model.h5"

# --- 📌 Vérification que le modèle existe avant d'exécuter les tests ---
@pytest.fixture(scope="module")
def model():
    if not os.path.exists(MODEL_PATH):
        pytest.fail(f"❌ Le modèle n'existe pas : {MODEL_PATH}")
    
    print(f"✅ Chargement du modèle : {MODEL_PATH}")
    return load_model(MODEL_PATH, compile=False)

# --- 📌 Fonction de prédiction simplifiée pour le test ---
def predict(model, image_array, audio_array):
    """
    Effectue une prédiction avec le modèle en entrée.
    """
    prediction_proba = model.predict([np.expand_dims(image_array, axis=0), np.expand_dims(audio_array, axis=0)])
    return int(np.argmax(prediction_proba, axis=1)[0])  # 0 = Chat, 1 = Chien, 2 = Erreur

# --- 📌 Cas de test ---
def test_model_prediction(model):
    """
    Teste la prédiction du modèle avec des entrées factices.
    """
    # 🔹 Création de données factices (images et audio en niveaux de gris 64x64)
    fake_image = np.random.rand(64, 64, 1).astype(np.float32)
    fake_audio = np.random.rand(64, 64, 1).astype(np.float32)

    # 🔹 Exécution de la prédiction
    predicted_class = predict(model, fake_image, fake_audio)

    # 🔹 Vérification que la sortie est bien une classe valide (0, 1 ou 2)
    assert predicted_class in [0, 1, 2], f"❌ Classe prédite invalide : {predicted_class}"
    print(f"✅ Prédiction correcte : Classe {predicted_class}")

# --- 📌 Test additionnel pour vérifier si le modèle est bien chargé ---
def test_model_loading():
    """
    Vérifie que le modèle se charge sans erreur.
    """
    try:
        loaded_model = load_model(MODEL_PATH, compile=False)
        assert loaded_model is not None, "❌ Échec du chargement du modèle."
        print("✅ Modèle chargé avec succès !")
    except Exception as e:
        pytest.fail(f"❌ Erreur lors du chargement du modèle : {e}")
