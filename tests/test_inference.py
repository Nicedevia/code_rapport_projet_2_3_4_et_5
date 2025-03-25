import os
import numpy as np
import tensorflow as tf
import pytest

from scripts.newmodel import CustomInputLayer, build_fusion_model, load_pretrained_models

# 🔹 Définition du chemin du modèle
MODEL_PATH = "models/fusion.h5"

# --- 📌 Vérification que le modèle existe avant d'exécuter les tests ---
@pytest.fixture(scope="module")
def model():
    if not os.path.exists(MODEL_PATH):
        pytest.fail(f"❌ Le modèle n'existe pas : {MODEL_PATH}")

    print(f"✅ Reconstruction du modèle puis chargement des poids depuis : {MODEL_PATH}")
    image_model, audio_model = load_pretrained_models()
    fusion_model = build_fusion_model(image_model, audio_model)
    fusion_model.load_weights(MODEL_PATH)
    return fusion_model

# --- 📌 Fonction de prédiction simplifiée pour le test ---
def predict(model, image_array, audio_array):
    """
    Effectue une prédiction avec le modèle en entrée.
    """
    prediction_proba = model.predict([
        np.expand_dims(image_array, axis=0),
        np.expand_dims(audio_array, axis=0)
    ])
    return int(np.argmax(prediction_proba, axis=1)[0])  # 0 = Chat, 1 = Chien, 2 = Erreur

# --- 📌 Cas de test ---
def test_model_prediction(model):
    """
    Teste la prédiction du modèle avec des entrées factices.
    """
    fake_image = np.random.rand(64, 64, 1).astype(np.float32)
    fake_audio = np.random.rand(64, 64, 1).astype(np.float32)
    predicted_class = predict(model, fake_image, fake_audio)
    assert predicted_class in [0, 1, 2], f"❌ Classe prédite invalide : {predicted_class}"
    print(f"✅ Prédiction correcte : Classe {predicted_class}")

# --- 📌 Test additionnel pour vérifier si le modèle est bien chargé ---
def test_model_loading():
    """
    Vérifie que le modèle se charge sans erreur via reconstruction + poids.
    """
    try:
        image_model, audio_model = load_pretrained_models()
        fusion_model = build_fusion_model(image_model, audio_model)
        fusion_model.load_weights(MODEL_PATH)
        assert fusion_model is not None, "❌ Échec de la reconstruction du modèle."
        print("✅ Modèle reconstruit et chargé avec succès !")
    except Exception as e:
        pytest.fail(f"❌ Erreur lors de la reconstruction/chargement du modèle !: {e}")