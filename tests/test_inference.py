import sys
import os
# Ajoute le chemin du projet dans les variables d'environnement Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import tensorflow as tf
from scripts.newmodel import predict  # Fonction de prédiction


model = tf.keras.models.load_model("models/image_audio_fusion_model_v10.keras")

def test_prediction():
    result = predict(model, "data/images/test/cat.jpg", "data/audio/test/cat.wav")
    assert result in [0, 1, 2], "Erreur: Prédiction invalide"
