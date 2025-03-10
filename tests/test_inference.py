import sys
import os
# Ajoute le chemin du projet dans les variables d'environnement Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import tensorflow as tf
from scripts.newmodel import predict  # Fonction de prédiction


model = tf.keras.models.load_model("models/image_audio_fusion_new_model_v2.keras", compile=False)

def test_prediction():
    result = predict(model, "data/images/cleaned/test_set/cats/cat.16.jpg", "data/audio/cleaned/train/cats/cat_1.wav")
    assert result in [0, 1, 2], "Erreur: Prédiction invalide"
