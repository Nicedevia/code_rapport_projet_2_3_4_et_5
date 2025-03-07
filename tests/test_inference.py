import pytest
import tensorflow as tf
from scripts.predict import predict  # Fonction de prédiction

model = tf.keras.models.load_model("models/image_audio_fusion_model_v5.keras")

def test_prediction():
    result = predict(model, "data/images/test/cat.jpg", "data/audio/test/cat.wav")
    assert result in [0, 1, 2], "Erreur: Prédiction invalide"
