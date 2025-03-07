import pytest
import numpy as np
from scripts.preprocess import preprocess_image, preprocess_audio  # Import de tes fonctions

def test_preprocess_image():
    img = preprocess_image("data/images/test/cat.jpg")
    assert img is not None, "Erreur: Image non chargée"
    assert img.shape == (64, 64, 1), "Erreur: Format d'image incorrect"

def test_preprocess_audio():
    spec = preprocess_audio("data/audio/test/cat.wav")
    assert spec is not None, "Erreur: Audio non chargé"
    assert spec.shape == (64, 64, 1), "Erreur: Format audio incorrect"
