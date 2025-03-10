import sys
import os

import pytest
import numpy as np
# Ajoute le chemin du projet dans les variables d'environnement Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.newmodel import preprocess_image, preprocess_audio  # Import de tes fonctions



import sys
import os

import pytest
import numpy as np
# Ajoute le chemin du projet dans les variables d'environnement Python
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.newmodel import preprocess_image, preprocess_audio  # Import de tes fonctions

def test_preprocess_image():
    img = preprocess_image("data/images/cleaned/test_set/cats/cat.16.jpg")
    assert img is not None, "Erreur: Image non chargée"
    print(f"Image shape: {img.shape}")  # Debugging print statement
    assert img.shape == (64, 64, 1), f"Erreur: Format d'image incorrect, obtenu: {img.shape}"

def test_preprocess_audio():
    spec = preprocess_audio("data/audio/cleaned/train/cats/cat_1.wav")
    assert spec is not None, "Erreur: Audio non chargé"
    print(f"Audio shape: {spec.shape}")  # Debugging print statement
    assert spec.shape == (64, 64, 1), f"Erreur: Format audio incorrect, obtenu: {spec.shape}"