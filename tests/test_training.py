import pytest
import numpy as np
from scripts.train_model import train_model  # Fonction d'entraînement

def test_training():
    history = train_model(epochs=2)  # On réduit les epochs pour le test
    assert history.history["accuracy"][-1] > 0.6, "Erreur: Accuracy trop basse"
    assert history.history["loss"][-1] < history.history["loss"][0], "Erreur: Le modèle n'apprend pas"

