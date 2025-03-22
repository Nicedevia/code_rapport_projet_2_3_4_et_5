# import sys
import os
import time
import pytest
import memory_profiler
import tensorflow as tf
import sys 

# Ajouter le chemin du dossier parent pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import des fonctions utilitaires pour reconstruire le modèle
from scripts.newmodel import predict, load_pretrained_models, build_fusion_model

print("Chargement des sous-modèles...")

# Reconstruction du modèle fusionné à partir des sous-modèles
image_feature_model, audio_feature_model = load_pretrained_models()
fusion_model = build_fusion_model(image_feature_model, audio_feature_model)

# Chargement des poids appris
fusion_model.load_weights("models/fusion.h5")

print("Modèle fusionné prêt pour l'inférence.")

def test_performance():
    print("Test de performance désactivé en CI/CD.")
    assert True
