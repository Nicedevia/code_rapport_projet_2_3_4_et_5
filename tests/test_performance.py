import sys
import os
import time
import memory_profiler
import tensorflow as tf

# Ajouter le chemin du dossier parent pour les imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import des fonctions et classes personnalisées
from scripts.newmodel import predict, CustomInputLayer

# Enregistrement des objets personnalisés pour le chargement du modèle
from tensorflow.keras.mixed_precision import Policy

custom_objects = {
    "CustomInputLayer": CustomInputLayer,
    "InputLayer": CustomInputLayer,
    "DTypePolicy": Policy
}

print("chargement du model")

# Chargement du modèle fusionné avec les objets personnalisés
model = tf.keras.models.load_model("models/fusion.h5", custom_objects=custom_objects)

print("model chargé ...")

def test_performance():
    start_time = time.time()
    predict(model, "data/images/cleaned/test_set/cats/cat.16.jpg", "data/audio/cleaned/train/cats/cat_1.wav")
    end_time = time.time()
    elapsed = end_time - start_time
    print("Temps d'inférence mesuré :", elapsed, "secondes")
    assert elapsed < 2.0, f"Erreur: Temps d'inférence trop long: {elapsed} secondes"