import numpy as np
import tensorflow as tf
import os
from config import MODEL_PATHS

def load_model(path):
    """
    Charge un modèle depuis le chemin spécifié.
    """
    assert os.path.exists(path), f"Le modèle n'existe pas: {path}"
    return tf.keras.models.load_model(path)

import numpy as np

def test_image_model_inference():
    # Chargement du modèle (supposons qu'il soit déjà chargé)
    model = load_model(MODEL_PATHS["image"])
    
    # Crée un dummy input avec la forme attendue : 1 image, 64x64 pixels, 1 canal (niveaux de gris)
    dummy_input = np.zeros((1, 64, 64, 1), dtype=np.float32)
    
    # Lancer l'inférence
    prediction = model.predict(dummy_input)
    
    # Effectuer les assertions nécessaires, par exemple vérifier le type de la prédiction
    assert prediction is not None
    print("Test de prédiction sur dummy input réussi.")

def test_audio_model_inference():
    """Vérifie que le modèle audio peut effectuer une prédiction."""
    model = load_model(MODEL_PATHS["audio"])
    # Supposons que le modèle attend une entrée de taille 64x64x1 (exemple pour un spectrogramme)
    dummy_input = np.random.rand(1, 64, 64, 1).astype(np.float32)
    prediction = model.predict(dummy_input)
    assert prediction.shape[0] == 1, "Le modèle doit renvoyer une prédiction pour 1 échantillon"
    assert np.all(prediction >= 0) and np.all(prediction <= 1), "La sortie doit être normalisée"

def test_fusion_model_inference():
    """Vérifie que le modèle de fusion combine correctement les deux vecteurs de caractéristiques."""
    model = load_model(MODEL_PATHS["fusion"])
    # Supposons que le modèle de fusion attend deux vecteurs de taille (1,256)
    dummy_input1 = np.random.rand(1, 256).astype(np.float32)
    dummy_input2 = np.random.rand(1, 256).astype(np.float32)
    prediction = model.predict([dummy_input1, dummy_input2])
    # Par exemple, le modèle de fusion doit renvoyer une prédiction sur 3 classes
    assert prediction.shape[0] == 1, "La fusion doit renvoyer une prédiction pour 1 échantillon"
    # Vous pouvez ajuster ce test en fonction du nombre de classes attendu
    assert prediction.shape[1] == 3, "Le modèle de fusion doit renvoyer une sortie à 3 dimensions"

if __name__ == "__main__":
    # Pour exécuter les tests directement
    test_image_model_inference()
    test_audio_model_inference()
    test_fusion_model_inference()
    print("Tous les tests de modèle se sont exécutés avec succès.")
