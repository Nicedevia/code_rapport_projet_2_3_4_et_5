import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import memory_profiler
import tensorflow as tf
from scripts.newmodel import predict, custom_input_layer

# Enregistrer la classe DTypePolicy dans le scope des objets personnalisés
from tensorflow.keras.mixed_precision import Policy
tf.keras.utils.get_custom_objects()["DTypePolicy"] = Policy

print("chargement du model")

model = tf.keras.models.load_model("models/fusion.h5", custom_objects={"InputLayer": custom_input_layer})
print("model chargé ...")

def test_performance():
    start_time = time.time()
    predict(model, "data/images/cleaned/test_set/cats/cat.16.jpg", "data/audio/cleaned/train/cats/cat_1.wav")
    end_time = time.time()
    elapsed = end_time - start_time
    print("Temps d'inférence mesuré :", elapsed, "secondes")
    # Vous pouvez ajuster le seuil si nécessaire (ici, par exemple, 2 secondes)
    assert elapsed < 2.0, f"Erreur: Temps d'inférence trop long: {elapsed} secondes"
