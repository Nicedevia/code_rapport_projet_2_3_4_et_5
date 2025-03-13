import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import memory_profiler
from scripts.newmodel import predict
import tensorflow as tf

from scripts.newmodel import predict
import tensorflow as tf

def custom_input_layer(*args, **kwargs):
    # Extraire et supprimer batch_shape de kwargs
    batch_shape = kwargs.pop("batch_shape", None)
    if batch_shape is not None:
        # Utiliser la forme sans la dimension batch (la première dimension)
        kwargs["shape"] = tuple(batch_shape[1:])
    return tf.keras.layers.InputLayer(*args, **kwargs)

print("chargement du model")

model = tf.keras.models.load_model("models/fusion.h5", custom_objects={"InputLayer": custom_input_layer})

print("model chargé ...")

def test_performance():
    start_time = time.time()
    predict(model, "data/images/cleaned/test_set/cats/cat.16.jpg", "data/audio/cleaned/train/cats/cat_1.wav")
    end_time = time.time()
    assert (end_time - start_time) < 1.0, "Erreur: Temps d'inférence trop long"

