import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import memory_profiler
from scripts.newmodel import predict
import tensorflow as tf

print("chargement du model")

model = tf.keras.models.load_model("models/image_audio_fusion_new_model.h5")

print("model chargé ...")

def test_performance():
    start_time = time.time()
    predict(model, "data/images/cleaned/test_set/cats/cat.16.jpg", "data/audio/cleaned/train/cats/cat_1.wav")
    end_time = time.time()
    assert (end_time - start_time) < 1.0, "Erreur: Temps d'inférence trop long"

    # mem_usage = memory_profiler.memory_usage()
    # assert max(mem_usage) < 2000, "Erreur: Trop de mémoire utilisée"
