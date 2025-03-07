import time
import memory_profiler
from scripts.predict import predict
import tensorflow as tf

model = tf.keras.models.load_model("models/image_audio_fusion_model_v5.keras")

def test_performance():
    start_time = time.time()
    predict(model, "data/images/test/cat.jpg", "data/audio/test/cat.wav")
    end_time = time.time()
    
    assert (end_time - start_time) < 1.0, "Erreur: Temps d'inférence trop long"

    mem_usage = memory_profiler.memory_usage()
    assert max(mem_usage) < 500, "Erreur: Trop de mémoire utilisée"
