# api/model_loader.py

import tensorflow as tf

def load_image_model():
    print("🔍 Chargement du modèle IMAGE...")
    model = tf.keras.models.load_model("models/image_classifier.h5", compile=False)
    # Si l'input n'est pas défini, appeler le modèle avec un tenseur fictif pour le construire.
    if not model.inputs:
        dummy_input = tf.zeros((1, 64, 64, 1))
        model(dummy_input)
    print("✅ Modèle IMAGE chargé avec succès :", model.summary())
    return model

def load_audio_model():
    print("🔍 Chargement du modèle AUDIO...")
    model = tf.keras.models.load_model("models/audio_classifier.h5", compile=False)
    print("✅ Modèle AUDIO chargé avec succès :", model.summary())
    return model

def load_fusion_model():
    print("🔍 Chargement du modèle FUSION...")
    model = tf.keras.models.load_model("models/image_audio_fusion_model_v2.h5", compile=False)
    print("✅ Modèle FUSION chargé avec succès :", model.summary())
    return model
