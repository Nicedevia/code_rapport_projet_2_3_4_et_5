#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm.keras import TqdmCallback
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# --- 📂 Configuration et chemins ---
MAPPING_CSV = "data/data_fusion_model/fusion_mapping.csv"
MODEL_PATH = "models/fusion.h5"  # Sauvegarde en `.h5`

# --- 📌 Fonctions de prétraitement ---
def preprocess_image(image_path):
    """Charge et pré-traite une image en niveau de gris."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64)) / 255.0
    return img.reshape(64, 64, 1)

def preprocess_audio(audio_path):
    """Charge un spectrogramme pré-généré et le pré-traite."""
    spec_path = audio_path.replace("cleaned", "spectrograms").replace(".wav", ".png")
    if not os.path.exists(spec_path):
        print(f"❌ Spectrogramme introuvable pour {audio_path} -> {spec_path}")
        return None
    spec_img = cv2.imread(spec_path, cv2.IMREAD_GRAYSCALE)
    if spec_img is None:
        return None
    spec_img = cv2.resize(spec_img, (64, 64)) / 255.0
    return spec_img.reshape(64, 64, 1)

# --- 📌 Chargement des données ---
def load_data():
    df = pd.read_csv(MAPPING_CSV)
    print(f"🔍 Nombre d'exemples dans le mapping : {len(df)}")

    X_images, X_audio, y_labels = [], [], []
    for _, row in df.iterrows():
        img = preprocess_image(row["image_path"])
        aud = preprocess_audio(row["audio_path"])
        if img is None or aud is None:
            continue
        X_images.append(img)
        X_audio.append(aud)
        y_labels.append(row["label"])
        
    return np.array(X_images), np.array(X_audio), np.array(y_labels)

# --- 📌 Chargement des modèles pré-entraînés ---
def load_pretrained_models():
    print("🔍 Chargement des modèles individuels pré-entraînés...")
    image_model = tf.keras.models.load_model("models/image_classifier_final.keras")
    audio_model = tf.keras.models.load_model("models/audio_classifier_final.keras")
    
    # Extraction des features jusqu'à la couche Flatten
    flatten_image_layer = next(layer for layer in image_model.layers if isinstance(layer, tf.keras.layers.Flatten))
    flatten_audio_layer = next(layer for layer in audio_model.layers if isinstance(layer, tf.keras.layers.Flatten))

    image_feature_model = Model(inputs=image_model.input, outputs=flatten_image_layer.output, name="image_feature_extractor")
    audio_feature_model = Model(inputs=audio_model.input, outputs=flatten_audio_layer.output, name="audio_feature_extractor")

    image_feature_model.trainable = False
    audio_feature_model.trainable = False

    return image_feature_model, audio_feature_model

# --- 📌 Création du modèle fusionné ---
def build_fusion_model(image_feature_model, audio_feature_model):
    image_input = Input(shape=(64, 64, 1), name="image_input")
    audio_input = Input(shape=(64, 64, 1), name="audio_input")

    image_features = image_feature_model(image_input)
    audio_features = audio_feature_model(audio_input)

    combined_features = concatenate([image_features, audio_features], name="fusion_layer")
    fc = Dense(128, activation="relu")(combined_features)
    fc = Dropout(0.3)(fc)
    fc = Dense(64, activation="relu")(fc)
    final_output = Dense(3, activation="softmax", name="output_layer")(fc)

    fusion_model = Model(inputs=[image_input, audio_input], outputs=final_output, name="fusion_model")
    fusion_model.compile(optimizer="adam", 
                         loss="sparse_categorical_crossentropy", 
                         metrics=["accuracy"])

    return fusion_model

# --- 📌 Fonction de prédiction ---
def predict(model, image_path, audio_path):
    """Effectue une prédiction avec le modèle fusionné."""
    img = preprocess_image(image_path)
    aud = preprocess_audio(audio_path)
    if img is None or aud is None:
        return 2  # Classe erreur

    prediction_proba = model.predict([np.expand_dims(img, axis=0), np.expand_dims(aud, axis=0)])
    return int(np.argmax(prediction_proba, axis=1)[0])

# --- 📌 Entraînement ---
def train_fusion_model(fusion_model, X_images, X_audio, y_labels):
    X_train_img, X_val_img, X_train_audio, X_val_audio, y_train, y_val = train_test_split(
        X_images, X_audio, y_labels, test_size=0.2, random_state=42
    )

    # Définition des poids de classe
    class_weights = {0: 1.0, 1: 1.0, 2: 2.0}

    # Callbacks personnalisés
    class LoggingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"Epoch {epoch+1} - Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        LoggingCallback(),
        TqdmCallback(verbose=1)
    ]

    print("🚀 Entraînement du modèle fusionné...")
    fusion_model.fit([X_train_img, X_train_audio], y_train,
                     validation_data=([X_val_img, X_val_audio], y_val),
                     epochs=10, batch_size=16, callbacks=callbacks,
                     class_weight=class_weights)  # ✅ Ajout des poids de classe

    return fusion_model

# --- 📌 Sauvegarde en `.h5` ---
def save_model_h5(model, filename=MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    model.save(filename)
    print(f"✅ Modèle sauvegardé en {filename}")

# --- 📌 Programme principal ---
def main():
    # Chargement des données
    X_images, X_audio, y_labels = load_data()

    # Chargement des modèles pré-entraînés
    image_feature_model, audio_feature_model = load_pretrained_models()

    # Création du modèle fusionné
    fusion_model = build_fusion_model(image_feature_model, audio_feature_model)
    fusion_model.summary()

    # Entraînement
    trained_model = train_fusion_model(fusion_model, X_images, X_audio, y_labels)

    # Sauvegarde du modèle en `.h5`
    save_model_h5(trained_model)

if __name__ == "__main__":
    main()
