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
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm.keras import TqdmCallback
from sklearn.model_selection import train_test_split

# --- 📂 Configuration et chemins ---
MAPPING_CSV = "data/data_fusion_model/fusion_mapping.csv"
MODEL_PATH = "models/fusion.h5"

image_model = tf.keras.models.load_model("models/image.keras")
audio_model = tf.keras.models.load_model("models/audio.keras")
print(image_model.summary())
print(audio_model.summary())

# --- 📌 Fonctions de prétraitement ---
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64)) / 255.0
    return img.reshape(64, 64, 1)

def preprocess_audio(audio_path):
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

# --- 📌 Chargement des modèles pré-entraînés individuels ---
def load_pretrained_models():
    print("🔍 Chargement des modèles individuels pré-entraînés...")

    # Charger les modèles
    image_model = tf.keras.models.load_model("models/image.keras")
    audio_model = tf.keras.models.load_model("models/audio.keras")

    # 🔹 Forcer un appel aux modèles pour s'assurer qu'ils sont initialisés
    dummy_input_image = tf.zeros((1, 64, 64, 1))
    dummy_input_audio = tf.zeros((1, 64, 64, 1))
    
    image_model(dummy_input_image)
    audio_model(dummy_input_audio)

    # 🔹 Extraction des couches Flatten
    flatten_image_layer = next((layer for layer in image_model.layers if isinstance(layer, Flatten)), None)
    flatten_audio_layer = next((layer for layer in audio_model.layers if isinstance(layer, Flatten)), None)

    if flatten_image_layer is None or flatten_audio_layer is None:
        raise ValueError("❌ Erreur: Impossible de trouver une couche Flatten dans les modèles.")

    # 🔹 Création des modèles de feature extraction
    image_input = Input(shape=(64, 64, 1), name="image_input")
    audio_input = Input(shape=(64, 64, 1), name="audio_input")

    image_feature_output = flatten_image_layer(image_model(image_input))
    audio_feature_output = flatten_audio_layer(audio_model(audio_input))

    image_feature_model = Model(inputs=image_input, outputs=image_feature_output, name="image_feature_extractor")
    audio_feature_model = Model(inputs=audio_input, outputs=audio_feature_output, name="audio_feature_extractor")

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
    fusion_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return fusion_model

# --- 📌 Entraînement du modèle ---
def train_fusion_model(fusion_model, X_images, X_audio, y_labels):
    X_train_img, X_val_img, X_train_audio, X_val_audio, y_train, y_val = train_test_split(
        X_images, X_audio, y_labels, test_size=0.2, random_state=42
    )
    
    class_weights = {0: 1.0, 1: 1.0, 2: 2.0}
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        TqdmCallback(verbose=1)
    ]
    
    print("🚀 Entraînement du modèle fusionné...")
    fusion_model.fit(
        [X_train_img, X_train_audio], y_train,
        validation_data=([X_val_img, X_val_audio], y_val),
        epochs=10, batch_size=16, callbacks=callbacks,
        class_weight=class_weights
    )
    return fusion_model

# --- 📌 Sauvegarde du modèle ---
def save_model_h5(model, filename=MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    model.save(filename)
    print(f"✅ Modèle sauvegardé en {filename}")

# --- 📌 Programme principal ---
def main():
    X_images, X_audio, y_labels = load_data()
    image_feature_model, audio_feature_model = load_pretrained_models()
    fusion_model = build_fusion_model(image_feature_model, audio_feature_model)
    trained_model = train_fusion_model(fusion_model, X_images, X_audio, y_labels)
    save_model_h5(trained_model)

if __name__ == "__main__":
    main()
