#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tqdm.keras import TqdmCallback

# --- 📂 Config ---
MAPPING_CSV = "data/data_fusion_model/fusion_mapping.csv"
OLD_MODEL_PATH = "models/image_audio_fusion_new_model.h5"
NEW_MODEL_PATH = "models/image_audio_fusion_model_retrained.h5"

# --- 📌 Vérification du fichier CSV ---
if not os.path.exists(MAPPING_CSV):
    raise FileNotFoundError(f"❌ Fichier manquant: {MAPPING_CSV}. Vérifiez son existence.")

# --- 📌 Prétraitement ---
def preprocess_image(image_path):
    """Charge et pré-traite une image."""
    if not os.path.exists(image_path):
        print(f"⚠️ Image introuvable : {image_path}")
        return None
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64)) / 255.0
    return img.reshape(64, 64, 1)

def preprocess_audio(audio_path):
    """Charge et pré-traite un spectrogramme audio."""
    spec_path = audio_path.replace("cleaned", "spectrograms").replace(".wav", ".png")
    if not os.path.exists(spec_path):
        print(f"⚠️ Spectrogramme introuvable : {spec_path}")
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
    missing_entries = 0
    
    for _, row in df.iterrows():
        img = preprocess_image(row["image_path"])
        aud = preprocess_audio(row["audio_path"])
        if img is None or aud is None:
            missing_entries += 1
            continue
        X_images.append(img)
        X_audio.append(aud)
        y_labels.append(row["label"])
    
    print(f"✅ Données chargées: {len(X_images)} paires valides.")
    if missing_entries > 0:
        print(f"⚠️ {missing_entries} entrées ignorées à cause de fichiers manquants.")
    
    return np.array(X_images), np.array(X_audio), np.array(y_labels)

# --- 📌 Chargement et modification du modèle ---
def load_and_modify_model():
    """Charge l'ancien modèle et ajoute une nouvelle couche pour le réentraînement."""
    old_model = load_model(OLD_MODEL_PATH)
    
    for layer in old_model.layers:
        layer.trainable = False  # On gèle les couches pré-entraînées
    
    # Extraction de la sortie de la dernière couche cachée
    penultimate_output = old_model.layers[-2].output

    # Nouvelle couche de réentraînement
    hidden = Dense(128, activation="relu", name="retrain_dense_1")(penultimate_output)
    hidden = Dropout(0.2, name="retrain_dropout_1")(hidden)
    new_output = Dense(3, activation="softmax", name="retrain_output")(hidden)

    # Création du modèle modifié
    new_model = Model(inputs=old_model.input, outputs=new_output)
    new_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    print("✅ Modèle chargé et modifié avec succès.")
    return new_model

# --- 📌 Entraînement du modèle ---
def train_model(new_model, X_images, X_audio, y_labels):
    """Entraîne le modèle réentraîné."""
    X_train_img, X_val_img, X_train_audio, X_val_audio, y_train, y_val = train_test_split(
        X_images, X_audio, y_labels, test_size=0.2, random_state=42
    )
    
    class_weights = {0: 1.0, 1: 1.0, 2: 2.0}
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        TqdmCallback()
    ]
    
    print("🚀 Début de l'entraînement du modèle réentraîné...")
    
    new_model.fit(
        [X_train_img, X_train_audio], y_train,
        validation_data=([X_val_img, X_val_audio], y_val),
        epochs=10, batch_size=16, class_weight=class_weights,
        callbacks=callbacks
    )
    
    print("✅ Entraînement terminé avec succès.")
    return new_model

# --- 📌 Programme principal ---
def main():
    """Charge les données, modifie le modèle et l'entraîne."""
    X_images, X_audio, y_labels = load_data()
    new_model = load_and_modify_model()
    trained_model = train_model(new_model, X_images, X_audio, y_labels)
    
    # 🔥 Sauvegarde en `.h5`
    trained_model.save(NEW_MODEL_PATH)
    print(f"✅ Modèle réentraîné et sauvegardé sous {NEW_MODEL_PATH}")

if __name__ == "__main__":
    main()
