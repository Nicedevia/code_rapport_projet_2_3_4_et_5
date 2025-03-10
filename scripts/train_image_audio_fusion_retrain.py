#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import cv2
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 📌 Paramètres
# -------------------------------
IMG_SIZE = (64, 64)
BATCH_SIZE = 64
EPOCHS = 10

# 📂 Dossiers pour GitHub Actions (mini dataset)
TRAIN_IMAGE_CHAT_DIR = "data_retrain/training/images/chat"
TRAIN_AUDIO_CHAT_DIR = "data_retrain/training/audio/chat"
TRAIN_IMAGE_DOG_DIR = "data_retrain/training/images/dog"
TRAIN_AUDIO_DOG_DIR = "data_retrain/training/audio/dog"

# 📂 Chemin du modèle original et sauvegarde du modèle retrainé
OLD_MODEL_PATH = "models/image_audio_fusion_new_model.h5"
NEW_MODEL_PATH = "models/image_audio_fusion_model_retrained.h5"

# -------------------------------
# 📌 Fonctions de Prétraitement
# -------------------------------
def prepare_image(image_path):
    """Charge et prétraite une image en noir et blanc."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Erreur : Impossible de charger l'image {image_path}")
        return None
    img = cv2.resize(img, IMG_SIZE) / 255.0
    return img.reshape(IMG_SIZE[0], IMG_SIZE[1], 1)

def prepare_audio(audio_path):
    """Charge et prétraite un fichier audio en spectrogramme."""
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        spec_img = cv2.resize(S_db, IMG_SIZE)
        return spec_img.reshape(IMG_SIZE[0], IMG_SIZE[1], 1)
    except Exception as e:
        print(f"❌ Erreur : Impossible de traiter l'audio {audio_path} ({e})")
        return None

# -------------------------------
# 📌 Chargement des fichiers et création des paires
# -------------------------------
def load_files(directory, extensions):
    """Retourne la liste des fichiers valides dans un répertoire."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if any(f.endswith(ext) for ext in extensions)]

def create_data_pairs():
    """Crée des paires Image-Audio pour le réentraînement."""
    img_chat = load_files(TRAIN_IMAGE_CHAT_DIR, [".jpg", ".jpeg", ".png"])
    img_dog = load_files(TRAIN_IMAGE_DOG_DIR, [".jpg", ".jpeg", ".png"])
    aud_chat = load_files(TRAIN_AUDIO_CHAT_DIR, [".wav"])
    aud_dog = load_files(TRAIN_AUDIO_DOG_DIR, [".wav"])

    pairs, labels = [], []

    # Chat (0)
    for img, aud in zip(img_chat, aud_chat):
        pairs.append((img, aud))
        labels.append(0)

    # Chien (1)
    for img, aud in zip(img_dog, aud_dog):
        pairs.append((img, aud))
        labels.append(1)

    # Erreur (2) : chat + chien et chien + chat
    min_length = min(len(img_chat), len(aud_dog))
    for i in range(min_length):
        pairs.append((img_chat[i], aud_dog[i]))
        labels.append(2)

    min_length = min(len(img_dog), len(aud_chat))
    for i in range(min_length):
        pairs.append((img_dog[i], aud_chat[i]))
        labels.append(2)

    return pairs, np.array(labels)

pairs, labels = create_data_pairs()

# -------------------------------
# 📌 Prétraitement des paires
# -------------------------------
X_images, X_audio, y_labels = [], [], []

for img_path, aud_path in pairs:
    img = prepare_image(img_path)
    aud = prepare_audio(aud_path)
    if img is not None and aud is not None:
        X_images.append(img)
        X_audio.append(aud)
        y_labels.append(labels[len(X_images)-1])

X_images = np.array(X_images)
X_audio = np.array(X_audio)
y_labels = np.array(y_labels)

# -------------------------------
# 📌 Séparation train / validation
# -------------------------------
X_img_train, X_img_val, X_aud_train, X_aud_val, y_train, y_val = train_test_split(
    X_images, X_audio, y_labels, test_size=0.2, random_state=42
)

# -------------------------------
# 📌 Chargement du modèle et ajout d'une nouvelle couche
# -------------------------------
old_model = load_model(OLD_MODEL_PATH)

for layer in old_model.layers:
    layer.trainable = False  # On gèle les poids des anciennes couches

penultimate_output = old_model.layers[-2].output

# 🔥 Ajout d'une nouvelle couche cachée pour l'adaptation
retrain_hidden = Dense(64, activation="relu", name="retrain_dense_1")(penultimate_output)
retrain_dropout = Dropout(0.3, name="retrain_dropout_1")(retrain_hidden)

# 📌 Nouvelle couche de sortie
new_output = Dense(3, activation="softmax", name="retrain_output")(retrain_dropout)

new_model = Model(inputs=old_model.input, outputs=new_output)
new_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
new_model.summary()

# -------------------------------
# 📌 Entraînement
# -------------------------------
history = new_model.fit(
    [X_img_train, X_aud_train],
    y_train,
    epochs=EPOCHS,
    validation_data=([X_img_val, X_aud_val], y_val),
    batch_size=BATCH_SIZE
)

# -------------------------------
# 📌 Évaluation et Matrice de Confusion
# -------------------------------
y_pred = np.argmax(new_model.predict([X_img_val, X_aud_val]), axis=1)
cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Chat", "Chien", "Erreur"], yticklabels=["Chat", "Chien", "Erreur"])
plt.xlabel("Prédictions")
plt.ylabel("Vraie Classe")
plt.title("Matrice de Confusion")
plt.show()

# -------------------------------
# 📌 Sauvegarde du modèle en `.h5`
# -------------------------------
new_model.save(NEW_MODEL_PATH)
print(f"✅ Modèle sauvegardé sous {NEW_MODEL_PATH}")

# 📊 Affichage du rapport de classification
print("📊 Rapport de classification :")
print(classification_report(y_val, y_pred, target_names=["Chat", "Chien", "Erreur"]))
