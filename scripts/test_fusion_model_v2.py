#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# ----- Configuration -----
# Chemin du mapping de test (ce fichier doit contenir les colonnes "image_path" et "audio_path")
TEST_CSV = r"data/audio/test_image_audio_mapping.csv"
# Chemin vers le modèle fusionné entraîné
FUSION_MODEL_PATH = r"models/image_audio_fusion_model_v5.keras"

# ----- Fonctions de prétraitement -----
def preprocess_image(image_path):
    """Charge et prétraite une image en niveaux de gris."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64)) / 255.0
    return img.reshape(1, 64, 64, 1)

def preprocess_audio(audio_path):
    r"""
    Charge le spectrogramme pré-généré correspondant au fichier audio.
    On suppose que le chemin audio est de la forme :
        ...\cleaned\audio\...
    et que le spectrogramme est pré-généré dans :
        ...\spectrograms\...
    avec l'extension .wav remplacée par .png.
    """
    spec_path = audio_path.replace("cleaned", "spectrograms").replace(".wav", ".png")
    if not os.path.exists(spec_path):
        print(f"❌ Spectrogramme introuvable pour {audio_path} -> {spec_path}")
        return None
    spec_img = cv2.imread(spec_path, cv2.IMREAD_GRAYSCALE)
    if spec_img is None:
        return None
    spec_img = cv2.resize(spec_img, (64, 64)) / 255.0
    return spec_img.reshape(1, 64, 64, 1)

# ----- Chargement du mapping de test -----
print("🔄 Chargement du mapping de test...")
test_df = pd.read_csv(TEST_CSV)

X_images, X_audio = [], []
y_true = []  # Étiquette dérivée du nom de fichier

for _, row in test_df.iterrows():
    image_path = row["image_path"]
    audio_path = row["audio_path"]
    if not os.path.exists(image_path) or not os.path.exists(audio_path):
        continue
    img = preprocess_image(image_path)
    aud = preprocess_audio(audio_path)
    if img is None or aud is None:
        continue
    X_images.append(img)
    X_audio.append(aud)
    # Définir le label à partir des chemins :
    # Si les deux chemins contiennent "cats" -> label 0, "dogs" -> label 1, sinon -> label 2 (Erreur)
    if "cats" in image_path.lower() and "cats" in audio_path.lower():
        y_true.append(0)
    elif "dogs" in image_path.lower() and "dogs" in audio_path.lower():
        y_true.append(1)
    else:
        y_true.append(2)

if len(X_images) == 0 or len(X_audio) == 0:
    print("❌ Aucun échantillon de test valide trouvé.")
    exit()

# Conversion des listes en tenseurs
X_images = np.vstack(X_images)
X_audio = np.vstack(X_audio)
y_true = np.array(y_true)

print(f"🔄 Nombre d'échantillons de test utilisés : {X_images.shape[0]}")

# ----- Chargement du modèle fusionné -----
print("🔄 Chargement du modèle fusionné...")
fusion_model = tf.keras.models.load_model(FUSION_MODEL_PATH)
print("✅ Modèle fusionné chargé.")

# ----- Prédictions -----
print("🔄 Prédictions du modèle fusionné...")
y_pred_probs = fusion_model.predict([X_images, X_audio])
y_pred = np.argmax(y_pred_probs, axis=1)

# ----- Évaluation -----
label_names = {0: "Chat", 1: "Chien", 2: "Erreur"}
target_names = [label_names[0], label_names[1], label_names[2]]

print("\n📌 Rapport de classification :")
print(classification_report(y_true, y_pred, target_names=target_names))

conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Prédictions")
plt.ylabel("Vraies classes")
plt.title("Matrice de Confusion - Modèle Fusionné")
plt.show()

accuracy = np.mean(y_pred == y_true) * 100
print(f"\n🎯 Accuracy du modèle fusionné sur le test: {accuracy:.2f}%")
