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
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm.keras import TqdmCallback

# --- Configuration et chemins ---
MAPPING_CSV = r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\fusion_mapping.csv"

# --- Fonctions de prétraitement ---
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64)) / 255.0
    return img.reshape(64, 64, 1)

def preprocess_audio(audio_path):
    r"""
    Charge le spectrogramme pré-généré correspondant au fichier audio.
    On suppose que le chemin de l'audio dans le mapping est de la forme :
      ...\cleaned\audio\...\xxx.wav
    et que le spectrogramme est pré-généré dans :
      ...\spectrograms\...\xxx.png
    """
    spec_path = audio_path.replace("cleaned", "spectrograms").replace(".wav", ".png")
    if not os.path.exists(spec_path):
        print(f"❌ Spectrogramme introuvable pour {audio_path} -> {spec_path}")
        return None
    spec_img = cv2.imread(spec_path, cv2.IMREAD_GRAYSCALE)
    if spec_img is None:
        return None
    spec_img = cv2.resize(spec_img, (64, 64)) / 255.0
    return spec_img.reshape(64, 64, 1)

def predict(model, image_path, audio_path):
    """
    Effectue la prédiction en chargeant et prétraitant l'image et l'audio donnés.
    Retourne un entier parmi [0, 1, 2] représentant la classe prédite.
      - 0 : Chat
      - 1 : Chien
      - 2 : Erreur
    """
    # Prétraitement de l'image
    img = preprocess_image(image_path)
    if img is None:
        raise ValueError(f"Impossible de charger l'image à {image_path}")
    
    # Prétraitement de l'audio
    aud = preprocess_audio(audio_path)
    if aud is None:
        raise ValueError(f"Impossible de charger l'audio à {audio_path}")
    
    # Effectuer la prédiction
    prediction_proba = model.predict([np.expand_dims(img, axis=0), np.expand_dims(aud, axis=0)])
    predicted_class = int(np.argmax(prediction_proba, axis=1)[0])
    
    return predicted_class


def main():
    # --- Chargement du mapping ---
    df = pd.read_csv(MAPPING_CSV)
    print(f"Nombre d'exemples dans le mapping : {len(df)}")

    X_images, X_audio, y_labels = [], [], []
    for _, row in df.iterrows():
        img = preprocess_image(row["image_path"])
        aud = preprocess_audio(row["audio_path"])
        if img is None or aud is None:
            continue
        X_images.append(img)
        X_audio.append(aud)
        y_labels.append(row["label"])
        
    X_images = np.array(X_images)
    X_audio = np.array(X_audio)
    y_labels = np.array(y_labels)
    
    print(f"Dataset final : {X_images.shape[0]} exemples")

    # --- Chargement des modèles individuels pré-entraînés ---
    print("Chargement des modèles individuels pré-entraînés...")
    # Remplacer par le chemin correct selon vos sauvegardes
    image_model = tf.keras.models.load_model("models/image_classifier_final.keras")
    audio_model = tf.keras.models.load_model("models/audio_classifier_final.keras")
    print("Modèles individuels chargés.")

    # --- Extraction des features jusqu'à la couche Flatten ---
    # Pour le modèle image
    flatten_image_layer = None
    for layer in image_model.layers:
        if isinstance(layer, tf.keras.layers.Flatten):
            flatten_image_layer = layer
            break
    if flatten_image_layer is None:
        raise ValueError("Aucune couche Flatten trouvée dans le modèle image.")

    image_feature_model = Model(
        inputs=image_model.input,
        outputs=flatten_image_layer.output,
        name="image_feature_extractor"
    )

    # Pour le modèle audio
    flatten_audio_layer = None
    for layer in audio_model.layers:
        if isinstance(layer, tf.keras.layers.Flatten):
            flatten_audio_layer = layer
            break
    if flatten_audio_layer is None:
        raise ValueError("Aucune couche Flatten trouvée dans le modèle audio.")

    audio_feature_model = Model(
        inputs=audio_model.input,
        outputs=flatten_audio_layer.output,
        name="audio_feature_extractor"
    )

    # Optionnel : geler les extracteurs pour se concentrer sur l'entraînement des couches de fusion
    image_feature_model.trainable = False
    audio_feature_model.trainable = False

    # --- Création du modèle fusionné ---
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
    fusion_model.summary()

    # --- Callbacks personnalisés ---
    class LoggingCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            print(f"----- Epoch {epoch+1} started -----")
        def on_epoch_end(self, epoch, logs=None):
            loss = logs.get("loss")
            acc = logs.get("accuracy")
            val_loss = logs.get("val_loss")
            val_acc = logs.get("val_accuracy")
            print(f"----- Epoch {epoch+1} ended: loss={loss:.4f}, accuracy={acc:.4f}, val_loss={val_loss:.4f}, val_accuracy={val_acc:.4f} -----")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        LoggingCallback(),
        TqdmCallback(verbose=1)
    ]

    # --- Entraînement ---
    print("Entraînement du modèle fusionné...")
    history = fusion_model.fit([X_images, X_audio], y_labels,
                               epochs=10, validation_split=0.2, batch_size=16,
                               callbacks=callbacks)

    # Sauvegarde du modèle fusionné
    os.makedirs("models", exist_ok=True)
    fusion_model.save("models/image_audio_fusion_new_model_v2.keras")
    print("Modèle fusionné sauvegardé avec succès !")
    # --- Visualisation des courbes d'entraînement ---
    # Affichage du log loss et de l'accuracy
    plt.figure(figsize=(12,5))

    # Log Loss
    plt.subplot(1,2,1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Courbe du Log Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Courbe de l'Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # --- Evaluation sur un ensemble de validation séparé ---
    # Pour obtenir une matrice de confusion, nous allons reconstruire un ensemble de validation
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, classification_report

    # Séparation en train/validation (20% pour validation)
    X_train_img, X_val_img, X_train_audio, X_val_audio, y_train, y_val = train_test_split(
        X_images, X_audio, y_labels, test_size=0.2, random_state=42
    )

    # Prédictions sur l'ensemble de validation
    y_pred_proba = fusion_model.predict([X_val_img, X_val_audio])
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Affichage des métriques
    print("Classification Report :")
    print(classification_report(y_val, y_pred))

    # Calcul et affichage de la matrice de confusion
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Matrice de Confusion")
    plt.xlabel("Prédictions")
    plt.ylabel("Véritables labels")
    plt.show()

if __name__ == "__main__":
    main()




