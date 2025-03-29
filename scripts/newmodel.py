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
from tensorflow.keras.mixed_precision import Policy

# --- Enregistrer notre InputLayer personnalisé ---
class CustomInputLayer(tf.keras.layers.InputLayer):
    def __init__(self, *args, **kwargs):
        if "batch_shape" in kwargs:
            batch_shape = kwargs.pop("batch_shape")
            kwargs["batch_input_shape"] = tuple(batch_shape)  # Correction ici
        super().__init__(*args, **kwargs)

# --- Configuration et chemins ---
MAPPING_CSV = "data/data_fusion_model/fusion_mapping.csv"
FUSION_MODEL_PATH = "models/fusion.h5"
IMAGE_MODEL_PATH = "models/image.keras"
AUDIO_MODEL_PATH = "models/audio.keras"

# --- Fonctions de prétraitement ---
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

# --- Chargement des données ---
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

# --- Re-sauvegarde des modèles individuels avec input défini ---
def re_save_individual_models():
    print("🔄 Re-sauvegarde des modèles IMAGE et AUDIO avec input défini...")
    custom_objects = {
    "InputLayer": CustomInputLayer,
    "CustomInputLayer": CustomInputLayer,
    "input_layer": CustomInputLayer,
    "DTypePolicy": Policy
}
    image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH, custom_objects=custom_objects)
    audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH, custom_objects=custom_objects)

    if isinstance(image_model, tf.keras.Sequential) and not image_model.built:
        image_model.build((None, 64, 64, 1))
    if isinstance(audio_model, tf.keras.Sequential) and not audio_model.built:
        audio_model.build((None, 64, 64, 1))

    dummy_image = tf.zeros((1, 64, 64, 1))
    dummy_audio = tf.zeros((1, 64, 64, 1))
    _ = image_model(dummy_image)
    _ = audio_model(dummy_audio)

    image_model.save(IMAGE_MODEL_PATH)
    audio_model.save(AUDIO_MODEL_PATH)
    print("✅ Modèles IMAGE et AUDIO re-sauvegardés avec input défini.")

# --- Chargement des modèles pré-entraînés individuels ---
def load_pretrained_models():
    print("🔍 Chargement des modèles individuels pré-entraînés...")
    custom_objects = {
    "InputLayer": CustomInputLayer,
    "CustomInputLayer": CustomInputLayer,
    "input_layer": CustomInputLayer,
    "DTypePolicy": Policy
}
    image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH, custom_objects=custom_objects)
    audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH, custom_objects=custom_objects)

    if not image_model.inputs:
        raise ValueError("❌ Le modèle IMAGE n'a pas d'input défini.")
    if not audio_model.inputs:
        raise ValueError("❌ Le modèle AUDIO n'a pas d'input défini.")

    flatten_image_layer = next((layer for layer in image_model.layers if isinstance(layer, Flatten)), None)
    flatten_audio_layer = next((layer for layer in audio_model.layers if isinstance(layer, Flatten)), None)
    if flatten_image_layer is None or flatten_audio_layer is None:
        raise ValueError("❌ Erreur: Impossible de trouver une couche Flatten dans les modèles.")

    image_input = Input(shape=(64, 64, 1), name="image_input")
    audio_input = Input(shape=(64, 64, 1), name="audio_input")
    image_feature_output = flatten_image_layer(image_model(image_input))
    audio_feature_output = flatten_audio_layer(audio_model(audio_input))
    image_feature_model = Model(inputs=image_input, outputs=image_feature_output, name="image_feature_extractor")
    audio_feature_model = Model(inputs=audio_input, outputs=audio_feature_output, name="audio_feature_extractor")
    image_feature_model.trainable = False
    audio_feature_model.trainable = False
    return image_feature_model, audio_feature_model

# --- Création du modèle fusionné ---
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

# --- Entraînement du modèle fusionné ---
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

# --- Sauvegarde du modèle fusionné ---
def save_model_h5(model, filename=FUSION_MODEL_PATH):
    os.makedirs("models", exist_ok=True)
    model.save(filename)
    print(f"✅ Modèle FUSION sauvegardé en {filename}")

# --- Fonction predict pour l'inférence ---
def predict(model, image_path, audio_path):
    img = preprocess_image(image_path)
    aud = preprocess_audio(audio_path)
    if img is None or aud is None:
        return None, None
    img = np.expand_dims(img, axis=0)
    aud = np.expand_dims(aud, axis=0)
    prediction = model.predict([img, aud])
    class_index = int(np.argmax(prediction, axis=1)[0])
    return class_index, prediction

# --- Programme principal ---
def main():
    re_save_individual_models()
    X_images, X_audio, y_labels = load_data()
    image_feature_model, audio_feature_model = load_pretrained_models()
    fusion_model = build_fusion_model(image_feature_model, audio_feature_model)
    trained_model = train_fusion_model(fusion_model, X_images, X_audio, y_labels)
    save_model_h5(trained_model)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def test_model_confusion(model):
    # Charger le CSV de mapping
    df = pd.read_csv(MAPPING_CSV)
    df["label"] = df["label"].astype(int)
    
    # Sélectionner 300 exemples par classe
    sampled_dfs = []
    for label in [0, 1, 2]:
        class_df = df[df["label"] == label]
        if len(class_df) >= 300:
            sampled_dfs.append(class_df.sample(300, random_state=42))
        else:
            print(f"Attention, il n'y a que {len(class_df)} exemples pour la classe {label}")
            sampled_dfs.append(class_df)
    sampled_df = pd.concat(sampled_dfs)
    
    X_images, X_audio, y_true = [], [], []
    for _, row in sampled_df.iterrows():
        img = preprocess_image(row["image_path"])
        aud = preprocess_audio(row["audio_path"])
        # Ne conserver que les exemples où les deux fichiers existent
        if img is None or aud is None:
            continue
        X_images.append(img)
        X_audio.append(aud)
        y_true.append(row["label"])
    
    X_images = np.array(X_images)
    X_audio = np.array(X_audio)
    y_true = np.array(y_true)
    
    # Prédiction
    y_pred_prob = model.predict([X_images, X_audio])
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calcul et affichage de la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Chat", "Chien", "Erreur"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matrice de confusion sur 300 exemples par classe")
    plt.show()

# Exemple d'utilisation après l'entraînement :
if __name__ == "__main__":
    # Exécution du pipeline d'entraînement
    re_save_individual_models()
    X_images, X_audio, y_labels = load_data()
    image_feature_model, audio_feature_model = load_pretrained_models()
    fusion_model = build_fusion_model(image_feature_model, audio_feature_model)
    trained_model = train_fusion_model(fusion_model, X_images, X_audio, y_labels)
    save_model_h5(trained_model)
    
    # Test sur 100 exemples par classe et affichage de la matrice de confusion
    test_model_confusion(trained_model)
