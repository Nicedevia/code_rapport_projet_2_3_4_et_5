import os
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tqdm.keras import TqdmCallback

# --- 📂 Config ---
DATA_DIR = "C:\\Users\\briac\\Desktop\\projet_3\\data_retrain\\training"
MAPPING_CSV = "C:\\Users\\briac\\Desktop\\projet_3\\data_retrain\\mapping.csv"
OLD_MODEL_PATH = "models/image_audio_fusion_new_model.h5"
NEW_MODEL_PATH = "models/image_audio_fusion_model_retrained.h5"

# --- 📌 Création du mapping ---
def create_mapping_csv():
    categories = {"chat": 0, "dog": 1}
    data = []
    
    for label, class_id in categories.items():
        image_dir = os.path.join(DATA_DIR, "images", label)
        audio_dir = os.path.join(DATA_DIR, "audio", label)
        
        if not os.path.exists(image_dir) or not os.path.exists(audio_dir):
            print(f"🚨 Dossier manquant : {image_dir} ou {audio_dir}")
            continue

        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png"))])
        audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".wav")])
        
        for img, aud in zip(image_files, audio_files):
            img_path = os.path.join(image_dir, img)
            aud_path = os.path.join(audio_dir, aud)
            if not os.path.exists(img_path):
                print(f"🚨 Image introuvable : {img_path}")
                continue
            if not os.path.exists(aud_path):
                print(f"🚨 Audio introuvable : {aud_path}")
                continue
            data.append([img_path, aud_path, class_id])
    
    df = pd.DataFrame(data, columns=["image_path", "audio_path", "label"])
    df.to_csv(MAPPING_CSV, index=False)
    print(f"✅ Mapping CSV créé : {MAPPING_CSV} (Total: {len(df)})")

# --- 📌 Prétraitement ---
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64)) / 255.0
    return img.reshape(64, 64, 1)

def generate_spectrogram(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min())
        return cv2.resize(S_db, (64, 64)).reshape(64, 64, 1)
    except Exception as e:
        print(f"🚨 Erreur de chargement audio : {audio_path} ({e})")
        return None

# --- 📌 Chargement des données ---
def load_data():
    df = pd.read_csv(MAPPING_CSV)
    X_images, X_audio, y_labels = [], [], []
    for _, row in df.iterrows():
        img = preprocess_image(row["image_path"])
        aud = generate_spectrogram(row["audio_path"])
        if img is None or aud is None:
            continue
        X_images.append(img)
        X_audio.append(aud)
        y_labels.append(row["label"])
    print(f"✅ Données chargées : {len(X_images)} échantillons valides")
    return np.array(X_images), np.array(X_audio), np.array(y_labels)

# --- 📌 Chargement et modification du modèle ---
def load_and_modify_model():
    old_model = load_model(OLD_MODEL_PATH)
    for layer in old_model.layers:
        layer.trainable = False
    
    penultimate_output = old_model.layers[-2].output
    hidden = Dense(128, activation="relu", name="retrain_dense_1")(penultimate_output)
    hidden = Dropout(0.2, name="retrain_dropout_1")(hidden)
    new_output = Dense(3, activation="softmax", name="retrain_output")(hidden)
    
    new_model = Model(inputs=old_model.input, outputs=new_output)
    new_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return new_model

# --- 📌 Entraînement du modèle ---
def train_model(new_model, X_images, X_audio, y_labels):
    if len(X_images) == 0:
        print("🚨 Aucune donnée valide trouvée. Vérifiez votre dataset !")
        return None
    
    X_train_img, X_val_img, X_train_audio, X_val_audio, y_train, y_val = train_test_split(
        X_images, X_audio, y_labels, test_size=0.2, random_state=42
    )
    
    class_weights = {0: 1.0, 1: 1.0, 2: 2.0}
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        TqdmCallback()
    ]
    
    new_model.fit(
        [X_train_img, X_train_audio], y_train,
        validation_data=([X_val_img, X_val_audio], y_val),
        epochs=10, batch_size=16, class_weight=class_weights,
        callbacks=callbacks
    )
    return new_model

# --- 📌 Programme principal ---
def main():
    create_mapping_csv()
    X_images, X_audio, y_labels = load_data()
    new_model = load_and_modify_model()
    trained_model = train_model(new_model, X_images, X_audio, y_labels)
    if trained_model:
        trained_model.save(NEW_MODEL_PATH)
        print(f"✅ Modèle réentraîné et sauvegardé sous {NEW_MODEL_PATH}")

if __name__ == "__main__":
    main()