import os
import random
import time
import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
import pygame

# Enregistrement global pour BatchNormalization (pour éviter d'éventuelles erreurs de désérialisation)
tf.keras.utils.get_custom_objects()['BatchNormalization'] = tf.keras.layers.BatchNormalization
tf.keras.utils.get_custom_objects()['BatchNormalizationV2'] = tf.keras.layers.BatchNormalization

# --- Définition des chemins ---
MODEL_PATH = "models/fusion.h5"  # Modèle de fusion
EXAMPLE_IMAGE_FOLDER = "data/images/cleaned/test_set"
EXAMPLE_AUDIO_FOLDER = "data/audio/cleaned/test"

# Correction du mapping du module fonctionnel si nécessaire
from tensorflow.python.keras.engine import functional as keras_functional
import sys
sys.modules["keras.src.engine.functional"] = keras_functional

# ------------------------------------------------------------------------------
# Charger le modèle de fusion (mise en cache)
@st.cache_resource
def load_fusion_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    # Forcer la construction du modèle s'il n'a pas ses inputs définis
    if not model.inputs or len(model.inputs) == 0:
        dummy_img = tf.zeros((1, 64, 64, 1))
        dummy_aud = tf.zeros((1, 64, 64, 1))
        _ = model([dummy_img, dummy_aud])
    return model

fusion_model = load_fusion_model()

# ------------------------------------------------------------------------------
# Prétraitement de l'image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        st.error("Erreur lors de la lecture de l'image.")
        return None
    img = cv2.resize(img, (64, 64)) / 255.0
    return img.reshape(64, 64, 1)

# ------------------------------------------------------------------------------
# Prétraitement de l'audio
# Ici, nous générons un melspectrogramme à la volée à partir du fichier audio.
def preprocess_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=2)
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'audio: {e}")
        return None
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    # Normalisation sur [0,1]
    norm_spec = (spectrogram_db - spectrogram_db.min()) / (spectrogram_db.max() - spectrogram_db.min())
    spec_img = cv2.resize(norm_spec, (64, 64))
    return spec_img.reshape(64, 64, 1)

# ------------------------------------------------------------------------------
# Prédiction avec le modèle de fusion
def predict_fusion(image_path, audio_path):
    img = preprocess_image(image_path)
    aud = preprocess_audio(audio_path)
    if img is None or aud is None:
        return None, None
    # Ajout de la dimension batch
    img_batch = np.expand_dims(img, axis=0)
    aud_batch = np.expand_dims(aud, axis=0)
    prediction = fusion_model.predict([img_batch, aud_batch])
    class_index = int(np.argmax(prediction, axis=1)[0])
    return class_index, prediction

# ------------------------------------------------------------------------------
# Sélection aléatoire d'un fichier dans un dossier
def get_random_file(folder):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if not files:
        return None
    return os.path.join(folder, random.choice(files))

# ------------------------------------------------------------------------------
# Sauvegarde d'un fichier uploadé
def save_uploaded_file(uploaded_file, category, file_type="image"):
    base_dir = "data/training"  # Dossier de stockage pour les uploads
    sub_dir = "images" if file_type == "image" else "audio"
    category = category.lower()
    save_dir = os.path.join(base_dir, sub_dir, category)
    os.makedirs(save_dir, exist_ok=True)
    timestamp = int(time.time())
    filename = f"{timestamp}_{uploaded_file.name}"
    file_path = os.path.join(save_dir, filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# ------------------------------------------------------------------------------
# Initialisation des états de session
if "uploaded_image_path" not in st.session_state:
    st.session_state["uploaded_image_path"] = None
if "uploaded_image_name" not in st.session_state:
    st.session_state["uploaded_image_name"] = None
if "test_image_path" not in st.session_state:
    st.session_state["test_image_path"] = None

if "uploaded_audio_path" not in st.session_state:
    st.session_state["uploaded_audio_path"] = None
if "uploaded_audio_name" not in st.session_state:
    st.session_state["uploaded_audio_name"] = None
if "test_audio_path" not in st.session_state:
    st.session_state["test_audio_path"] = None

# ------------------------------------------------------------------------------
# Interface Utilisateur
st.title("🐱🐶 Classification Chat / Chien")
st.header("Uploader et Tester vos Fichiers")

cols = st.columns(2)

# --- Colonne de gauche : Image ---
with cols[0]:
    st.subheader("Image")
    image_cat_choice = st.radio("", ["Chat", "Chien"], key="image_cat", horizontal=True)
    uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"], key="upload_image", label_visibility="collapsed", help="Uploader votre image")
    if uploaded_image is not None:
        if st.session_state.get("uploaded_image_name") != uploaded_image.name:
            saved_image_path = save_uploaded_file(uploaded_image, image_cat_choice, file_type="image")
            st.session_state["uploaded_image_path"] = saved_image_path
            st.session_state["uploaded_image_name"] = uploaded_image.name
            st.session_state["test_image_path"] = saved_image_path
            st.success("Téléchargement OK")
    if st.session_state["test_image_path"] is None:
        folder = os.path.join(EXAMPLE_IMAGE_FOLDER, "cats" if image_cat_choice == "Chat" else "dogs")
        st.session_state["test_image_path"] = get_random_file(folder)
    st.image(st.session_state["test_image_path"], caption="Image sélectionnée", width=200)
    if st.button("🔄 Changer l'image"):
        folder = os.path.join(EXAMPLE_IMAGE_FOLDER, "cats" if image_cat_choice == "Chat" else "dogs")
        st.session_state["test_image_path"] = get_random_file(folder)
        st.experimental_rerun()

# --- Colonne de droite : Audio ---
with cols[1]:
    st.subheader("Audio")
    audio_cat_choice = st.radio("", ["Chat", "Chien"], key="audio_cat", horizontal=True)
    uploaded_audio = st.file_uploader("", type=["wav"], key="upload_audio", label_visibility="collapsed", help="Uploader votre audio")
    if uploaded_audio is not None:
        if st.session_state.get("uploaded_audio_name") != uploaded_audio.name:
            saved_audio_path = save_uploaded_file(uploaded_audio, audio_cat_choice, file_type="audio")
            st.session_state["uploaded_audio_path"] = saved_audio_path
            st.session_state["uploaded_audio_name"] = uploaded_audio.name
            st.session_state["test_audio_path"] = saved_audio_path
            st.success("Téléchargement OK")
    if st.session_state["test_audio_path"] is None:
        folder = os.path.join(EXAMPLE_AUDIO_FOLDER, "cats" if audio_cat_choice == "Chat" else "dogs")
        st.session_state["test_audio_path"] = get_random_file(folder)
    st.audio(st.session_state["test_audio_path"])
    if st.button("🔄 Changer l'audio"):
        folder = os.path.join(EXAMPLE_AUDIO_FOLDER, "cats" if audio_cat_choice == "Chat" else "dogs")
        st.session_state["test_audio_path"] = get_random_file(folder)
        st.experimental_rerun()

st.markdown("---")
st.header("Prédiction")
if st.button("🔮 Prédire"):
    if not st.session_state["test_image_path"] or not st.session_state["test_audio_path"]:
        st.warning("⚠️ Sélectionnez une image et un audio.")
    else:
        class_index, prediction = predict_fusion(st.session_state["test_image_path"], st.session_state["test_audio_path"])
        if class_index is None:
            st.error("Erreur lors de la prédiction.")
        else:
            # Libellés des classes (à adapter si besoin)
            class_labels = ["🐱 Chat", "🐶 Chien", "❌ Erreur"]
            confidence = f"{np.max(prediction) * 100:.2f}%"
            st.success(f"✅ Prédiction : {class_labels[class_index]} (Confiance : {confidence})")
