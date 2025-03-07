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

# ------------------------------------------------------------------------------
# 📂 Définition des chemins
FUSION_MODEL_PATH = "models/image_audio_fusion_model_v2.h5"
IMAGE_MODEL_PATH  = "models/image_classifier.h5"
AUDIO_MODEL_PATH  = "models/audio_classifier.h5"

# Dossiers d'exemples (pour la sélection par défaut)
EXEMPLE_IMAGE_FOLDER = "data/images/cleaned/test_set"
EXEMPLE_AUDIO_FOLDER = "data/audio/cleaned/test"

# ------------------------------------------------------------------------------
# 📦 Charger les modèles et créer les extracteurs de features
@st.cache_resource
def load_models():
    fusion_model = tf.keras.models.load_model(FUSION_MODEL_PATH)
    image_model  = tf.keras.models.load_model(IMAGE_MODEL_PATH)
    audio_model  = tf.keras.models.load_model(AUDIO_MODEL_PATH)
    # Création d'extracteurs à partir de la couche avant-dernière
    image_feature_extractor = tf.keras.Model(inputs=image_model.input,
                                               outputs=image_model.layers[-2].output)
    audio_feature_extractor = tf.keras.Model(inputs=audio_model.input,
                                               outputs=audio_model.layers[-2].output)
    return fusion_model, image_feature_extractor, audio_feature_extractor

fusion_model, image_feature_extractor, audio_feature_extractor = load_models()

# ------------------------------------------------------------------------------
# 🎨 Prétraitement de l'image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (64, 64)) / 255.0
    img = img.reshape(1, 64, 64, 1)
    features = image_feature_extractor.predict(img)
    return features

# ------------------------------------------------------------------------------
# 🎵 Prétraitement de l'audio (conversion en spectrogramme)
def preprocess_audio(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050, duration=2)
    except Exception as e:
        st.error(f"Erreur de chargement audio : {e}")
        return None
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    fig, ax = plt.subplots(figsize=(3, 3))
    librosa.display.specshow(spectrogram_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    ax.axis("off")
    temp_img_path = "temp_spectrogram.png"
    fig.savefig(temp_img_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    spec_img = cv2.imread(temp_img_path, cv2.IMREAD_GRAYSCALE)
    if spec_img is None:
        return None
    spec_img = cv2.resize(spec_img, (64, 64)) / 255.0
    spec_img = spec_img.reshape(1, 64, 64, 1)
    features = audio_feature_extractor.predict(spec_img)
    return features

# ------------------------------------------------------------------------------
# 🔊 Lecture d'un son avec Pygame
def play_audio(audio_path):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()

# ------------------------------------------------------------------------------
# 🎲 Sélection aléatoire d'un fichier dans un dossier
def get_random_file(folder):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if not files:
        return None
    return os.path.join(folder, random.choice(files))

# ------------------------------------------------------------------------------
# 🔒 Sauvegarde d'un fichier uploadé (unique si nouveau)
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
# Initialisation des états de session (avec des clés distinctes)
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
# Interface Utilisateur (Disposition en deux colonnes)
st.title("🐱🐶 Classification Chat / Chien")
st.header("Uploader et Tester vos Fichiers")

cols = st.columns(2)

# --- Colonne de gauche : Image ---
with cols[0]:
    st.subheader("Image")
    # Choix de la catégorie (icône sans texte)
    image_cat_choice = st.radio("", ["Chat", "Chien"], key="image_cat", horizontal=True)
    # Upload de l'image
    uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"], key="upload_image", label_visibility="collapsed", help="Uploader votre image")
    if uploaded_image is not None:
        # Si l'image uploadée est différente de celle précédemment sauvegardée
        if st.session_state.get("uploaded_image_name") != uploaded_image.name:
            saved_image_path = save_uploaded_file(uploaded_image, image_cat_choice, file_type="image")
            st.session_state["uploaded_image_path"] = saved_image_path
            st.session_state["uploaded_image_name"] = uploaded_image.name
            st.session_state["test_image_path"] = saved_image_path
            st.success("Téléchargement OK")
    # Sinon, si aucun fichier n'est disponible, utiliser un exemple
    if st.session_state["test_image_path"] is None:
        folder = os.path.join(EXEMPLE_IMAGE_FOLDER, "cats" if image_cat_choice == "Chat" else "dogs")
        st.session_state["test_image_path"] = get_random_file(folder)
    st.image(st.session_state["test_image_path"], caption="Image sélectionnée", width=200)
    if st.button("🔄 Changer l'image"):
        folder = os.path.join(EXEMPLE_IMAGE_FOLDER, "cats" if image_cat_choice == "Chat" else "dogs")
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
        folder = os.path.join(EXEMPLE_AUDIO_FOLDER, "cats" if audio_cat_choice == "Chat" else "dogs")
        st.session_state["test_audio_path"] = get_random_file(folder)
    st.audio(st.session_state["test_audio_path"])
    if st.button("🔄 Changer l'audio"):
        folder = os.path.join(EXEMPLE_AUDIO_FOLDER, "cats" if audio_cat_choice == "Chat" else "dogs")
        st.session_state["test_audio_path"] = get_random_file(folder)
        st.experimental_rerun()

# --- Bouton de Prédiction ---
st.markdown("---")
st.header("Prédiction")
if st.button("🔮 Prédire"):
    if not st.session_state["test_image_path"] or not st.session_state["test_audio_path"]:
        st.warning("⚠️ Sélectionnez une image et un audio.")
    else:
        X_image = preprocess_image(st.session_state["test_image_path"])
        X_audio = preprocess_audio(st.session_state["test_audio_path"])
        if X_image is None or X_audio is None:
            st.error("Erreur lors du prétraitement.")
        else:
            prediction = fusion_model.predict([X_image, X_audio])
            class_labels = ["🐱 Chat", "🐶 Chien", "❌ Erreur"]
            class_index = np.argmax(prediction)
            confidence = f"{np.max(prediction) * 100:.2f}%"
            st.success(f"✅ Prédiction : {class_labels[class_index]} (Confiance : {confidence})")
