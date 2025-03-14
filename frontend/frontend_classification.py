import os
import random
import streamlit as st
import requests

# URL de base de l'API (à adapter si nécessaire)
API_BASE_URL = "http://localhost:8000"

# Dossiers de test pour la sélection aléatoire
TEST_IMAGE_FOLDER = "data/images/cleaned/test_set"
TEST_AUDIO_FOLDER = "data/audio/cleaned/test"

# Fonction pour sélectionner aléatoirement un fichier dans un dossier
def get_random_file(folder):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    if not files:
        return None
    return os.path.join(folder, random.choice(files))

# Initialisation des états de session
if "image_path" not in st.session_state:
    st.session_state["image_path"] = None
if "audio_path" not in st.session_state:
    st.session_state["audio_path"] = None

st.title("🐱🐶 Classification Chat / Chien")
st.write("Sélectionnez ou uploadez une image et un fichier audio pour tester la classification via l'API FastAPI.")

col1, col2 = st.columns(2)

# --- Gestion de l'image ---
with col1:
    st.subheader("Image")
    image_category = st.radio("Catégorie d'image :", ["Chat", "Chien"], key="image_cat", index=0)
    
    # Option d'upload d'image
    uploaded_image = st.file_uploader("Uploader une image", type=["jpg", "jpeg", "png"], key="upload_image")
    if uploaded_image:
        # Si un fichier est uploadé, on ne conserve pas la sélection aléatoire
        st.session_state["image_path"] = None
        st.image(uploaded_image, caption="Image uploadée", use_column_width=True)
    else:
        # Utiliser la sélection aléatoire si aucun fichier n'est uploadé
        if st.session_state["image_path"] is None:
            folder = os.path.join(TEST_IMAGE_FOLDER, "cats" if image_category == "Chat" else "dogs")
            st.session_state["image_path"] = get_random_file(folder)
        if st.session_state["image_path"]:
            st.image(st.session_state["image_path"], caption="Image sélectionnée", use_column_width=True)
    
    if st.button("🔄 Changer l'image", key="change_img"):
        folder = os.path.join(TEST_IMAGE_FOLDER, "cats" if image_category == "Chat" else "dogs")
        st.session_state["image_path"] = get_random_file(folder)
        st.experimental_rerun()

# --- Gestion de l'audio ---
with col2:
    st.subheader("Audio")
    audio_category = st.radio("Catégorie de son :", ["Chat", "Chien"], key="audio_cat", index=0)
    
    # Option d'upload d'audio
    uploaded_audio = st.file_uploader("Uploader un fichier audio", type=["wav"], key="upload_audio")
    if uploaded_audio:
        st.session_state["audio_path"] = None
        st.audio(uploaded_audio)
    else:
        if st.session_state["audio_path"] is None:
            folder = os.path.join(TEST_AUDIO_FOLDER, "cats" if audio_category == "Chat" else "dogs")
            st.session_state["audio_path"] = get_random_file(folder)
        if st.session_state["audio_path"]:
            st.audio(st.session_state["audio_path"])
    
    if st.button("🔄 Changer l'audio", key="change_audio"):
        folder = os.path.join(TEST_AUDIO_FOLDER, "cats" if audio_category == "Chat" else "dogs")
        st.session_state["audio_path"] = get_random_file(folder)
        st.experimental_rerun()

# --- Bouton de Prédiction ---
if st.button("🔮 Prédire"):
    # Déterminer l'image à utiliser
    if uploaded_image:
        image_file = uploaded_image
    elif st.session_state["image_path"]:
        image_file = open(st.session_state["image_path"], "rb")
    else:
        st.warning("Veuillez sélectionner ou uploader une image.")
        st.stop()

    # Déterminer l'audio à utiliser
    if uploaded_audio:
        audio_file = uploaded_audio
    elif st.session_state["audio_path"]:
        audio_file = open(st.session_state["audio_path"], "rb")
    else:
        st.warning("Veuillez sélectionner ou uploader un fichier audio.")
        st.stop()

    # Préparation des fichiers pour l'API
    files = {
        "image_file": ("image.jpg", image_file, "image/jpeg"),
        "audio_file": ("audio.wav", audio_file, "audio/wav")
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/predict/multimodal", files=files)
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API : {e}")
        st.stop()
    
    if response.status_code == 200:
        result = response.json()
        prediction = result.get("prediction", "Inconnu")
        confidence = result.get("confidence", 0)
        st.success(f"✅ Prédiction : {prediction} (Confiance : {confidence:.2f})")
    else:
        st.error(f"Erreur API {response.status_code} : {response.text}")
