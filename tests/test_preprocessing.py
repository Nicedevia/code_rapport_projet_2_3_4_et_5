import sys
import os
import pytest
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Ajoute le chemin du projet pour importer les fonctions correctement
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.newmodel import preprocess_image, preprocess_audio  # Import des fonctions

# 📌 Fonction pour générer le spectrogramme à partir d'un fichier audio
def generate_spectrogram(audio_path):
    """ Génère un spectrogramme PNG pour un fichier audio donné """
    spectrogram_path = audio_path.replace(".wav", ".png")  # Convertir en chemin de spectrogramme
    if os.path.exists(spectrogram_path):
        print(f"✅ Spectrogramme déjà existant : {spectrogram_path}")
        return spectrogram_path  # Pas besoin de le générer à nouveau

    print(f"🔄 Génération du spectrogramme pour {audio_path}...")

    try:
        y, sr = librosa.load(audio_path, sr=22050)  # Charger l'audio
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Sauvegarder le spectrogramme
        plt.figure(figsize=(4, 4))
        librosa.display.specshow(S_dB, sr=sr, cmap='gray_r')
        plt.axis('off')
        plt.savefig(spectrogram_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        print(f"✅ Spectrogramme généré : {spectrogram_path}")
        return spectrogram_path
    except Exception as e:
        print(f"❌ Erreur lors de la génération du spectrogramme : {e}")
        return None

@pytest.mark.parametrize("image_path", [
    os.path.abspath("data_sample/images/cats/cat.0.jpg"),
    os.path.abspath("data_sample/images/dogs/dog.0.jpg")
])
def test_preprocess_image(image_path):
    """ Teste le prétraitement des images """
    assert os.path.exists(image_path), f"❌ Fichier image introuvable: {image_path}"

    img = preprocess_image(image_path)
    assert img is not None, f"❌ Erreur: Image non chargée {image_path}"
    assert img.shape == (64, 64, 1), f"❌ Erreur: Format d'image incorrect pour {image_path}"

@pytest.mark.parametrize("audio_path", [
    os.path.abspath("data_sample/audio/cats/cat_1.wav"),
    os.path.abspath("data_sample/audio/dogs/dog_barking_1.wav")
])
def test_preprocess_audio(audio_path):
    """ Teste le prétraitement des fichiers audio """

    # Vérifie que l'audio existe avant de tester
    assert os.path.exists(audio_path), f"❌ Fichier audio introuvable: {audio_path}"

    # Générer le spectrogramme s'il n'existe pas
    spectrogram_path = generate_spectrogram(audio_path)
    assert spectrogram_path is not None, f"❌ Impossible de générer le spectrogramme pour {audio_path}"
    assert os.path.exists(spectrogram_path), f"❌ Spectrogramme introuvable après génération: {spectrogram_path}"

    # Tester le prétraitement
    spec = preprocess_audio(audio_path)
    assert spec is not None, f"❌ Erreur: Audio non chargé {audio_path}"
    assert spec.shape == (64, 64, 1), f"❌ Erreur: Format audio incorrect pour {audio_path}"
