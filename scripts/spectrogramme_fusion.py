#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 📂 Chemins des fichiers
# Utilise les fichiers audio nettoyés dans la nouvelle arborescence
AUDIO_DIR = r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\cleaned\audio"
# Dossier de sortie pour les spectrogrammes
SPECTROGRAM_DIR = r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\spectrograms"

# 🔄 Créer les dossiers de sortie si non existants pour train et test, pour chaque catégorie
for split in ["train", "test"]:
    for category in ["cats", "dogs"]:
        os.makedirs(os.path.join(SPECTROGRAM_DIR, split, category), exist_ok=True)

# 🎵 Fonction pour convertir un fichier audio en spectrogramme
def audio_to_spectrogram(audio_path, output_path):
    try:
        # Chargement de l'audio avec une durée max de 2 secondes et un taux d'échantillonnage de 22050 Hz
        y, sr = librosa.load(audio_path, sr=22050, duration=2)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        plt.figure(figsize=(3, 3))
        librosa.display.specshow(spectrogram_db, sr=sr, x_axis="time", y_axis="mel")
        plt.axis("off")

        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"✅ Spectrogramme généré : {output_path}")

    except Exception as e:
        print(f"❌ Erreur lors de la conversion {audio_path}: {e}")

# 🔄 Générer les spectrogrammes pour chaque fichier audio dans les dossiers train et test
for split in ["train", "test"]:
    for category in ["cats", "dogs"]:
        audio_folder = os.path.join(AUDIO_DIR, split, category)
        output_folder = os.path.join(SPECTROGRAM_DIR, split, category)

        if not os.path.exists(audio_folder):
            print(f"⚠️ Dossier audio introuvable : {audio_folder}")
            continue

        for audio_file in os.listdir(audio_folder):
            if audio_file.lower().endswith(".wav"):
                audio_path = os.path.join(audio_folder, audio_file)
                output_path = os.path.join(output_folder, audio_file.replace(".wav", ".png"))

                # Vérifier si le spectrogramme existe déjà
                if not os.path.exists(output_path):
                    audio_to_spectrogram(audio_path, output_path)
                else:
                    print(f"🔁 Spectrogramme déjà existant : {output_path}")

print("✅ Génération des spectrogrammes terminée !")
