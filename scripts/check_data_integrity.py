import os
import pandas as pd

IMAGE_TRAIN_FOLDER = "data/extracted/training_set"
AUDIO_TRAIN_FOLDER = "data/audio/cats_dogs/train"
TEST_CSV = "data/audio/test_image_audio_mapping.csv"
TRAIN_CSV = "data/audio/train_image_audio_mapping.csv"
SPECTROGRAM_FOLDER = "data/audio/spectrograms"

def count_files(folder):
    return sum([len(files) for _, _, files in os.walk(folder)])

print("📊 Vérification des fichiers...")
print(f"📸 Images d'entraînement : {count_files(IMAGE_TRAIN_FOLDER)}")
print(f"🎵 Audios d'entraînement : {count_files(AUDIO_TRAIN_FOLDER)}")
print(f"🖼 Spectrogrammes générés : {count_files(SPECTROGRAM_FOLDER)}")

if os.path.exists(TRAIN_CSV):
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"✅ {len(train_df)} associations image-audio dans le dataset d'entraînement")
else:
    print("⚠️ Fichier train_image_audio_mapping.csv manquant !")

if os.path.exists(TEST_CSV):
    test_df = pd.read_csv(TEST_CSV)
    print(f"✅ {len(test_df)} associations image-audio dans le dataset de test")
else:
    print("⚠️ Fichier test_image_audio_mapping.csv manquant !")
