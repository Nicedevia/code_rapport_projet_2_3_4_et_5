import os
import shutil

# 📂 Dossiers source
IMAGE_SRC_CATS = "data/images/cleaned/training_set/cats"
IMAGE_SRC_DOGS = "data/images/cleaned/training_set/dogs"
AUDIO_SRC_CATS = "data/audio/cleaned/train/cats"  # 📢 Audios .wav
AUDIO_SRC_DOGS = "data/audio/cleaned/train/dogs"  # 📢 Audios .wav
SPECTROGRAM_SRC_CATS = "data/data_fusion_model/spectrograms/train/cats"  # 🎵 Spectrogrammes
SPECTROGRAM_SRC_DOGS = "data/data_fusion_model/spectrograms/train/dogs"  # 🎵 Spectrogrammes

# 📂 Dossiers destination (mini dataset)
IMAGE_DEST_CATS = "data_sample/images/cats"
IMAGE_DEST_DOGS = "data_sample/images/dogs"
AUDIO_DEST_CATS = "data_sample/audio/cats"
AUDIO_DEST_DOGS = "data_sample/audio/dogs"
SPECTROGRAM_DEST_CATS = "data_sample/spectrograms/cats"
SPECTROGRAM_DEST_DOGS = "data_sample/spectrograms/dogs"

# 📂 Création des dossiers s'ils n'existent pas
os.makedirs(IMAGE_DEST_CATS, exist_ok=True)
os.makedirs(IMAGE_DEST_DOGS, exist_ok=True)
os.makedirs(AUDIO_DEST_CATS, exist_ok=True)
os.makedirs(AUDIO_DEST_DOGS, exist_ok=True)
os.makedirs(SPECTROGRAM_DEST_CATS, exist_ok=True)
os.makedirs(SPECTROGRAM_DEST_DOGS, exist_ok=True)

# 📌 Fonction pour copier les 10 premiers fichiers
def copy_first_n_files(src_dir, dest_dir, n=10, file_ext=(".jpg", ".jpeg", ".png", ".wav")):
    """Copie les n premiers fichiers d'un dossier source vers un dossier destination."""
    if not os.path.exists(src_dir):
        print(f"❌ Source introuvable : {src_dir}")
        return

    # Filtrage des fichiers par extension
    files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(file_ext)])[:n]
    
    for file in files:
        src_path = os.path.join(src_dir, file)
        dest_path = os.path.join(dest_dir, file)
        shutil.copy(src_path, dest_path)
        print(f"✅ Copié : {src_path} -> {dest_path}")

# 📥 Copie des fichiers pour les CHATS
print("📥 Copie des 10 premières images de chats...")
copy_first_n_files(IMAGE_SRC_CATS, IMAGE_DEST_CATS, n=10, file_ext=(".jpg", ".jpeg", ".png"))

print("📥 Copie des 10 premiers fichiers audio WAV de chats...")
copy_first_n_files(AUDIO_SRC_CATS, AUDIO_DEST_CATS, n=10, file_ext=(".wav"))

print("📥 Copie des 10 premiers spectrogrammes de chats...")
copy_first_n_files(SPECTROGRAM_SRC_CATS, SPECTROGRAM_DEST_CATS, n=10, file_ext=(".png"))

# 📥 Copie des fichiers pour les CHIENS
print("📥 Copie des 10 premières images de chiens...")
copy_first_n_files(IMAGE_SRC_DOGS, IMAGE_DEST_DOGS, n=10, file_ext=(".jpg", ".jpeg", ".png"))

print("📥 Copie des 10 premiers fichiers audio WAV de chiens...")
copy_first_n_files(AUDIO_SRC_DOGS, AUDIO_DEST_DOGS, n=10, file_ext=(".wav"))

print("📥 Copie des 10 premiers spectrogrammes de chiens...")
copy_first_n_files(SPECTROGRAM_SRC_DOGS, SPECTROGRAM_DEST_DOGS, n=10, file_ext=(".png"))

print("✅ Mini dataset complet avec images, audios WAV et spectrogrammes !")
