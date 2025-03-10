import os
import shutil

# 📂 Dossiers source
IMAGE_SRC_DIR = "data/images/cleaned/training_set/dogs"
AUDIO_SRC_DIR = "data/data_fusion_model/spectrograms/test/dogs"

# 📂 Dossiers de destination (mini dataset)
IMAGE_DEST_DIR = "data_sample/images/dogs"
AUDIO_DEST_DIR = "data_sample/audio/dogs"

# Création des dossiers s'ils n'existent pas
os.makedirs(IMAGE_DEST_DIR, exist_ok=True)
os.makedirs(AUDIO_DEST_DIR, exist_ok=True)

# Fonction pour copier les 10 premiers fichiers
def copy_first_n_files(src_dir, dest_dir, n=10):
    files = sorted(os.listdir(src_dir))[:n]  # Trie et prend les premiers fichiers
    for file in files:
        src_path = os.path.join(src_dir, file)
        dest_path = os.path.join(dest_dir, file)
        shutil.copy(src_path, dest_path)
        print(f"✅ Copié : {src_path} -> {dest_path}")

# Copier les fichiers
print("📥 Copie des 10 premières images...")
copy_first_n_files(IMAGE_SRC_DIR, IMAGE_DEST_DIR, n=10)

print("📥 Copie des 10 premiers spectrogrammes...")
copy_first_n_files(AUDIO_SRC_DIR, AUDIO_DEST_DIR, n=10)

print("✅ Mini dataset créé avec succès !")
