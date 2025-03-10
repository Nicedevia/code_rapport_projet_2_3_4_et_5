import os
import shutil

# 📂 Dossiers source (grande base)
IMAGE_SRC_CHAT = "data/images/cleaned/training_set/cats"
AUDIO_SRC_CHAT = "data/audio/cleaned/train/cats"

IMAGE_SRC_DOG = "data/images/cleaned/training_set/dogs"
AUDIO_SRC_DOG = "data/audio/cleaned/train/dogs"

# 📂 Dossiers de destination (mini dataset `data_retrain`)
IMAGE_DEST_CHAT = "data_retrain/training/images/chat"
AUDIO_DEST_CHAT = "data_retrain/training/audio/chat"

IMAGE_DEST_DOG = "data_retrain/training/images/dog"
AUDIO_DEST_DOG = "data_retrain/training/audio/dog"

# Création des dossiers s'ils n'existent pas
os.makedirs(IMAGE_DEST_CHAT, exist_ok=True)
os.makedirs(AUDIO_DEST_CHAT, exist_ok=True)
os.makedirs(IMAGE_DEST_DOG, exist_ok=True)
os.makedirs(AUDIO_DEST_DOG, exist_ok=True)

# Fonction pour copier les 30 premiers fichiers
def copy_first_n_files(src_dir, dest_dir, n=30):
    if not os.path.exists(src_dir):
        print(f"❌ Dossier introuvable : {src_dir}")
        return
    files = sorted(os.listdir(src_dir))[:n]  # Trie et prend les premiers fichiers
    for file in files:
        src_path = os.path.join(src_dir, file)
        dest_path = os.path.join(dest_dir, file)
        shutil.copy(src_path, dest_path)
        print(f"✅ Copié : {src_path} -> {dest_path}")

# Copier les fichiers
print("📥 Copie des 30 premières images de chat...")
copy_first_n_files(IMAGE_SRC_CHAT, IMAGE_DEST_CHAT, n=30)

print("📥 Copie des 30 premiers audios de chat...")
copy_first_n_files(AUDIO_SRC_CHAT, AUDIO_DEST_CHAT, n=30)

print("📥 Copie des 30 premières images de chien...")
copy_first_n_files(IMAGE_SRC_DOG, IMAGE_DEST_DOG, n=30)

print("📥 Copie des 30 premiers audios de chien...")
copy_first_n_files(AUDIO_SRC_DOG, AUDIO_DEST_DOG, n=30)

print("✅ Dataset `data_retrain` généré avec succès !")
