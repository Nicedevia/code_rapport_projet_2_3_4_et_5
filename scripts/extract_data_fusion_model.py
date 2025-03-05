import os
import zipfile
import shutil

def extract_zip(zip_path, extract_to):
    """Extrait le contenu du fichier ZIP vers le dossier extract_to."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def move_files(src_folder, dst_folder, file_extensions=None):
    """
    Déplace les fichiers de src_folder vers dst_folder.
    Si file_extensions est défini, ne déplace que les fichiers dont l'extension correspond.
    """
    os.makedirs(dst_folder, exist_ok=True)
    for filename in os.listdir(src_folder):
        if file_extensions:
            if not any(filename.lower().endswith(ext) for ext in file_extensions):
                continue
        shutil.move(os.path.join(src_folder, filename), os.path.join(dst_folder, filename))

# --- Paramètres à adapter ---

# Chemin vers le fichier ZIP
zip_path = r"C:\Users\briac\Desktop\projet_3\data\cat-and-dog.zip"
# Dossier temporaire pour l'extraction
temp_extract_dir = r"C:\Users\briac\Desktop\projet_3\data\extracted_cat_and_dog"

# Arborescence interne attendue dans le ZIP
# Pour les images d'entraînement et de test, par classe
train_dogs_src = os.path.join(temp_extract_dir, "training_set", "training_set", "dogs")
train_cats_src = os.path.join(temp_extract_dir, "training_set", "training_set", "cats")
test_dogs_src = os.path.join(temp_extract_dir, "test_set", "test_set", "dogs")
test_cats_src = os.path.join(temp_extract_dir, "test_set", "test_set", "cats")

# Dossiers de destination
dst_train_dogs = r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\train\images\dogs"
dst_train_cats = r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\train\images\cats"
dst_test_dogs  = r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\test\images\dogs"
dst_test_cats  = r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\test\images\cats"

# Extensions d'images à traiter
image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

# --- Processus d'extraction et déplacement ---

# 1. Extraction du fichier ZIP dans le dossier temporaire
extract_zip(zip_path, temp_extract_dir)
print("Extraction terminée.")

# 2. Déplacement des images dans l'arborescence souhaitée
move_files(train_dogs_src, dst_train_dogs, image_extensions)
move_files(train_cats_src, dst_train_cats, image_extensions)
move_files(test_dogs_src,  dst_test_dogs,  image_extensions)
move_files(test_cats_src,  dst_test_cats,  image_extensions)

print("Déplacement des images terminé.")

import os
import zipfile
import shutil
import random
import math

def extract_zip(zip_path, extract_to):
    """Extrait l'intégralité du contenu du fichier ZIP dans le dossier spécifié."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def move_files_with_split(src_folder, dst_train, dst_test, file_extensions, test_ratio=0.2):
    """
    Déplace les fichiers audio depuis src_folder vers les dossiers de train et test.
    - test_ratio définit la part des fichiers à envoyer en test.
    - file_extensions : liste des extensions à considérer.
    """
    os.makedirs(dst_train, exist_ok=True)
    os.makedirs(dst_test, exist_ok=True)
    
    # Récupérer les fichiers avec les extensions souhaitées
    files = [f for f in os.listdir(src_folder)
             if any(f.lower().endswith(ext) for ext in file_extensions)]
    
    # Mélange aléatoire pour répartir les fichiers
    random.shuffle(files)
    
    test_count = math.ceil(len(files) * test_ratio)
    test_files = files[:test_count]
    train_files = files[test_count:]
    
    for f in train_files:
        shutil.move(os.path.join(src_folder, f), os.path.join(dst_train, f))
    for f in test_files:
        shutil.move(os.path.join(src_folder, f), os.path.join(dst_test, f))

# --- Paramètres à adapter ---
zip_path = r"C:\Users\briac\Desktop\projet_3\data\DvC.zip"
temp_extract_dir = r"C:\Users\briac\Desktop\projet_3\data\extracted_DvC"

# Extraction du ZIP
extract_zip(zip_path, temp_extract_dir)
print("Extraction terminée.")

# Dossiers sources (contenu du ZIP)
cats_src = os.path.join(temp_extract_dir, "DvC", "Cats")
dogs_src = os.path.join(temp_extract_dir, "DvC", "Dogs")

# Dossiers de destination pour l'audio
dst_train_cats = r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\train\audio\cats"
dst_train_dogs = r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\train\audio\dogs"
dst_test_cats  = r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\test\audio\cats"
dst_test_dogs  = r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\test\audio\dogs"

# Extensions audio à traiter (adapter si besoin)
audio_extensions = [".mp3", ".wav"]

# Répartition et déplacement des fichiers audio pour chaque classe
move_files_with_split(cats_src, dst_train_cats, dst_test_cats, audio_extensions, test_ratio=0.2)
move_files_with_split(dogs_src, dst_train_dogs, dst_test_dogs, audio_extensions, test_ratio=0.2)

print("Répartition et déplacement des fichiers audio terminé.")
