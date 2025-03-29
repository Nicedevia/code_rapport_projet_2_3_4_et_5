import os
import shutil
import hashlib
import librosa
import numpy as np

# ====================================================
# Fonction utilitaire : calcul du hash MD5 d'un fichier
# ====================================================
def compute_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# ====================================================
# Vérification de la qualité audio basée sur le clipping
# ====================================================
def is_audio_quality_good(file_path, clipping_threshold=0.01):
    """
    Vérifie si le fichier audio ne présente pas de clipping excessif.
    
    Paramètres :
        - file_path : chemin vers le fichier audio.
        - clipping_threshold : ratio maximal d'échantillons clipés toléré.
    
    Retourne True si le fichier est de qualité acceptable, False sinon.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        print(f"Erreur lors du chargement du fichier {file_path} : {e}")
        return False

    clipping_samples = np.sum(np.isclose(np.abs(y), 1.0, atol=1e-3))
    clipping_ratio = clipping_samples / len(y)
    if clipping_ratio > clipping_threshold:
        print(f"🛑 Fichier avec clipping excessif (ratio = {clipping_ratio:.2%}) : {file_path}")
        return False

    return True

# ====================================================
# Nettoyage des fichiers audio dans un dossier source
# ====================================================
def clean_audio_folder(source_dir, dest_dir, expected_label):
    """
    Traite les fichiers audio .wav dans source_dir.
    Seuls les fichiers dont le nom contient expected_label (ex: "cat" ou "dog")
    et qui passent le test de qualité (vérification du clipping) sont acceptés.
    Les doublons (vérifiés par hash) ou fichiers au format incorrect sont supprimés.
    Les fichiers validés sont déplacés vers dest_dir.
    Retourne un dictionnaire de compteurs.
    """
    os.makedirs(dest_dir, exist_ok=True)
    counters = {"accepted": 0, "ignored": 0, "duplicates": 0}
    hash_set = {}

    for file in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file)
        if not os.path.isfile(file_path):
            continue

        if not file.lower().endswith(".wav"):
            print(f"🛑 Audio ignoré (format incorrect) : {file}")
            os.remove(file_path)
            counters["ignored"] += 1
            continue

        if expected_label not in file.lower():
            print(f"🛑 Audio ignoré (nom non conforme, attendu '{expected_label}') : {file}")
            os.remove(file_path)
            counters["ignored"] += 1
            continue

        if not is_audio_quality_good(file_path, clipping_threshold=0.01):
            os.remove(file_path)
            counters["ignored"] += 1
            continue

        file_hash = compute_hash(file_path)
        if file_hash in hash_set:
            print(f"⚠️ Audio doublon détecté : {file}. Fichier supprimé.")
            os.remove(file_path)
            counters["duplicates"] += 1
            continue
        else:
            hash_set[file_hash] = file_path

        dest_file = os.path.join(dest_dir, file)
        if os.path.exists(dest_file):
            print(f"⚠️ Audio doublon (nom identique) : {file}. Fichier supprimé.")
            os.remove(file_path)
            counters["duplicates"] += 1
            continue

        shutil.move(file_path, dest_file)
        print(f"📂 Audio déplacé : {file} -> {dest_dir}")
        counters["accepted"] += 1

    return counters

# ====================================================
# Nettoyage des fichiers images dans un dossier source
# ====================================================
def clean_image_folder(source_dir, dest_dir, expected_label):
    """
    Traite les fichiers image dans source_dir.
    Seuls les fichiers avec extension .jpg, .jpeg ou .png sont acceptés.
    Le nom doit contenir expected_label (ex: "cat" ou "dog").
    Les doublons sont supprimés (vérification par hash).
    Les fichiers validés sont déplacés vers dest_dir.
    Retourne un dictionnaire de compteurs.
    """
    os.makedirs(dest_dir, exist_ok=True)
    counters = {"accepted": 0, "ignored": 0, "duplicates": 0}
    hash_set = {}
    allowed_exts = (".jpg", ".jpeg", ".png")

    for file in os.listdir(source_dir):
        file_path = os.path.join(source_dir, file)
        if not os.path.isfile(file_path):
            continue

        if not file.lower().endswith(allowed_exts):
            print(f"🛑 Image ignorée (format incorrect) : {file}")
            os.remove(file_path)
            counters["ignored"] += 1
            continue

        if expected_label not in file.lower():
            print(f"🛑 Image ignorée (nom non conforme, attendu '{expected_label}') : {file}")
            os.remove(file_path)
            counters["ignored"] += 1
            continue

        file_hash = compute_hash(file_path)
        if file_hash in hash_set:
            print(f"⚠️ Image doublon détectée : {file}. Fichier supprimé.")
            os.remove(file_path)
            counters["duplicates"] += 1
            continue
        else:
            hash_set[file_hash] = file_path

        dest_file = os.path.join(dest_dir, file)
        if os.path.exists(dest_file):
            print(f"⚠️ Image doublon (nom identique) : {file}. Fichier supprimé.")
            os.remove(file_path)
            counters["duplicates"] += 1
            continue

        shutil.move(file_path, dest_file)
        print(f"📂 Image déplacée : {file} -> {dest_dir}")
        counters["accepted"] += 1

    return counters

# ====================================================
# Nettoyage global des données audio
# ====================================================
def clean_audio_data():
    r"""
    Parcourt les répertoires audio de train et test pour les deux catégories.
    Les sources sont issues des dossiers de fusion :
        - TRAIN : C:\Users\briac\Desktop\projet_3\data\data_fusion_model\train\audio\{cats,dogs}
        - TEST  : C:\Users\briac\Desktop\projet_3\data\data_fusion_model\test\audio\{cats,dogs}
    Les fichiers validés seront déplacés vers les dossiers CLEANED correspondants.
    """
    audio_sets = [
        ("train", "cats",
         r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\train\audio\cats",
         r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\cleaned\audio\train\cats"),
        ("train", "dogs",
         r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\train\audio\dogs",
         r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\cleaned\audio\train\dogs"),
        ("test", "cats",
         r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\test\audio\cats",
         r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\cleaned\audio\test\cats"),
        ("test", "dogs",
         r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\test\audio\dogs",
         r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\cleaned\audio\test\dogs"),
    ]
    total_counters = {"accepted": 0, "ignored": 0, "duplicates": 0}
    for subset, category, src, dst in audio_sets:
        if not os.path.exists(src):
            print(f"❌ Répertoire source audio introuvable : {src}")
            continue
        print(f"\n🟢 Traitement audio {subset.upper()} - {category.upper()} :")
        counters = clean_audio_folder(src, dst, expected_label=category[:-1])
        print(f"   Résultats : {counters['accepted']} acceptés, {counters['ignored']} ignorés, {counters['duplicates']} doublons.")
        for key in total_counters:
            total_counters[key] += counters[key]
    print("\n🎵 Nettoyage audio global terminé !")
    print(f"Total : {total_counters['accepted']} acceptés, {total_counters['ignored']} ignorés, {total_counters['duplicates']} doublons supprimés.")

# ====================================================
# Nettoyage global des données images
# ====================================================
def clean_image_data():
    r"""
    Parcourt les répertoires images de train et test (issus de la fusion) et dépose les fichiers validés
    dans les dossiers CLEANED correspondants.
    Les sources sont :
        - TRAIN : C:\Users\briac\Desktop\projet_3\data\data_fusion_model\train\images\{cats,dogs}
        - TEST  : C:\Users\briac\Desktop\projet_3\data\data_fusion_model\test\images\{cats,dogs}
    """
    image_sets = [
        ("train", "cats",
         r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\train\images\cats",
         r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\cleaned\images\train\cats"),
        ("train", "dogs",
         r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\train\images\dogs",
         r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\cleaned\images\train\dogs"),
        ("test", "cats",
         r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\test\images\cats",
         r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\cleaned\images\test\cats"),
        ("test", "dogs",
         r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\test\images\dogs",
         r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\cleaned\images\test\dogs"),
    ]
    total_counters = {"accepted": 0, "ignored": 0, "duplicates": 0}
    for subset, category, src, dst in image_sets:
        if not os.path.exists(src):
            print(f"❌ Répertoire source image introuvable : {src}")
            continue
        print(f"\n🟢 Traitement image {subset.upper()} - {category.upper()} :")
        counters = clean_image_folder(src, dst, expected_label=category[:-1])
        print(f"   Résultats : {counters['accepted']} acceptées, {counters['ignored']} ignorées, {counters['duplicates']} doublons.")
        for key in total_counters:
            total_counters[key] += counters[key]
    print("\n🖼️ Nettoyage images global terminé !")
    print(f"Total : {total_counters['accepted']} acceptées, {total_counters['ignored']} ignorées, {total_counters['duplicates']} doublons supprimés.")

# ====================================================
# Fonction principale
# ====================================================
def main():
    print("Démarrage du nettoyage AUDIO...")
    clean_audio_data()
    print("\nDémarrage du nettoyage IMAGES...")
    clean_image_data()

if __name__ == "__main__":
    main()
