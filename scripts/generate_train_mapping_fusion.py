import os
import random
import csv

discordant_sample_rate = 0.50  # Pourcentage de paires discordantes

def create_matching_pairs(image_cats, audio_cats, image_dogs, audio_dogs):
    """
    Génère des paires image-audio et leurs labels pour les chats et les chiens.
    """
    mapping_rows = []

    # --- Paires concordantes (Matching) ---
    def get_files(path, extensions):
        """Retourne une liste de fichiers valides dans un dossier donné."""
        if os.path.exists(path):
            return [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(extensions)]
        return []

    # 📂 Chargement des fichiers avec les bonnes extensions
    cat_images = get_files(image_cats, (".jpg", ".jpeg", ".png"))  # Chats: images
    dog_images = get_files(image_dogs, (".jpg", ".jpeg", ".png"))  # Chiens: images
    cat_audios = get_files(audio_cats, (".wav"))  # Chats: audios
    dog_audios = get_files(audio_dogs, (".wav"))  # Chiens: audios

    # 📊 Vérifier si les fichiers existent bien
    print(f"📂 Chat images: {len(cat_images)}, Chat audios: {len(cat_audios)}")
    print(f"📂 Chien images: {len(dog_images)}, Chien audios: {len(dog_audios)}")

    #  Matching des chats (label 0)
    for img in cat_images:
        if cat_audios:
            aud = random.choice(cat_audios)
            mapping_rows.append([img, aud, 0])  # 0 = Chat

    #  Matching des chiens (label 1)
    for img in dog_images:
        if dog_audios:
            aud = random.choice(dog_audios)
            mapping_rows.append([img, aud, 1])  # 1 = Chien

    # --- Paires discordantes (Erreur, label 2) ---
    # Chat image + Chien audio
    for img in cat_images:
        if dog_audios and random.random() < discordant_sample_rate:
            aud = random.choice(dog_audios)
            mapping_rows.append([img, aud, 2])  # 2 = Erreur

    # Chien image + Chat audio
    for img in dog_images:
        if cat_audios and random.random() < discordant_sample_rate:
            aud = random.choice(cat_audios)
            mapping_rows.append([img, aud, 2])  # 2 = Erreur

    return mapping_rows

# --- Exécution principale ---
if __name__ == "__main__":
    # Chemins mis à jour avec ceux utilisés dans ton projet
    image_cats = "data/images/cleaned/training_set/cats"
    audio_cats = "data/audio/cleaned/train/cats"
    image_dogs = "data/images/cleaned/training_set/dogs"
    audio_dogs = "data/audio/cleaned/train/dogs"

    # Vérification que les dossiers existent bien
    for path in [image_cats, audio_cats, image_dogs, audio_dogs]:
        if not os.path.exists(path):
            print(f"⚠️ ATTENTION: Le dossier {path} n'existe pas !")
    
    # Génération des paires
    mapping = create_matching_pairs(image_cats, audio_cats, image_dogs, audio_dogs)

    # Sauvegarde du mapping
    output_csv = "data/data_fusion_model/fusion_mapping.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "audio_path", "label"])
        writer.writerows(mapping)

    print(f"✅ Mapping généré avec succès : {len(mapping)} paires enregistrées dans {output_csv}")
    print(f"📊 Répartition des classes : "
          f"Chat: {sum(1 for row in mapping if row[2] == 0)}, "
          f"Chien: {sum(1 for row in mapping if row[2] == 1)}, "
          f"Erreur: {sum(1 for row in mapping if row[2] == 2)}")  
