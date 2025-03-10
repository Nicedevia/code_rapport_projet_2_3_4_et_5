#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import csv

discordant_sample_rate = 0.50

def create_matching_pairs(image_cats, audio_cats, image_dogs, audio_dogs):
    """
    Génère des paires image-audio et leurs labels pour les chats et les chiens.
    """
    mapping_rows = []

    # --- Paires concordantes (Matching) ---
    def get_files(path, extensions=(".jpg", ".jpeg", ".png", ".wav")):
        """Retourne une liste de fichiers valides dans un dossier donné."""
        return [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(extensions)]
    
    cat_images = get_files(image_cats, (".jpg", ".jpeg", ".png"))
    dog_images = get_files(image_dogs, (".jpg", ".jpeg", ".png"))
    cat_audios = get_files(audio_cats, (".wav"))
    dog_audios = get_files(audio_dogs, (".wav"))

    # 📂 Vérifier si les fichiers existent bien
    print(f"📂 Chat images: {len(cat_images)}, Chat audios: {len(cat_audios)}")
    print(f"📂 Chien images: {len(dog_images)}, Chien audios: {len(dog_audios)}")

    # Chat (label 0)
    for img in cat_images:
        if cat_audios:
            aud = random.choice(cat_audios)
            mapping_rows.append([img, aud, 0])  # 0 = Chat

    # Chien (label 1)
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

if __name__ == "__main__":
    mapping = create_matching_pairs()
    output_csv = "data/data_fusion_model/fusion_mapping.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "audio_path", "label"])
        writer.writerows(mapping)

    print(f"✅ Mapping généré avec succès : {len(mapping)} paires enregistrées dans {output_csv}")
    print(f"📊 Répartition des classes : Chat: {sum(1 for row in mapping if row[2] == 0)}, "
          f"Chien: {sum(1 for row in mapping if row[2] == 1)}, "
          f"Erreur: {sum(1 for row in mapping if row[2] == 2)}")
