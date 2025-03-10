#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import csv

discordant_sample_rate = 0.50

def create_matching_pairs(image_train_cats="data/data_fusion_model/cleaned/images/train/cats",
                          image_train_dogs="data/data_fusion_model/cleaned/images/train/dogs",
                          audio_train_cats="data/data_fusion_model/spectrograms/train/cats",
                          audio_train_dogs="data/data_fusion_model/spectrograms/train/dogs"):
    """
    Génère des paires image-audio et leurs labels.
    Retourne une liste de paires sous la forme [image_path, audio_path, label].
    """
    mapping_rows = []

    def list_files(directory, extensions):
        """Liste les fichiers dans un dossier avec certaines extensions."""
        return [os.path.join(directory, f) for f in os.listdir(directory)
                if f.lower().endswith(extensions)] if os.path.exists(directory) else []

    # Charger les fichiers
    cat_images = list_files(image_train_cats, (".jpg", ".jpeg", ".png"))
    cat_audios = list_files(audio_train_cats, (".png", ".jpg", ".jpeg"))
    dog_images = list_files(image_train_dogs, (".jpg", ".jpeg", ".png"))
    dog_audios = list_files(audio_train_dogs, (".png", ".jpg", ".jpeg"))

    print(f"📂 Chat images: {len(cat_images)}, Chat audios: {len(cat_audios)}")
    print(f"📂 Chien images: {len(dog_images)}, Chien audios: {len(dog_audios)}")

    # Création des paires concordantes
    for img in cat_images:
        if cat_audios:
            aud = random.choice(cat_audios)
            mapping_rows.append([img, aud, 0])  # Chat

    for img in dog_images:
        if dog_audios:
            aud = random.choice(dog_audios)
            mapping_rows.append([img, aud, 1])  # Chien

    # Création des paires discordantes (erreurs)
    for img in cat_images:
        if dog_audios and random.random() < discordant_sample_rate:
            aud = random.choice(dog_audios)
            mapping_rows.append([img, aud, 2])  # Erreur

    for img in dog_images:
        if cat_audios and random.random() < discordant_sample_rate:
            aud = random.choice(cat_audios)
            mapping_rows.append([img, aud, 2])  # Erreur

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
