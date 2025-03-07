#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import csv

discordant_sample_rate = 0.50

def create_matching_pairs(image_train_cats=None, image_train_dogs=None,
                            audio_train_cats=None, audio_train_dogs=None):
    """
    Génère des paires image-audio et leurs labels.
    
    Les paires sont créées ainsi :
      - Paires concordantes (Matching) :
          * image de chat + audio de chat  -> label 0
          * image de chien + audio de chien -> label 1
      - Paires discordantes (Mismatching, Erreur) :
          * image de chat + audio de chien  -> label 2
          * image de chien + audio de chat  -> label 2
    
    Si aucun chemin n'est fourni, des chemins par défaut sont utilisés.
    
    Retourne une liste de paires sous la forme [image_path, audio_path, label].
    """
    # Chemins par défaut
    if image_train_cats is None:
        image_train_cats = r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\cleaned\images\train\cats"
    if image_train_dogs is None:
        image_train_dogs = r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\cleaned\images\train\dogs"
    if audio_train_cats is None:
        audio_train_cats = r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\spectrograms\train\cats"
    if audio_train_dogs is None:
        audio_train_dogs = r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\spectrograms\train\dogs"

    mapping_rows = []

    # --- Paires concordantes (Matching) ---
    # Chat (label 0)
    if os.path.exists(image_train_cats) and os.path.exists(audio_train_cats):
        cat_images = [os.path.join(image_train_cats, f) for f in os.listdir(image_train_cats)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        cat_audios = [os.path.join(audio_train_cats, f) for f in os.listdir(audio_train_cats)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for img in cat_images:
            if cat_audios:
                aud = random.choice(cat_audios)
                mapping_rows.append([img, aud, 0])  # 0 = Chat

    # Chien (label 1)
    if os.path.exists(image_train_dogs) and os.path.exists(audio_train_dogs):
        dog_images = [os.path.join(image_train_dogs, f) for f in os.listdir(image_train_dogs)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        dog_audios = [os.path.join(audio_train_dogs, f) for f in os.listdir(audio_train_dogs)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for img in dog_images:
            if dog_audios:
                aud = random.choice(dog_audios)
                mapping_rows.append([img, aud, 1])  # 1 = Chien

    # --- Paires discordantes (Mismatching, Erreur, label 2) ---
    # Chat image + Chien audio
    if os.path.exists(image_train_cats) and os.path.exists(audio_train_dogs):
        cat_images = [os.path.join(image_train_cats, f) for f in os.listdir(image_train_cats)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        dog_audios = [os.path.join(audio_train_dogs, f) for f in os.listdir(audio_train_dogs)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for img in cat_images:
            if dog_audios and random.random() < discordant_sample_rate:
                aud = random.choice(dog_audios)
                mapping_rows.append([img, aud, 2])  # 2 = Erreur

    # Chien image + Chat audio
    if os.path.exists(image_train_dogs) and os.path.exists(audio_train_cats):
        dog_images = [os.path.join(image_train_dogs, f) for f in os.listdir(image_train_dogs)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        cat_audios = [os.path.join(audio_train_cats, f) for f in os.listdir(audio_train_cats)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for img in dog_images:
            if cat_audios and random.random() < discordant_sample_rate:
                aud = random.choice(cat_audios)
                mapping_rows.append([img, aud, 2])  # 2 = Erreur

    return mapping_rows

if __name__ == "__main__":
    mapping = create_matching_pairs()
    output_csv = r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\fusion_mapping.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "audio_path", "label"])
        writer.writerows(mapping)
    print(f"Mapping généré avec succès : {len(mapping)} paires enregistrées dans {output_csv}")
    # Affichage de la répartition des classes
    count_chat   = sum(1 for row in mapping if row[2] == 0)
    count_chien  = sum(1 for row in mapping if row[2] == 1)
    count_erreur = sum(1 for row in mapping if row[2] == 2)
    print(f"Répartition des classes : Chat: {count_chat}, Chien: {count_chien}, Erreur: {count_erreur}")
