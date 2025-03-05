#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import csv

# Paramètre de sous-échantillonnage pour les paires discordantes (ici 20 %)
discordant_sample_rate = 0.20

# Répertoires de test (images et audio nettoyés) avec les nouveaux chemins
image_test_cats = r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\cleaned\images\test\cats"
image_test_dogs = r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\cleaned\images\test\dogs"
audio_test_cats = r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\cleaned\audio\test\cats"
audio_test_dogs = r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\cleaned\audio\test\dogs"

mapping_rows = []

# --- Paires concordantes ---
# Pour les chats (label 0)
if os.path.exists(image_test_cats) and os.path.exists(audio_test_cats):
    test_cat_images = [os.path.join(image_test_cats, f)
                       for f in os.listdir(image_test_cats)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    test_cat_audios = [os.path.join(audio_test_cats, f)
                       for f in os.listdir(audio_test_cats)
                       if f.lower().endswith(".wav")]
    for img in test_cat_images:
        if test_cat_audios:
            aud = random.choice(test_cat_audios)
            mapping_rows.append([img, aud, 0])  # 0 = Chat

# Pour les chiens (label 1)
if os.path.exists(image_test_dogs) and os.path.exists(audio_test_dogs):
    test_dog_images = [os.path.join(image_test_dogs, f)
                       for f in os.listdir(image_test_dogs)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    test_dog_audios = [os.path.join(audio_test_dogs, f)
                       for f in os.listdir(audio_test_dogs)
                       if f.lower().endswith(".wav")]
    for img in test_dog_images:
        if test_dog_audios:
            aud = random.choice(test_dog_audios)
            mapping_rows.append([img, aud, 1])  # 1 = Chien

# --- Paires discordantes (Erreur, label 2) ---
# Chat image + Chien audio
if os.path.exists(image_test_cats) and os.path.exists(audio_test_dogs):
    for img in test_cat_images:
        if test_dog_audios and random.random() < discordant_sample_rate:
            aud = random.choice(test_dog_audios)
            mapping_rows.append([img, aud, 2])  # 2 = Erreur

# Chien image + Chat audio
if os.path.exists(image_test_dogs) and os.path.exists(audio_test_cats):
    for img in test_dog_images:
        if test_cat_audios and random.random() < discordant_sample_rate:
            aud = random.choice(test_cat_audios)
            mapping_rows.append([img, aud, 2])  # 2 = Erreur

# Sauvegarde du mapping dans un fichier CSV
output_csv = r"C:\Users\briac\Desktop\projet_3\data\data_fusion_model\test_image_audio_mapping.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_path", "audio_path", "label"])
    writer.writerows(mapping_rows)

print(f"Mapping de test généré : {len(mapping_rows)} paires enregistrées dans {output_csv}")
