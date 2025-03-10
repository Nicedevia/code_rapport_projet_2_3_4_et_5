import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from scripts.generate_train_mapping_fusion import create_matching_pairs
import os

def test_matching_pairs():
    image_base = r"data_sample/images/cats"
    audio_base = r"data_sample/audio/cats"


    # 🔍 Vérifier que les fichiers existent bien avant le test
    print("📂 Vérification des fichiers dans", image_base)
    print("📂 Vérification des fichiers dans", audio_base)

    images = os.listdir(image_base) if os.path.exists(image_base) else []
    audios = os.listdir(audio_base) if os.path.exists(audio_base) else []

    print(f"📸 Images trouvées ({len(images)} fichiers) : {images}")
    print(f"🎵 Audios trouvés ({len(audios)} fichiers) : {audios}")

    assert len(images) > 0, "❌ Aucune image trouvée dans data_sample/images/cats"
    assert len(audios) > 0, "❌ Aucun spectrogramme trouvé dans data_sample/audio/cats"

    # 🏆 Création des paires
    pairs = create_matching_pairs(image_base, audio_base)
    assert len(pairs) > 0, "❌ Aucune paire formée"

def test_matching_pairs():
    # Utilisation des chemins fournis :
    # Pour les images, on utilise "data/images/cleaned/training_set"
    # Pour l'audio, on utilise "data/data_fusion_model/spectrograms/test"

    image_base = r"data_sample/images/cats"
    audio_base = r"data_sample/audio/cats"

    
    pairs = create_matching_pairs(image_base, audio_base)
    assert len(pairs) > 0, "Erreur: Aucune paire formée"
    
    # Vérifie que chaque paire a bien un label 0, 1 ou 2
    for img_path, aud_path, label in pairs:
        assert label in [0, 1, 2], f"Erreur: Label incorrect ({label})"
