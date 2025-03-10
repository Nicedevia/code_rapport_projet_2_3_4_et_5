import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from scripts.generate_train_mapping_fusion import create_matching_pairs

@pytest.mark.parametrize("image_cats, audio_cats, image_dogs, audio_dogs", [
    (r"data_sample/images/cats", r"data_sample/audio/cats", r"data_sample/images/dogs", r"data_sample/audio/dogs")
])
def test_matching_pairs(image_cats, audio_cats, image_dogs, audio_dogs):
    """Teste la génération des paires image-audio pour chats et chiens."""
    
    print(f"\n🔍 Vérification des fichiers pour {image_cats} et {audio_cats}")
    print(f"🔍 Vérification des fichiers pour {image_dogs} et {audio_dogs}")

    # Vérifie si les dossiers existent
    assert os.path.exists(image_cats), f"❌ Dossier images non trouvé: {image_cats}"
    assert os.path.exists(audio_cats), f"❌ Dossier audio non trouvé: {audio_cats}"
    assert os.path.exists(image_dogs), f"❌ Dossier images non trouvé: {image_dogs}"
    assert os.path.exists(audio_dogs), f"❌ Dossier audio non trouvé: {audio_dogs}"

    # Lister les fichiers
    images_cats = os.listdir(image_cats)
    audios_cats = os.listdir(audio_cats)
    images_dogs = os.listdir(image_dogs)
    audios_dogs = os.listdir(audio_dogs)

    print(f"📸 {len(images_cats)} images de chats trouvées : {images_cats[:5]}...")
    print(f"🎵 {len(audios_cats)} audios de chats trouvés : {audios_cats[:5]}...")
    print(f"📸 {len(images_dogs)} images de chiens trouvées : {images_dogs[:5]}...")
    print(f"🎵 {len(audios_dogs)} audios de chiens trouvés : {audios_dogs[:5]}...")

    # Vérifier qu'il y a bien des fichiers
    assert len(images_cats) > 0, f"❌ Aucune image trouvée dans {image_cats}"
    assert len(audios_cats) > 0, f"❌ Aucun spectrogramme trouvé dans {audio_cats}"
    assert len(images_dogs) > 0, f"❌ Aucune image trouvée dans {image_dogs}"
    assert len(audios_dogs) > 0, f"❌ Aucun spectrogramme trouvé dans {audio_dogs}"

    # 🏆 Création des paires
    pairs = create_matching_pairs(image_cats, audio_cats, image_dogs, audio_dogs)
    assert len(pairs) > 0, "❌ Aucune paire formée"

    # Vérifie que chaque paire a bien un label 0, 1 ou 2
    for img_path, aud_path, label in pairs:
        assert label in [0, 1, 2], f"❌ Erreur: Label incorrect ({label}) pour {img_path} - {aud_path}"
    
    print(f"✅ {len(pairs)} paires générées correctement pour les chats et chiens")
