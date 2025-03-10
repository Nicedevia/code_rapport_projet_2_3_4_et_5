import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from scripts.generate_train_mapping_fusion import create_matching_pairs

@pytest.mark.parametrize("image_base, audio_base", [
    (r"data_sample/images/cats", r"data_sample/audio/cats"),  # Test pour les chats
    (r"data_sample/images/dogs", r"data_sample/audio/dogs")   # Test pour les chiens
])
def test_matching_pairs(image_base, audio_base):
    """Teste la génération des paires image-audio pour chats et chiens."""
    
    print(f"\n🔍 Vérification des fichiers pour {image_base} et {audio_base}")

    # Vérifie si les dossiers existent
    assert os.path.exists(image_base), f"❌ Dossier images non trouvé: {image_base}"
    assert os.path.exists(audio_base), f"❌ Dossier audio non trouvé: {audio_base}"

    # Lister les fichiers
    images = os.listdir(image_base)
    audios = os.listdir(audio_base)

    print(f"📸 {len(images)} images trouvées : {images[:5]}...")  # Afficher quelques fichiers
    print(f"🎵 {len(audios)} audios trouvés : {audios[:5]}...")

    # Vérifier qu'il y a au moins 1 fichier
    assert len(images) > 0, f"❌ Aucune image trouvée dans {image_base}"
    assert len(audios) > 0, f"❌ Aucun spectrogramme trouvé dans {audio_base}"

    # 🏆 Création des paires
    pairs = create_matching_pairs(image_base, audio_base)
    assert len(pairs) > 0, "❌ Aucune paire formée"

    # Vérifie que chaque paire a bien un label 0, 1 ou 2
    for img_path, aud_path, label in pairs:
        assert label in [0, 1, 2], f"❌ Erreur: Label incorrect ({label}) pour {img_path} - {aud_path}"
    
    print(f"✅ {len(pairs)} paires générées correctement pour {image_base}")
