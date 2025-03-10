import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from scripts.generate_train_mapping_fusion import create_matching_pairs

def test_matching_pairs():
    # Utilisation des chemins fournis :
    # Pour les images, on utilise "data/images/cleaned/training_set"
    # Pour l'audio, on utilise "data/data_fusion_model/spectrograms/test"
    image_base = r"data_sample/images/cleaned/training_set/cats"
    audio_base = r"data_sample/data_fusion_model/spectrograms/test/cats"
    
    pairs = create_matching_pairs(image_base, audio_base)
    assert len(pairs) > 0, "Erreur: Aucune paire formée"
    
    # Vérifie que chaque paire a bien un label 0, 1 ou 2
    for img_path, aud_path, label in pairs:
        assert label in [0, 1, 2], f"Erreur: Label incorrect ({label})"
