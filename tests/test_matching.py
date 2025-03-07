import pytest
from scripts.matching import create_matching_pairs  # Fonction qui fait le matching des paires

def test_matching_pairs():
    pairs = create_matching_pairs("data/images/", "data/audio/")
    assert len(pairs) > 0, "Erreur: Aucune paire formée"
    
    # Vérifie que chaque paire a bien un label 0, 1 ou 2
    for _, _, label in pairs:
        assert label in [0, 1, 2], "Erreur: Label incorrect"

