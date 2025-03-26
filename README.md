# API de Classification Chat/Chien

Cette API REST, développée avec FastAPI, permet de classer des images et des fichiers audio pour déterminer s'il s'agit d'un chat ou d'un chien. Elle inclut un pipeline complet pour le traitement des données, l'entraînement des modèles, et le déploiement via Docker.

---

## Table des matières

- [Pipeline du Projet](#pipeline-du-projet)
- [Installation](#installation)
- [Lancement de l'API](#lancement-de-lapi)
- [Utilisation](#utilisation)
- [Tests](#tests)
- [Commandes Docker](#commandes-docker)
- [Structure du Projet](#structure-du-projet)
- [Contributions](#contributions)
- [Licence](#licence)

---

## Pipeline du Projet

Le pipeline du projet est organisé en plusieurs étapes :

1. **Préparation des données** :
   - Nettoyage des données (images et audio).
   - Génération des spectrogrammes pour les fichiers audio.
   - Fusion des données pour le modèle multimodal.

2. **Entraînement des modèles** :
   - Entraînement séparé des modèles pour les images et les fichiers audio.
   - Entraînement du modèle de fusion multimodal.

3. **Déploiement** :
   - Déploiement de l'API via FastAPI.
   - Monitoring avec Prometheus et Alertmanager.
   - Conteneurisation avec Docker.

4. **Tests** :
   - Tests unitaires pour les fonctions de traitement des données.
   - Tests d'intégration pour les modèles et l'API.

---

## Installation

1. Clonez le dépôt :
    ```bash
    git clone https://github.com/votre-utilisateur/projet_3.git
    cd projet_3
    ```

2. Créez un environnement virtuel :
    ```bash
    python -m venv env
    source env/bin/activate  # Sur Windows : env\Scripts\activate
    ```

3. Installez les dépendances :
    ```bash
    pip install -r requirements.txt
    ```

---

## Lancement de l'API

1. Lancez l'API avec Uvicorn :
    ```bash
    uvicorn api.api:app --host 0.0.0.0 --port 8000 --reload
    ```

2. Accédez à la documentation interactive de l'API :
    [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Utilisation

### Prédiction d'une image
Envoyez une requête POST à `/predict/image` avec une image en pièce jointe.

### Prédiction d'un fichier audio
Envoyez une requête POST à `/predict/audio` avec un fichier audio en pièce jointe.

### Prédiction multimodale (image + audio)
Envoyez une requête POST à `/predict/multimodal` avec une image et un fichier audio en pièces jointes.

---

## Tests

Pour exécuter les tests, utilisez `pytest` :
```bash
pytest tests/
```

Commandes Docker
1. Construire et lancer les conteneurs
Utilisez docker-compose pour construire et lancer tous les services :

bash docker-compose up --build -d

2. Arrêter et supprimer tous les conteneurs
Le script stop_all_docker.sh arrête et supprime tous les conteneurs Docker :
bash [stop_all_docker.sh]

3. Nettoyer les conteneurs et images Docker
Le script cleanup_docker.sh nettoie les conteneurs, images, volumes et réseaux inutilisés :
bash [cleanup_docker.sh]

4. Relancer tous les services Docker
Le script start_all_docker.sh relance tous les services définis dans docker-compose.yml :
bash [start_all_docker.sh]

Structure du Projet
projet_3/
├── api/
│   ├── api.py
│   ├── routes.py
│   ├── monitoring.py
│   └── model_loader.py
├── config/
│   └── kaggle.json
├── data/
│   ├── audio/
│   ├── images/
│   └── data_fusion_model/
├── models/
│   ├── image_audio_fusion_new_model_v2.keras
│   └── fusion.h5
├── scripts/
│   ├── preprocess_images.py
│   ├── preprocess_audio.py
│   ├── train_image_audio_fusion.py
│   └── ...
├── tests/
│   ├── test_api.py
│   ├── test_inference.py
│   ├── test_models.py
│   ├── test_preprocessing.py
│   └── ...
├── Dockerfile
├── [docker-compose.yml]
├── [prometheus.yml]
├── [alertmanager.yml]
├── [requirements.txt]
└── [README.md]