# API de Classification Chat/Chien

Cette API REST, développée avec FastAPI, permet de classer des images (et ultérieurement des fichiers audio) pour déterminer s'il s'agit d'un chat ou d'un chien.

## Installation

1. Créez un environnement virtuel :

2. Installez les dépendances :

## Lancement de l'API

Dans le répertoire racine du projet, lancez :
uvicorn api.api:app --host 0.0.0.0 --port 8000 --reload
http://localhost:8000/docs