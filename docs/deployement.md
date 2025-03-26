# Procédure de Déploiement : Local et DockerHub

Ce document décrit les étapes nécessaires pour déployer votre projet localement et sur DockerHub.

---

## Table des matières

- [Déploiement Local](#déploiement-local)
  - [Prérequis](#prérequis)
  - [Étapes de Déploiement Local](#étapes-de-déploiement-local)
- [Déploiement sur DockerHub](#déploiement-sur-dockerhub)
  - [Prérequis](#prérequis-1)
  - [Étapes de Déploiement sur DockerHub](#étapes-de-déploiement-sur-dockerhub)
- [Commandes Utiles](#commandes-utiles)

---

## Déploiement Local

### Prérequis

1. **Docker** : Assurez-vous que Docker est installé sur votre machine.
   - [Télécharger Docker](https://www.docker.com/products/docker-desktop)

2. **Docker Compose** : Docker Compose est généralement inclus avec Docker Desktop.

3. **Python** : Installez Python (version 3.9 ou supérieure) si vous souhaitez exécuter l'API sans Docker.

### Étapes de Déploiement Local

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/votre-utilisateur/projet_3.git
   cd projet_3# Procédure de Déploiement : Local et DockerHub

Ce document décrit les étapes nécessaires pour déployer votre projet localement et sur DockerHub.

---

## Table des matières

- [Déploiement Local](#déploiement-local)
  - [Prérequis](#prérequis)
  - [Étapes de Déploiement Local](#étapes-de-déploiement-local)
- [Déploiement sur DockerHub](#déploiement-sur-dockerhub)
  - [Prérequis](#prérequis-1)
  - [Étapes de Déploiement sur DockerHub](#étapes-de-déploiement-sur-dockerhub)
- [Commandes Utiles](#commandes-utiles)

---

## Déploiement Local

### Prérequis

1. **Docker** : Assurez-vous que Docker est installé sur votre machine.
   - [Télécharger Docker](https://www.docker.com/products/docker-desktop)

2. **Docker Compose** : Docker Compose est généralement inclus avec Docker Desktop.

3. **Python** : Installez Python (version 3.9 ou supérieure) si vous souhaitez exécuter l'API sans Docker.

### Étapes de Déploiement Local

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/votre-utilisateur/projet_3.git
   cd projet_3

2. ** Construire et lancer les conteneurs Docker ** :
docker-compose up --build -d

3. Vérifier les services en cours d'exécution :
docker ps

4. Accéder à l'API :

L'API sera disponible à l'adresse : http://localhost:8000
La documentation interactive Swagger est accessible à : http://localhost:8000/docs

5. Arrêter les conteneurs :

docker-compose down


***************

Déploiement sur DockerHub
Prérequis
Compte DockerHub :

Créez un compte sur DockerHub si vous n'en avez pas déjà un.
Connexion à DockerHub :

Connectez-vous à DockerHub depuis votre terminal
docker login

Fichier Dockerfile : bien configuré 

Étapes de Déploiement sur DockerHub
 1. Construire l'image Docker 
    docker build -t votre-utilisateur/projet_3:latest .
 2. Tester l'image localement
     docker run -p 8000:8000 votre-utilisateur/projet_3:latest
 3. Pousser l'image sur DockerHub :
    docker push votre-utilisateur/projet_3:latest
 4. Vérifier l'image sur DockerHub :

    Connectez-vous à DockerHub et vérifiez que l'image a bien été poussée.
 5. Déployer depuis DockerHub :

    Sur une autre machine ou un serveur, tirez l'image depuis DockerHub :
    docker run -p 8000:8000 votre-utilisateur/projet_3:latest

    