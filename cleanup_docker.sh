#!/bin/bash

echo "🚀 [1/5] Arrêt et suppression des conteneurs..."
docker stop $(docker ps -aq) 2>/dev/null
docker rm $(docker ps -aq) 2>/dev/null

echo "🧹 [2/5] Suppression des images Docker..."
docker rmi $(docker images -q) -f 2>/dev/null

echo "🗑️ [3/5] Nettoyage des volumes et réseaux inutilisés..."
docker volume rm $(docker volume ls -q) 2>/dev/null
docker network prune -f 2>/dev/null

echo "🔄 [4/5] Reconstruction et relance de Docker Compose..."
docker-compose up --build -d

echo "✅ [5/5] Vérification des conteneurs actifs..."
docker ps -a

echo "🎯 Docker cleanup et rebuild terminé avec succès ! 🚀"

