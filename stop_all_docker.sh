#!/bin/bash

echo "🚀 Arrêt et suppression de tous les conteneurs..."
docker stop $(docker ps -aq) 2>/dev/null
docker rm $(docker ps -aq) 2>/dev/null

echo "🧹 Suppression des images Docker..."
docker rmi $(docker images -q) -f 2>/dev/null

echo "🗑️ Nettoyage des volumes et réseaux inutilisés..."
docker volume rm $(docker volume ls -q) 2>/dev/null
docker network prune -f 2>/dev/null

echo "✅ Tous les conteneurs, images, volumes et réseaux inutilisés ont été supprimés."
