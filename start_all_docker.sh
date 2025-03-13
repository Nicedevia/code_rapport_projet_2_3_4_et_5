#!/bin/bash

echo "🔄 Relance de tous les services Docker..."
docker-compose up --build -d

echo "✅ Vérification des conteneurs actifs..."
docker ps -a

echo "🎯 Tous les services ont été relancés avec succès ! 🚀"
