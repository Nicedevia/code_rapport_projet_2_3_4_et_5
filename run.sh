#!/bin/bash

echo "📁 Vérification du dossier logs..."
mkdir -p logs

echo "🧼 Suppression de l'ancien rapport s'il existe..."
rm -f logs/incident_report.md

echo "🛠️ Génération du rapport d'incident..."
python logs/incident_report.py

echo "📬 Envoi d'alerte mail si erreurs détectées..."
python logs/email_alert.py

echo "🚀 Lancement de l'API FastAPI sur http://localhost:8000 ..."
python -m api.api
