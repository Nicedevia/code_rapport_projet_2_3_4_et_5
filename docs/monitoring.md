# Monitoring avec Prometheus et Grafana

Ce document explique comment configurer et utiliser Prometheus et Grafana pour surveiller les métriques de votre projet.

---

## Table des matières

- [Introduction](#introduction)
- [Configuration de Prometheus](#configuration-de-prometheus)
- [Configuration de Grafana](#configuration-de-grafana)
- [Métriques disponibles](#métriques-disponibles)
- [Commandes Docker](#commandes-docker)
- [Exemple de Dashboard Grafana](#exemple-de-dashboard-grafana)

---

## Introduction

Prometheus est un outil de monitoring open-source qui collecte des métriques à partir de vos applications et services. Grafana est une plateforme de visualisation qui permet de créer des dashboards interactifs à partir des données collectées par Prometheus.

Dans ce projet, Prometheus est utilisé pour collecter des métriques sur les performances de l'API, et Grafana est utilisé pour visualiser ces métriques.

---

## Configuration de Prometheus

1. **Fichier de configuration `prometheus.yml`** :
   Le fichier `prometheus.yml` définit les cibles à surveiller et les règles de scraping.

   Exemple de configuration :
   ```yaml
   global:
     scrape_interval: 15s  # Intervalle de collecte des métriques

   scrape_configs:
     - job_name: "api"
       static_configs:
         - targets: ["localhost:8000"]  # Adresse de l'API

     - job_name: "docker"
       static_configs:
         - targets: ["localhost:9323"]  # Exporter Docker