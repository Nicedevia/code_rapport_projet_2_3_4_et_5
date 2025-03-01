# monitoring.py (corrigé)

from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, Response

app = FastAPI()

# ✅ Définition du compteur Prometheus pour suivre le nombre de requêtes HTTP
request_counter = Counter("http_requests_total", "Nombre total de requêtes reçues")

@app.middleware("http")
async def count_requests(request, call_next):
    request_counter.inc()
    return await call_next(request)

@app.get("/metrics")
def metrics():
    """Exporter les métriques pour Prometheus"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
