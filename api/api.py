from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import (
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    Gauge,
)
import time
import os

# Import du routeur de l'API
from .routes import router as api_router

# Logger configuré avec rotation
from logs.log_config import setup_logger
logger = setup_logger("main_logger", "logs/app.log")

# -------------------------------
# Création de l'application FastAPI
# -------------------------------
app = FastAPI(
    title="API de Classification Chat/Chien",
    description="API REST pour la classification multimodale (images, audio et fusion) de chats et chiens",
    version="1.0.0"
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion du routeur
app.include_router(api_router)

# -------------------------------
# Prometheus Metrics
# -------------------------------
registry = CollectorRegistry()

REQUEST_COUNTER = Counter(
    "http_requests_total",
    "Nombre total de requêtes reçues",
    ["method", "endpoint", "http_status"],
    registry=registry
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Durée des requêtes HTTP",
    ["method", "endpoint"],
    registry=registry
)

IN_PROGRESS = Gauge(
    "inprogress_requests",
    "Nombre de requêtes actuellement en cours de traitement",
    registry=registry
)

# -------------------------------
# Middleware Prometheus + Logs
# -------------------------------
@app.middleware("http")
async def prometheus_metrics(request: Request, call_next):
    IN_PROGRESS.inc()
    start_time = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(duration)
        REQUEST_COUNTER.labels(method=request.method, endpoint=request.url.path, http_status=500).inc()
        IN_PROGRESS.dec()
        logger.exception(f"Erreur {request.method} {request.url.path} - {str(e)}")
        raise e
    duration = time.time() - start_time
    REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(duration)
    REQUEST_COUNTER.labels(method=request.method, endpoint=request.url.path, http_status=response.status_code).inc()
    IN_PROGRESS.dec()
    logger.info(f"{request.method} {request.url.path} -> {response.status_code}")
    return response


# -------------------------------
# Endpoint des métriques
# -------------------------------
@app.get("/metrics")
def metrics():
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

# -------------------------------
# Lancement local (optionnel)
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True)
