from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import router as api_router
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter
from fastapi.responses import Response

app = FastAPI(
    title="API de Classification Chat/Chien",
    description="API REST pour la classification multimodale (images, audio et fusion) de chats et chiens",
    version="1.0.0"
)

# Middleware CORS (à adapter en production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclusion du routeur
app.include_router(api_router)

# ✅ Ajouter un compteur Prometheus pour suivre les requêtes
request_counter = Counter("http_requests_total", "Nombre total de requêtes reçues")

@app.middleware("http")
async def count_requests(request, call_next):
    request_counter.inc()
    return await call_next(request)

@app.get("/metrics")
def metrics():
    """Exporter les métriques pour Prometheus"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ✅ Garder la ligne de lancement d’Uvicorn en bas
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.api:app", host="0.0.0.0", port=8000, reload=True)
