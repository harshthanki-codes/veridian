from __future__ import annotations

from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from src.settings import get_settings
from src.storage import get_store

settings = get_settings()

app = FastAPI(
    title="Veridian - Financial Risk Intelligence API",
    description="Real-time fraud detection for transaction streams. XGBoost + SHAP explainability.",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", uuid4().hex)
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.on_event("startup")
def startup() -> None:
    get_store().initialize()


app.include_router(router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
def root():
    return {"service": "veridian", "docs": "/docs"}
