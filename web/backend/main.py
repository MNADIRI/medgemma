"""FastAPI application for MedGemma CT Chat."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import chat, dicom
from services.medgemma_service import MedGemmaService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

medgemma = MedGemmaService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load MedGemma model on startup."""
    medgemma.load_model()
    app.state.medgemma = medgemma
    logger.info("MedGemma service ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="MedGemma CT Chat",
    description="Drag & drop DICOM CT scans, chat with MedGemma 1.5 4B",
    lifespan=lifespan,
)

# CORS for local dev — allow all origins since frontend calls backend directly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dicom.router)
app.include_router(chat.router)


@app.get("/api/health")
async def health():
    return {"status": "ok", "model_loaded": medgemma.model is not None}


if __name__ == "__main__":
    import uvicorn

    # Log registered routes for debugging
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if path and methods:
            logger.info("Route: %s %s", methods, path)

    port = int(os.environ.get("PORT", 8000))
    # Pass app object directly instead of string "main:app" to avoid
    # module re-import issues that can cause routes to go missing
    uvicorn.run(app, host="0.0.0.0", port=port)
