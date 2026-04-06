"""FastAPI application for MedGemma CT Chat."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import anomaly, chat, dicom
from services.dinov2_codegraph_service import DINOv2CoDeGraphService
from services.medgemma_service import MedGemmaService
from services.medsam2_service import MedSAM2Service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

medgemma = MedGemmaService()
medsam2 = MedSAM2Service()
dinov2_codegraph = DINOv2CoDeGraphService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    medgemma.load_model()
    app.state.medgemma = medgemma

    medsam2.load_model()
    app.state.medsam2 = medsam2
    medgemma.medsam2 = medsam2  # Allow chat pipeline to use segmentation

    # DINOv2 loads lazily on first /detect-anomaly call (saves VRAM at startup)
    app.state.dinov2_codegraph = dinov2_codegraph

    logger.info("Services ready (MedGemma=%s, MedSAM2=%s, DINOv2=lazy)",
                medgemma.backend, "loaded" if medsam2.is_loaded else "disabled")
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
app.include_router(anomaly.router)


@app.get("/api/health")
async def health():
    ready = medgemma.model is not None or medgemma.hf_client is not None
    return {
        "status": "ok",
        "backend": medgemma.backend,
        "model_loaded": ready,
        "medsam2_loaded": medsam2.is_loaded,
        "dinov2_loaded": dinov2_codegraph.is_loaded,
    }


if __name__ == "__main__":
    import uvicorn

    # Log registered routes for debugging
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        if path and methods:
            logger.info("Route: %s %s", methods, path)

    port = int(os.environ.get("PORT", 8001))
    # Pass app object directly instead of string "main:app" to avoid
    # module re-import issues that can cause routes to go missing
    uvicorn.run(app, host="0.0.0.0", port=port)
