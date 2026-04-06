"""Anomaly detection endpoints — DINOv2 + CoDeGraph3D."""

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from models.schemas import AutoROI, DetectAnomalyRequest, DetectAnomalyResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["anomaly"])


@router.post("/detect-anomaly", response_model=DetectAnomalyResponse)
async def detect_anomaly(req: DetectAnomalyRequest, request: Request):
    """Run DINOv2 + CoDeGraph3D anomaly detection on the full CT volume."""
    medgemma = request.app.state.medgemma
    dinov2 = request.app.state.dinov2_codegraph

    session = medgemma.sessions.get(req.session_id)
    if session is None:
        raise HTTPException(404, "Session not found")

    try:
        result = dinov2.detect_anomaly(session)
    except RuntimeError as exc:
        raise HTTPException(503, f"DINOv2 not available: {exc}")
    except Exception as exc:
        logger.error("Anomaly detection failed: %s", exc, exc_info=True)
        raise HTTPException(500, f"Anomaly detection failed: {exc}")

    # Store in session for heatmap retrieval
    session.anomaly_result = result

    # Convert auto_rois keys to strings and values to AutoROI for Pydantic
    auto_rois_str = {
        str(k): AutoROI(**v)
        for k, v in result["auto_rois"].items()
    }

    return DetectAnomalyResponse(
        top_slices=result["top_slices"],
        auto_rois=auto_rois_str,
        slice_scores=result["slice_scores"],
    )


@router.get("/anomaly-heatmap/{session_id}/{index}")
async def anomaly_heatmap(session_id: str, index: int, request: Request):
    """Return a semi-transparent PNG heatmap overlay for a single slice."""
    medgemma = request.app.state.medgemma
    dinov2 = request.app.state.dinov2_codegraph

    session = medgemma.sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")

    if session.anomaly_result is None:
        raise HTTPException(404, "No anomaly detection result — run /detect-anomaly first")

    anomaly_volume = session.anomaly_result["anomaly_volume"]
    if index < 0 or index >= anomaly_volume.shape[0]:
        raise HTTPException(400, f"Slice index {index} out of range [0, {anomaly_volume.shape[0]})")

    slice_map = anomaly_volume[index]
    png_bytes = dinov2.render_heatmap_png(slice_map)

    return Response(content=png_bytes, media_type="image/png")
