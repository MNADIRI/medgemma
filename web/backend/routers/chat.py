"""Chat and analysis routes – sends user message + selected slices to MedGemma."""

from fastapi import APIRouter, HTTPException, Request

from models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ChatRequest,
    ChatResponse,
    UsageInfo,
)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    """Run a chat completion with selected CT slices in context."""
    service = request.app.state.medgemma

    if service.sessions.get(body.session_id) is None:
        raise HTTPException(404, "Session not found. Upload DICOM files first.")

    if not body.selected_slices:
        raise HTTPException(400, "No slices selected. Select at least one slice.")

    # Convert ROI string keys to int for the service layer
    rois = {int(k): v.model_dump() for k, v in body.rois.items()} if body.rois else {}

    try:
        result = service.chat(
            session_id=body.session_id,
            user_message=body.message,
            selected_slices=body.selected_slices,
            rois=rois,
            history=[m.model_dump() for m in body.history],
        )
    except Exception as exc:
        raise HTTPException(500, f"Inference error: {exc}") from exc

    usage = UsageInfo(**(result.get("usage") or {}))
    return ChatResponse(response=result["response"], usage=usage)


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: Request, body: AnalyzeRequest):
    """Run structured lesion analysis on a single slice with ROI.

    Returns parsed XML blocks (CHAIN_OF_THOUGHT, REPORT, DIAGNOSIS).
    """
    service = request.app.state.medgemma

    if service.sessions.get(body.session_id) is None:
        raise HTTPException(404, "Session not found. Upload DICOM files first.")

    try:
        result = service.analyze(
            session_id=body.session_id,
            slice_index=body.slice_index,
            roi=body.roi.model_dump(),
        )
    except Exception as exc:
        raise HTTPException(500, f"Analysis error: {exc}") from exc

    usage = UsageInfo(**(result.get("usage") or {}))
    return AnalyzeResponse(
        chain_of_thought=result.get("chain_of_thought"),
        report=result.get("report"),
        diagnosis=result.get("diagnosis"),
        raw_response=result["raw_response"],
        usage=usage,
    )
