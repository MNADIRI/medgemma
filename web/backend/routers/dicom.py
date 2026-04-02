"""DICOM upload and slice serving routes."""

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import Response

from models.schemas import SeriesMetadata, SliceInfo, UploadResponse
from services.dicom_processor import CTDicomProcessor

router = APIRouter(prefix="/api", tags=["dicom"])

dicom_processor = CTDicomProcessor()


@router.post("/upload-dicom", response_model=UploadResponse)
async def upload_dicom(request: Request, files: list[UploadFile]):
    """Upload one or more DICOM files, process them, and return session info."""
    if not files:
        raise HTTPException(400, "No DICOM files uploaded")

    # Read file contents
    file_contents: list[tuple[str, bytes]] = []
    for f in files:
        raw = await f.read()
        file_contents.append((f.filename or "unknown.dcm", raw))

    try:
        slices, meta = dicom_processor.process_files(file_contents)
    except Exception as exc:
        raise HTTPException(422, f"Failed to process DICOM files: {exc}") from exc

    # Store both model images (RGB windowed) and display images (grayscale)
    service = request.app.state.medgemma
    model_images = [s.model_image for s in slices]
    display_images = [s.display_image for s in slices]
    session_id = service.sessions.create(model_images, meta, display_images=display_images)

    # Build response
    slice_infos = [
        SliceInfo(
            index=s.index,
            position=s.position,
            preview_url=f"/api/slices/{session_id}/{i}",
            instance_number=s.metadata.get("instance_number"),
            slice_location=str(s.metadata.get("slice_location")) if s.metadata.get("slice_location") else None,
        )
        for i, s in enumerate(slices)
    ]

    return UploadResponse(
        session_id=session_id,
        num_slices=len(slices),
        slices=slice_infos,
        metadata=SeriesMetadata(**meta),
    )


@router.get("/slices/{session_id}/{index}")
async def get_slice(request: Request, session_id: str, index: int):
    """Return a JPEG preview of a single slice (grayscale for display)."""
    service = request.app.state.medgemma
    session = service.sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    if index < 0 or index >= len(session.display_images):
        raise HTTPException(404, "Slice index out of range")

    import io
    img = session.display_images[index]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return Response(content=buf.getvalue(), media_type="image/jpeg")
