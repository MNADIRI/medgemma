"""Pydantic request/response schemas for the API."""

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# DICOM Upload
# ---------------------------------------------------------------------------

class SliceInfo(BaseModel):
    index: int
    position: float
    preview_url: str
    instance_number: int | None = None
    slice_location: str | None = None


class SeriesMetadata(BaseModel):
    patient_id: str | None = None
    patient_name: str | None = None
    study_description: str | None = None
    series_description: str | None = None
    modality: str | None = None
    slice_thickness: str | None = None
    study_date: str | None = None
    rows: int | None = None
    columns: int | None = None


class UploadResponse(BaseModel):
    session_id: str
    num_slices: int
    slices: list[SliceInfo]
    metadata: SeriesMetadata


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ROIRegion(BaseModel):
    """Normalized ROI coordinates (0-1) relative to the image dimensions."""
    x: float       # left edge
    y: float       # top edge
    width: float
    height: float


class ChatRequest(BaseModel):
    session_id: str
    message: str
    selected_slices: list[int]  # indices of slices to send to the model
    rois: dict[str, ROIRegion] = {}  # key = slice index as string
    history: list[ChatMessage] = []


class UsageInfo(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class ChatResponse(BaseModel):
    response: str
    usage: UsageInfo | None = None
