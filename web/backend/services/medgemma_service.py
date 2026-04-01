"""MedGemma 1.5 4B service – model loading, session cache, and inference.

Loads the model once at startup, caches processed PIL images per session,
and constructs multi-slice conversations for each chat turn.
"""

import base64
import io
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import PIL.Image
import torch
import transformers

logger = logging.getLogger(__name__)

MODEL_ID = "google/medgemma-1.5-4b-it"
MAX_NEW_TOKENS = 2000


# ---------------------------------------------------------------------------
# Session cache
# ---------------------------------------------------------------------------

@dataclass
class SessionData:
    """Holds processed slice images for one upload session."""
    images: list[PIL.Image.Image] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """In-memory session store (keyed by session_id)."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionData] = {}

    def create(self, images: list[PIL.Image.Image], metadata: dict[str, Any]) -> str:
        sid = uuid.uuid4().hex
        self._sessions[sid] = SessionData(images=images, metadata=metadata)
        return sid

    def get(self, sid: str) -> SessionData | None:
        return self._sessions.get(sid)

    def delete(self, sid: str) -> None:
        self._sessions.pop(sid, None)


# ---------------------------------------------------------------------------
# Image encoding helpers
# ---------------------------------------------------------------------------

def _encode_pil_to_data_uri(img: PIL.Image.Image, fmt: str = "jpeg") -> str:
    """Encode a PIL image as a base64 data URI."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt};base64,{b64}"


# ---------------------------------------------------------------------------
# MedGemma service
# ---------------------------------------------------------------------------

class MedGemmaService:
    """Wraps HuggingFace model + processor for local inference."""

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device = None
        self.sessions = SessionManager()

    def load_model(self) -> None:
        """Load MedGemma 1.5 4B. Call once at startup."""
        logger.info("Loading MedGemma model %s …", MODEL_ID)

        # Determine device and dtype
        if torch.cuda.is_available():
            dtype = torch.float16
            device_map = "auto"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dtype = torch.float16
            device_map = "auto"
        else:
            dtype = torch.float32
            device_map = "cpu"

        self.processor = transformers.AutoProcessor.from_pretrained(
            MODEL_ID,
            use_fast=True,
        )
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self.device = self.model.device
        self._dtype = dtype
        logger.info("Model loaded on %s (dtype=%s)", self.device, dtype)

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        session_id: str,
        user_message: str,
        selected_slices: list[int],
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Run a chat turn with selected CT slices in context.

        Returns dict with 'response' and 'usage' keys.
        """
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown session: {session_id}")

        # Build conversation messages
        messages = self._build_messages(session, user_message, selected_slices, history)

        # Tokenise
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            continue_final_message=False,
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        )

        # Move to device
        inputs = inputs.to(self.model.device, dtype=self._dtype)
        input_len = inputs["input_ids"].shape[-1]

        # Generate
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        # Decode only the new tokens
        new_tokens = output_ids[0, input_len:]
        response_text = self.processor.decode(new_tokens, skip_special_tokens=True)
        output_len = len(new_tokens)

        return {
            "response": response_text.strip(),
            "usage": {
                "input_tokens": int(input_len),
                "output_tokens": int(output_len),
            },
        }

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        session: SessionData,
        user_message: str,
        selected_slices: list[int],
        history: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Build the HuggingFace chat message list with interleaved slices.

        Format follows the notebook pattern:
          instruction, [image, "SLICE N"]*, query
        """
        messages: list[dict[str, Any]] = []

        # Replay history (text-only for prior turns)
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Current user turn: images + question
        content: list[dict[str, Any]] = []

        # Add instruction preamble
        instruction = (
            "You are a medical AI assistant analyzing CT scan slices. "
            "The user has selected specific slices from a CT volume for your review. "
            "Each slice is labeled with its index number."
        )
        content.append({"type": "text", "text": instruction})

        # Add selected slice images with SLICE N markers
        for slice_idx in selected_slices:
            if 0 <= slice_idx < len(session.images):
                img = session.images[slice_idx]
                data_uri = _encode_pil_to_data_uri(img)
                content.append({"type": "image", "image": data_uri})
                content.append({"type": "text", "text": f"SLICE {slice_idx + 1}"})

        # Add user question
        content.append({"type": "text", "text": f"\n\n{user_message}"})

        messages.append({"role": "user", "content": content})
        return messages
