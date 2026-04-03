"""MedGemma 1.5 4B service – model loading, session cache, and inference.

Loads the model once at startup, caches processed PIL images per session,
and constructs multi-slice conversations for each chat turn.

Supports two backends (set via MEDGEMMA_BACKEND env var):
  - "local"  : loads model locally (default)
  - "remote" : uses HuggingFace Inference API (needs HF_TOKEN)
"""

import base64
import io
import logging
import os
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any

import PIL.Image

logger = logging.getLogger(__name__)

MODEL_ID = "google/medgemma-1.5-4b-it"
MAX_NEW_TOKENS = 512
# Max *images* per prompt — T4 (15GB) can handle ~4 in bfloat16.
# Each full slice = 1 image; each ROI crop = 1 additional image.
MAX_IMAGES_PER_PROMPT = 4
INFERENCE_BACKEND = os.environ.get("MEDGEMMA_BACKEND", "local").lower()

def _total_ram_gb() -> float:
    """Return total system RAM in GB (works on macOS and Linux)."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    # Fallback: read from sysctl (macOS) or /proc/meminfo (Linux)
    try:
        import subprocess
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return int(out.strip()) / (1024 ** 3)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) / (1024 ** 2)
    except Exception:
        pass
    return 16.0  # assume enough RAM if we can't detect


# ---------------------------------------------------------------------------
# Session cache
# ---------------------------------------------------------------------------

@dataclass
class SessionData:
    """Holds processed slice images for one upload session."""
    images: list[PIL.Image.Image] = field(default_factory=list)          # model images (RGB windowed)
    display_images: list[PIL.Image.Image] = field(default_factory=list)  # display images (grayscale)
    metadata: dict[str, Any] = field(default_factory=dict)


class SessionManager:
    """In-memory session store (keyed by session_id)."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionData] = {}

    def create(
        self,
        images: list[PIL.Image.Image],
        metadata: dict[str, Any],
        display_images: list[PIL.Image.Image] | None = None,
    ) -> str:
        sid = uuid.uuid4().hex
        self._sessions[sid] = SessionData(
            images=images,
            display_images=display_images or images,
            metadata=metadata,
        )
        return sid

    def get(self, sid: str) -> SessionData | None:
        return self._sessions.get(sid)

    def delete(self, sid: str) -> None:
        self._sessions.pop(sid, None)


# ---------------------------------------------------------------------------
# Image encoding helpers
# ---------------------------------------------------------------------------

def _clean_thinking_tokens(text: str) -> str:
    """Remove MedGemma's internal thinking/reasoning blocks from output.

    The model sometimes generates <unused94>thought...reasoning...</unused94>
    blocks before or within the actual response. Strip them out.
    """
    import re
    # Remove <unused94>thought ... </unused94> blocks (thinking tokens)
    text = re.sub(r"<unused\d+>thought.*?(?:</unused\d+>|$)", "", text, flags=re.DOTALL)
    # Remove any remaining <unused*> tags
    text = re.sub(r"<unused\d+>", "", text)
    return text.strip()


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
    """Wraps HuggingFace model + processor for local or remote inference."""

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device = None
        self.backend = INFERENCE_BACKEND  # "local" or "remote"
        self.hf_client = None
        self.sessions = SessionManager()
        self.medsam2 = None  # Set externally after MedSAM2 loads

    def load_model(self) -> None:
        """Load model locally or connect to HF Inference API."""
        logger.info("Backend mode: %s", self.backend)

        if self.backend == "remote":
            self._init_remote()
            return

        self._init_local()

    def _init_remote(self) -> None:
        """Set up remote inference client.

        Supports three configurations via env vars:
          1. MEDGEMMA_ENDPOINT_URL  → HF Inference Endpoint (dedicated GPU)
          2. MEDGEMMA_PROVIDER      → Third-party provider (e.g. novita, fireworks)
          3. Neither                → HF serverless API (may not work for gated models)

        All require HF_TOKEN.
        """
        from huggingface_hub import InferenceClient

        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "HF_TOKEN environment variable is required for remote mode. "
                "Get your token at https://huggingface.co/settings/tokens"
            )

        endpoint_url = os.environ.get("MEDGEMMA_ENDPOINT_URL")
        provider = os.environ.get("MEDGEMMA_PROVIDER")

        if endpoint_url:
            # Dedicated HF Inference Endpoint
            self.hf_client = InferenceClient(
                model=endpoint_url, token=token, timeout=120,
            )
            logger.info("Remote mode: using dedicated endpoint at %s", endpoint_url)
        elif provider:
            # Third-party provider (novita, fireworks, together, etc.)
            self.hf_client = InferenceClient(
                provider=provider, token=token, timeout=120,
            )
            logger.info("Remote mode: using provider '%s' for %s", provider, MODEL_ID)
        else:
            # Default: HF serverless API
            self.hf_client = InferenceClient(
                model=MODEL_ID, token=token, timeout=120,
            )
            logger.info("Remote mode: using HF serverless API for %s", MODEL_ID)

        self._remote_model = MODEL_ID

    def _init_local(self) -> None:
        """Load model locally with RAM-aware strategy."""
        import torch
        import transformers

        logger.info("Loading MedGemma model %s …", MODEL_ID)
        ram_gb = _total_ram_gb()
        self._low_ram = ram_gb <= 12
        logger.info("System RAM: %.1f GB — low_ram mode: %s", ram_gb, self._low_ram)

        self.processor = transformers.AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )

        if self._low_ram:
            self._load_for_low_ram()
        else:
            self._load_for_high_ram()

    def _load_for_low_ram(self) -> None:
        """Load model for ≤12 GB RAM systems (e.g. 8 GB Mac).

        Loads in float16 directly on CPU — no MPS (OOM), no device_map
        (causes unmapped-layer errors with quantization on Apple Silicon).
        macOS unified memory + swap keeps it alive. Inference is slow
        (~1-3 min) but stable.
        """
        import torch
        import transformers

        logger.info("Low-RAM mode: loading model in float16 on CPU…")
        self.model = transformers.AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = torch.device("cpu")
        self._dtype = torch.float16
        logger.info("Model loaded on CPU (float16). Inference will be slow but stable.")

    def _load_for_high_ram(self) -> None:
        """Load model for >12 GB RAM systems on MPS/CUDA."""
        import torch
        import transformers

        if torch.cuda.is_available():
            device = torch.device("cuda")
            # MedGemma is trained in bfloat16 — use it directly.
            # float16 conversion can corrupt weights (different value ranges).
            # T4 emulates bf16 via float32 internally — slower but correct.
            dtype = torch.bfloat16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            # MPS does not support bfloat16
            dtype = torch.float16
        else:
            device = torch.device("cpu")
            dtype = torch.bfloat16

        try:
            logger.info("Loading weights to %s (dtype=%s)…", device, dtype)
            self.model = transformers.AutoModelForImageTextToText.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            self.model = self.model.to(device)
            self.model.eval()
            self.device = device
            self._dtype = dtype
            logger.info("Model loaded on %s (dtype=%s)", self.device, dtype)
        except (RuntimeError, torch.mps.OutOfMemoryError if hasattr(torch, "mps") else RuntimeError) as e:
            logger.warning("OOM on %s (%s), falling back to low-RAM strategy.", device, e)
            self._load_for_low_ram()

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(
        self,
        session_id: str,
        user_message: str,
        selected_slices: list[int],
        rois: dict[int, dict] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Run a chat turn with selected CT slices in context.

        rois maps slice index → {x, y, width, height} in normalized 0-1 coords.
        Returns dict with 'response' and 'usage' keys.
        """
        if rois is None:
            rois = {}
        if history is None:
            history = []

        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown session: {session_id}")

        # Budget: count total images (each slice = 1, each ROI = +1)
        max_images = MAX_IMAGES_PER_PROMPT
        total_images = len(selected_slices) + sum(
            1 for idx in selected_slices if idx in rois
        )

        if total_images > max_images:
            # Prioritize slices that have ROIs (user explicitly marked them)
            with_roi = [s for s in selected_slices if s in rois]
            without_roi = [s for s in selected_slices if s not in rois]

            kept: list[int] = []
            budget = max_images

            # First, keep slices with ROIs (cost 2 each)
            for s in with_roi:
                if budget >= 2:
                    kept.append(s)
                    budget -= 2
                else:
                    break

            # Fill remaining budget with non-ROI slices (cost 1 each)
            for s in without_roi:
                if budget >= 1:
                    kept.append(s)
                    budget -= 1
                else:
                    break

            kept.sort()
            logger.warning(
                "Image budget exceeded (%d > %d), trimmed to %d slices (%d with ROIs)",
                total_images, max_images, len(kept),
                sum(1 for s in kept if s in rois),
            )
            selected_slices = kept
            # Remove ROIs for slices that were dropped
            rois = {k: v for k, v in rois.items() if k in selected_slices}

        logger.info(
            "Chat (%s): %d slices, %d ROIs, %d history messages",
            self.backend, len(selected_slices), len(rois), len(history),
        )

        try:
            if self.backend == "remote":
                return self._infer_remote(session, user_message, selected_slices, rois, history)
            return self._infer_local(session, user_message, selected_slices, rois, history)
        except Exception:
            logger.error("Inference failed:\n%s", traceback.format_exc())
            raise

    # ------------------------------------------------------------------
    # Inference backends
    # ------------------------------------------------------------------

    def _infer_local(
        self,
        session: SessionData,
        user_message: str,
        selected_slices: list[int],
        rois: dict[int, dict],
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Run inference using locally loaded model."""
        import torch

        messages = self._build_messages(session, user_message, selected_slices, rois, history, use_pil=True)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move to model device — do NOT force dtype, let the model handle conversions
        # (forcing float16 on pixel_values can cause degenerate output on some GPUs)
        inputs = inputs.to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        logger.info("Input tokens: %d", input_len)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        new_tokens = output_ids[0, input_len:]
        response_text = self.processor.decode(new_tokens, skip_special_tokens=True)
        output_len = len(new_tokens)

        # Clean up thinking/reasoning blocks that MedGemma sometimes generates
        response_text = _clean_thinking_tokens(response_text)

        logger.info("Output tokens: %d, response length: %d chars", output_len, len(response_text))

        return {
            "response": response_text.strip(),
            "usage": {
                "input_tokens": int(input_len),
                "output_tokens": int(output_len),
            },
        }

    def _infer_remote(
        self,
        session: SessionData,
        user_message: str,
        selected_slices: list[int],
        rois: dict[int, dict],
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Run inference via HuggingFace Inference API."""
        messages = self._build_messages(session, user_message, selected_slices, rois, history)

        # Convert to OpenAI-compatible format for HF Inference API
        openai_messages = []
        for msg in messages:
            new_content = []
            for block in msg["content"]:
                if block.get("type") == "image":
                    new_content.append({
                        "type": "image_url",
                        "image_url": {"url": block["image"]},
                    })
                else:
                    new_content.append(block)
            openai_messages.append({"role": msg["role"], "content": new_content})

        logger.info("Sending request to HF Inference API…")
        kwargs: dict[str, Any] = {
            "messages": openai_messages,
            "max_tokens": MAX_NEW_TOKENS,
        }
        # When using a provider (not a dedicated endpoint), pass model ID
        if os.environ.get("MEDGEMMA_PROVIDER"):
            kwargs["model"] = self._remote_model
        output = self.hf_client.chat_completion(**kwargs)

        choice = output.choices[0]
        usage = output.usage
        response_text = choice.message.content or ""

        logger.info(
            "HF API response: %d input tokens, %d output tokens",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )

        return {
            "response": response_text.strip(),
            "usage": {
                "input_tokens": usage.prompt_tokens if usage else None,
                "output_tokens": usage.completion_tokens if usage else None,
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
        rois: dict[int, dict],
        history: list[dict[str, str]],
        use_pil: bool = False,
    ) -> list[dict[str, Any]]:
        """Build the chat message list with interleaved slices and ROI crops.

        Format follows the notebook pattern:
          instruction, [image, (roi_image)?, "SLICE N"]*, query

        For slices with ROIs, both the full slice and the cropped region are
        sent so SigLip encodes both at full resolution, giving the model a
        zoomed-in view of the lesion alongside the full anatomical context.

        When use_pil=True (local inference), images are passed as PIL objects.
        When use_pil=False (remote inference), images are passed as data URIs.
        """
        messages: list[dict[str, Any]] = []

        # Replay history using list content format for consistency
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["content"]}],
            })

        # Current user turn: images + question
        content: list[dict[str, Any]] = []

        # Check if MedSAM2 segmentation is available
        has_segmentation = (
            self.medsam2 is not None
            and self.medsam2.is_loaded
            and bool(rois)
        )

        # Add instruction preamble
        instruction = (
            "You are a medical AI assistant analyzing CT scan slices. "
            "The user has selected specific slices from a CT volume for your review. "
            "Each slice is labeled with its index number."
        )
        if rois:
            if has_segmentation:
                instruction += (
                    " Some slices include a segmented region of interest (ROI) where "
                    "MedSAM2 has isolated the lesion from the surrounding tissue. "
                    "The segmented image shows only the lesion pixels (on a black "
                    "background) cropped from the ROI area. Analyze the lesion "
                    "morphology, density, and borders in detail and provide a "
                    "differential diagnosis."
                )
            else:
                instruction += (
                    " Some slices include a cropped region of interest (ROI) that the "
                    "radiologist has highlighted for focused analysis. When an ROI is "
                    "provided, describe the lesion within it in detail and suggest a "
                    "differential diagnosis."
                )
        content.append({"type": "text", "text": instruction})

        def _append_image(img: PIL.Image.Image) -> None:
            if use_pil:
                content.append({"type": "image", "image": img})
            else:
                content.append({"type": "image", "image": _encode_pil_to_data_uri(img)})

        # Add selected slice images with SLICE N markers
        for slice_idx in selected_slices:
            if 0 <= slice_idx < len(session.images):
                img = session.images[slice_idx]
                _append_image(img)

                roi = rois.get(slice_idx)
                if roi:
                    if has_segmentation:
                        # Use MedSAM2 to segment the lesion, then send isolated image
                        display_img = session.display_images[slice_idx]
                        try:
                            _, isolated_img = self.medsam2.segment_and_isolate(
                                img, display_img, roi
                            )
                            _append_image(isolated_img)
                            content.append({"type": "text", "text": (
                                f"SLICE {slice_idx + 1} — full view above, "
                                f"segmented lesion ROI above (MedSAM2 isolated)"
                            )})
                        except Exception as exc:
                            logger.warning("MedSAM2 failed for slice %d: %s, falling back to crop", slice_idx, exc)
                            # Fall back to simple crop
                            cropped = self._crop_roi(img, roi)
                            _append_image(cropped)
                            content.append({"type": "text", "text": (
                                f"SLICE {slice_idx + 1} — full view above, "
                                f"ROI crop above (segmentation unavailable)"
                            )})
                    else:
                        # Simple crop fallback when MedSAM2 not available
                        cropped = self._crop_roi(img, roi)
                        _append_image(cropped)
                        w, h = img.size
                        left = int(roi["x"] * w)
                        top = int(roi["y"] * h)
                        right = int((roi["x"] + roi["width"]) * w)
                        bottom = int((roi["y"] + roi["height"]) * h)
                        content.append({"type": "text", "text": (
                            f"SLICE {slice_idx + 1} — full view above, "
                            f"ROI detail above (region {left},{top} to {right},{bottom})"
                        )})
                else:
                    content.append({"type": "text", "text": f"SLICE {slice_idx + 1}"})

        # Add user question
        content.append({"type": "text", "text": f"\n\n{user_message}"})

        messages.append({"role": "user", "content": content})
        return messages

    @staticmethod
    def _crop_roi(img: PIL.Image.Image, roi: dict) -> PIL.Image.Image:
        """Crop an ROI region from a PIL image using normalized coordinates."""
        w, h = img.size
        left = max(0, int(roi["x"] * w))
        top = max(0, int(roi["y"] * h))
        right = min(w, int((roi["x"] + roi["width"]) * w))
        bottom = min(h, int((roi["y"] + roi["height"]) * h))
        return img.crop((left, top, right, bottom))
