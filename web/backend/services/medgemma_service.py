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
MAX_NEW_TOKENS_ANALYZE = 2048  # Structured analysis needs more tokens for 3 JSON blocks
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
    hu_arrays: list = field(default_factory=list)                        # raw HU arrays (np.ndarray float64)
    pixel_spacings: list = field(default_factory=list)                   # (row_mm, col_mm) per slice
    slice_metadata: list = field(default_factory=list)                   # per-slice DICOM metadata dicts
    metadata: dict[str, Any] = field(default_factory=dict)
    anomaly_result: dict | None = None                                   # DINOv2+CoDeGraph3D anomaly detection result


class SessionManager:
    """In-memory session store (keyed by session_id)."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionData] = {}

    def create(
        self,
        images: list[PIL.Image.Image],
        metadata: dict[str, Any],
        display_images: list[PIL.Image.Image] | None = None,
        hu_arrays: list | None = None,
        pixel_spacings: list | None = None,
        slice_metadata: list | None = None,
    ) -> str:
        sid = uuid.uuid4().hex
        self._sessions[sid] = SessionData(
            images=images,
            display_images=display_images or images,
            hu_arrays=hu_arrays or [],
            pixel_spacings=pixel_spacings or [],
            slice_metadata=slice_metadata or [],
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


def _parse_analysis_response(text: str) -> dict[str, Any]:
    """Parse the structured XML text blocks from MedGemma's analysis response.

    Extracts <LOCALISATION>, <ASPECT>, and <DIAGNOSIS> blocks as plain text.
    MedGemma 4B cannot reliably produce JSON, so we use simple text extraction
    and parse the diagnosis entries with regex.
    """
    import re

    result: dict[str, Any] = {
        "localisation": None,
        "aspect": None,
        "diagnosis": None,
    }

    # Extract text content from XML tags
    for tag in ("LOCALISATION", "ASPECT", "DIAGNOSIS"):
        pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            result[tag.lower()] = match.group(1).strip()

    # If tags weren't found, try to extract from raw text by section headers
    if not result["localisation"]:
        m = re.search(r"(?:LOCALI[SZ]ATION|Step\s*1)[:\s—-]*(.*?)(?=(?:ASPECT|Step\s*2|DIAGNOS|<)|$)", text, re.DOTALL | re.IGNORECASE)
        if m:
            result["localisation"] = m.group(1).strip()

    if not result["aspect"]:
        m = re.search(r"(?:ASPECT|CHARACTERI[SZ]|Step\s*2)[:\s—-]*(.*?)(?=(?:DIAGNOS|Step\s*3|<)|$)", text, re.DOTALL | re.IGNORECASE)
        if m:
            result["aspect"] = m.group(1).strip()

    if not result["diagnosis"]:
        m = re.search(r"(?:DIAGNOS\w*|Step\s*3)[:\s—-]*(.*?)$", text, re.DOTALL | re.IGNORECASE)
        if m:
            result["diagnosis"] = m.group(1).strip()

    # Parse individual diagnosis entries from the diagnosis text
    result["diagnosis_entries"] = _parse_diagnosis_entries(result.get("diagnosis") or "")

    return result


def _parse_diagnosis_entries(text: str) -> list[dict[str, str]]:
    """Extract ranked diagnosis entries from diagnosis text.

    Handles formats like:
      1. LIKELY: Name. Supporting: ... Against: ...
      1. **Likely** — Name: ...
      - Likely: Name ...
    """
    import re

    entries = []
    # Split on numbered items (1., 2., 3.) or bullet points
    parts = re.split(r"\n\s*(?:\d+[\.\)]\s*|[-•]\s*)", text)
    # Also try to match lines starting with tier keywords
    if len(parts) <= 1:
        parts = re.split(r"\n\s*(?=(?:LIKELY|POSSIBLE|UNLIKELY|Most likely|Possible|Unlikely))", text, flags=re.IGNORECASE)

    tier_map = {
        "likely": "likely",
        "most likely": "likely",
        "possible": "possible",
        "unlikely": "unlikely_but_to_exclude",
        "unlikely but to exclude": "unlikely_but_to_exclude",
    }

    for part in parts:
        part = part.strip()
        if not part or len(part) < 10:
            continue

        # Try to extract tier
        tier = "possible"  # default
        for keyword, tier_val in tier_map.items():
            if keyword.lower() in part.lower()[:50]:
                tier = tier_val
                break

        # Try to extract label (diagnosis name)
        # Pattern: TIER: Label. Supporting: ...
        label_match = re.search(
            r"(?:LIKELY|POSSIBLE|UNLIKELY[^:]*)[:\s—-]+\*?\*?([^.:\n]+)",
            part, re.IGNORECASE
        )
        label = label_match.group(1).strip().strip("*").strip() if label_match else part[:80]

        # Extract supporting features
        support_match = re.search(r"[Ss]upporting[:\s]+(.*?)(?=[Aa]gainst|$)", part, re.DOTALL)
        supporting = support_match.group(1).strip() if support_match else ""

        # Extract against features
        against_match = re.search(r"[Aa]gainst[:\s]+(.*?)$", part, re.DOTALL)
        against = against_match.group(1).strip() if against_match else ""

        entries.append({
            "tier": tier,
            "label": label,
            "supporting": supporting,
            "against": against,
            "raw": part,
        })

    return entries[:3]  # At most 3


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
    # Analyze (structured lesion analysis — single slice + ROI)
    # ------------------------------------------------------------------

    def analyze(
        self,
        session_id: str,
        slice_index: int,
        roi: dict,
    ) -> dict[str, Any]:
        """Run structured lesion analysis on a single slice with ROI.

        Uses the analysis system prompt. Returns parsed text blocks
        plus the quantitative ROI data from our pipeline.
        """
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown session: {session_id}")

        if slice_index < 0 or slice_index >= len(session.images):
            raise ValueError(f"Slice index {slice_index} out of range")

        messages, roi_data = self._build_analyze_messages(session, slice_index, roi)

        logger.info("Analyze (%s): slice %d with ROI", self.backend, slice_index)

        try:
            if self.backend == "remote":
                result = self._infer_remote_analyze(messages)
            else:
                result = self._infer_local_analyze(messages)
        except Exception:
            logger.error("Analyze inference failed:\n%s", traceback.format_exc())
            raise

        # Parse text blocks from model response
        raw = result["response"]
        parsed = _parse_analysis_response(raw)

        return {
            "localisation": parsed.get("localisation"),
            "aspect": parsed.get("aspect"),
            "diagnosis_text": parsed.get("diagnosis"),
            "diagnosis_entries": parsed.get("diagnosis_entries", []),
            "roi_data": roi_data,
            "raw_response": raw,
            "usage": result["usage"],
        }

    def _build_analyze_messages(
        self,
        session: SessionData,
        slice_index: int,
        roi: dict,
        use_pil: bool | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """Build messages for structured analysis.

        Returns (messages, roi_data) — roi_data is the quantitative dict
        from our pipeline (sent to frontend independently of model output).
        """
        from services.prompts import LESION_ANALYSIS_SYSTEM_PROMPT
        from services.roi_analysis import extract_roi_data, format_roi_data

        if use_pil is None:
            use_pil = self.backend != "remote"

        content: list[dict[str, Any]] = []
        roi_data: dict[str, Any] | None = None

        # System instruction
        content.append({"type": "text", "text": LESION_ANALYSIS_SYSTEM_PROMPT})

        def _append_image(img: PIL.Image.Image) -> None:
            if use_pil:
                content.append({"type": "image", "image": img})
            else:
                content.append({"type": "image", "image": _encode_pil_to_data_uri(img)})

        # Full slice image
        img = session.images[slice_index]
        _append_image(img)

        # Segment and isolate ROI
        display_img = session.display_images[slice_index]
        mask = None

        if self.medsam2 is not None and self.medsam2.is_loaded:
            try:
                mask, isolated_img = self.medsam2.segment_and_isolate(img, display_img, roi)
                _append_image(isolated_img)
                logger.info("Analyze: MedSAM2 segmented ROI → isolated %dx%d", isolated_img.width, isolated_img.height)
            except Exception as exc:
                logger.warning("Analyze: MedSAM2 failed: %s, falling back to crop", exc)
                cropped = self._crop_roi(img, roi)
                _append_image(cropped)
        else:
            cropped = self._crop_roi(img, roi)
            _append_image(cropped)

        # Extract quantitative ROI data
        roi_data_text = ""
        if mask is not None and slice_index < len(session.hu_arrays):
            hu = session.hu_arrays[slice_index]
            ps = session.pixel_spacings[slice_index]
            ipp = None
            iop = None
            if slice_index < len(session.slice_metadata):
                smeta = session.slice_metadata[slice_index]
                ipp = smeta.get("image_position_patient")
                iop = smeta.get("image_orientation_patient")
            try:
                roi_data = extract_roi_data(
                    hu, mask, ps,
                    image_position_patient=ipp,
                    image_orientation_patient=iop,
                )
                roi_data_text = format_roi_data(roi_data)
                logger.info(
                    "Analyze: ROI data extracted (mean=%.1f HU, area=%.1f mm²)",
                    roi_data["density"]["mean"],
                    roi_data["morphometry"]["area_mm2"],
                )
            except Exception as exc:
                logger.warning("Analyze: ROI data extraction failed: %s", exc)

        # Assemble prompt text
        prompt = "Full CT slice + segmented lesion above."
        if roi_data_text:
            prompt += f"\n\n{roi_data_text}"
        else:
            prompt += "\n\n(Quantitative ROI data unavailable — analyze visually.)"
        prompt += "\n\nPerform structured lesion analysis following the methodology above."

        content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}], roi_data

    def _infer_local_analyze(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Run local inference for structured analysis (higher token limit)."""
        import torch

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        logger.info("Analyze input tokens: %d", input_len)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS_ANALYZE,
            )

        new_tokens = output_ids[0, input_len:]
        response_text = self.processor.decode(new_tokens, skip_special_tokens=True)
        response_text = _clean_thinking_tokens(response_text)
        output_len = len(new_tokens)
        logger.info("Analyze output tokens: %d", output_len)

        return {
            "response": response_text.strip(),
            "usage": {"input_tokens": int(input_len), "output_tokens": int(output_len)},
        }

    def _infer_remote_analyze(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Run remote inference for structured analysis."""
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

        kwargs: dict[str, Any] = {
            "messages": openai_messages,
            "max_tokens": MAX_NEW_TOKENS_ANALYZE,
        }
        if os.environ.get("MEDGEMMA_PROVIDER"):
            kwargs["model"] = self._remote_model
        output = self.hf_client.chat_completion(**kwargs)

        choice = output.choices[0]
        usage = output.usage
        response_text = choice.message.content or ""

        return {
            "response": response_text.strip(),
            "usage": {
                "input_tokens": usage.prompt_tokens if usage else None,
                "output_tokens": usage.completion_tokens if usage else None,
            },
        }

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
        """Build the chat message list with interleaved slices, ROI images, and ROI data.

        When MedSAM2 + HU data are available, the prompt includes:
          - LESION_ANALYSIS_SYSTEM_PROMPT as system instruction
          - Full slice image + segmented lesion image per ROI
          - <ROI_DATA> block with quantitative measurements
          - User question

        Falls back to generic instruction + simple crop when MedSAM2 is unavailable.
        """
        from services.prompts import (
            GENERIC_INSTRUCTION,
            GENERIC_ROI_CROP_ADDENDUM,
            GENERIC_ROI_SEGMENTED_ADDENDUM,
            LESION_ANALYSIS_SYSTEM_PROMPT,
        )
        from services.roi_analysis import extract_roi_data, format_roi_data

        messages: list[dict[str, Any]] = []

        # Replay history
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": [{"type": "text", "text": msg["content"]}],
            })

        # Current user turn
        content: list[dict[str, Any]] = []

        has_segmentation = (
            self.medsam2 is not None
            and self.medsam2.is_loaded
            and bool(rois)
        )
        has_hu_data = bool(session.hu_arrays) and bool(session.pixel_spacings)

        # Use structured chain-of-thought prompt when we have full quantitative pipeline
        use_structured_prompt = has_segmentation and has_hu_data

        # System instruction
        if use_structured_prompt:
            content.append({"type": "text", "text": LESION_ANALYSIS_SYSTEM_PROMPT})
        else:
            instruction = GENERIC_INSTRUCTION
            if rois:
                instruction += (
                    GENERIC_ROI_SEGMENTED_ADDENDUM if has_segmentation
                    else GENERIC_ROI_CROP_ADDENDUM
                )
            content.append({"type": "text", "text": instruction})

        def _append_image(img: PIL.Image.Image) -> None:
            if use_pil:
                content.append({"type": "image", "image": img})
            else:
                content.append({"type": "image", "image": _encode_pil_to_data_uri(img)})

        # Add slice images with ROI processing
        for slice_idx in selected_slices:
            if slice_idx < 0 or slice_idx >= len(session.images):
                continue

            img = session.images[slice_idx]
            _append_image(img)

            roi = rois.get(slice_idx)
            if roi:
                if has_segmentation:
                    display_img = session.display_images[slice_idx]
                    try:
                        mask, isolated_img = self.medsam2.segment_and_isolate(
                            img, display_img, roi
                        )
                        logger.info(
                            "Slice %d: MedSAM2 segmented ROI → isolated %dx%d",
                            slice_idx, isolated_img.width, isolated_img.height,
                        )
                        _append_image(isolated_img)

                        # Extract and inject quantitative ROI data
                        if use_structured_prompt and slice_idx < len(session.hu_arrays):
                            hu = session.hu_arrays[slice_idx]
                            ps = session.pixel_spacings[slice_idx]
                            # Get spatial DICOM metadata for this slice
                            ipp = None
                            iop = None
                            if slice_idx < len(session.slice_metadata):
                                smeta = session.slice_metadata[slice_idx]
                                ipp = smeta.get("image_position_patient")
                                iop = smeta.get("image_orientation_patient")
                            try:
                                roi_data = extract_roi_data(
                                    hu, mask, ps,
                                    image_position_patient=ipp,
                                    image_orientation_patient=iop,
                                )
                                roi_block = format_roi_data(roi_data)
                                content.append({"type": "text", "text": (
                                    f"SLICE {slice_idx + 1} — full view + segmented lesion above.\n\n"
                                    f"{roi_block}"
                                )})
                                logger.info(
                                    "Slice %d: ROI data extracted (mean=%.1f HU, area=%.1f mm²)",
                                    slice_idx,
                                    roi_data["density"]["mean"],
                                    roi_data["morphometry"]["area_mm2"],
                                )
                            except Exception as exc:
                                logger.warning("ROI data extraction failed for slice %d: %s", slice_idx, exc)
                                content.append({"type": "text", "text": (
                                    f"SLICE {slice_idx + 1} — full view + segmented lesion above "
                                    f"(quantitative data unavailable)"
                                )})
                        else:
                            content.append({"type": "text", "text": (
                                f"SLICE {slice_idx + 1} — full view + segmented lesion above"
                            )})

                    except Exception as exc:
                        logger.warning("MedSAM2 failed for slice %d: %s, falling back to crop", slice_idx, exc)
                        cropped = self._crop_roi(img, roi)
                        _append_image(cropped)
                        content.append({"type": "text", "text": (
                            f"SLICE {slice_idx + 1} — full view + ROI crop above "
                            f"(segmentation unavailable)"
                        )})
                else:
                    cropped = self._crop_roi(img, roi)
                    _append_image(cropped)
                    content.append({"type": "text", "text": (
                        f"SLICE {slice_idx + 1} — full view + ROI crop above"
                    )})
            else:
                content.append({"type": "text", "text": f"SLICE {slice_idx + 1}"})

        # User question
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
