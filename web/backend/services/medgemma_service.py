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
# Max slices to send in a single prompt (memory safety for MPS/GPU)
MAX_SLICES_PER_PROMPT = 10
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
    """Wraps HuggingFace model + processor for local or remote inference."""

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device = None
        self.backend = INFERENCE_BACKEND  # "local" or "remote"
        self.hf_client = None
        self.sessions = SessionManager()

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
            dtype = torch.float16
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            dtype = torch.float16
        else:
            device = torch.device("cpu")
            dtype = torch.float32

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
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Run a chat turn with selected CT slices in context.

        Returns dict with 'response' and 'usage' keys.
        """
        session = self.sessions.get(session_id)
        if session is None:
            raise ValueError(f"Unknown session: {session_id}")

        # Limit number of slices to avoid OOM (local) or huge payloads (remote)
        max_slices = 3 if getattr(self, "_low_ram", False) else MAX_SLICES_PER_PROMPT
        if len(selected_slices) > max_slices:
            logger.warning(
                "Too many slices selected (%d), sampling %d uniformly",
                len(selected_slices), max_slices,
            )
            step = len(selected_slices) / max_slices
            selected_slices = [
                selected_slices[int(i * step)]
                for i in range(max_slices)
            ]

        logger.info(
            "Chat (%s): %d slices, %d history messages",
            self.backend, len(selected_slices), len(history),
        )

        try:
            if self.backend == "remote":
                return self._infer_remote(session, user_message, selected_slices, history)
            return self._infer_local(session, user_message, selected_slices, history)
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
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Run inference using locally loaded model."""
        import torch

        # Local processor expects PIL Image objects, not data URIs
        messages = self._build_messages(session, user_message, selected_slices, history, use_pil=True)

        # Debug: log message structure
        for msg in messages:
            for block in msg.get("content", []):
                if block.get("type") == "image":
                    img = block.get("image")
                    logger.info("Image block: type=%s, size=%s, mode=%s",
                                type(img).__name__,
                                getattr(img, "size", "N/A"),
                                getattr(img, "mode", "N/A"))

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Debug: log what keys are in inputs and pixel_values shape
        logger.info("Input keys: %s", list(inputs.keys()))
        if "pixel_values" in inputs:
            logger.info("pixel_values shape: %s, dtype: %s", inputs["pixel_values"].shape, inputs["pixel_values"].dtype)
        else:
            logger.warning("NO pixel_values in inputs — images were not processed!")

        # Move tensors to model device — cast only floating point tensors
        device = self.model.device
        model_dtype = self._dtype
        for k, v in inputs.items():
            if hasattr(v, "to"):
                if v.is_floating_point():
                    inputs[k] = v.to(device, dtype=model_dtype)
                else:
                    inputs[k] = v.to(device)
        input_len = inputs["input_ids"].shape[-1]
        # Check for image placeholder tokens in input
        input_ids = inputs["input_ids"][0].tolist()
        # Gemma3 uses token_id 262144 for <image_soft_token> placeholder
        image_token_count = sum(1 for t in input_ids if t >= 262000)
        logger.info("Input tokens: %d (image placeholder tokens: %d)", input_len, image_token_count)
        if image_token_count == 0:
            logger.warning("No image placeholder tokens found! The processor may not have processed the image.")
            # Log first 20 token IDs for debugging
            logger.info("First 20 token IDs: %s", input_ids[:20])

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        new_tokens = output_ids[0, input_len:]
        response_text = self.processor.decode(new_tokens, skip_special_tokens=True)
        # Also decode WITHOUT skipping special tokens to see what's generated
        raw_text = self.processor.decode(new_tokens, skip_special_tokens=False)
        output_len = len(new_tokens)

        logger.info("Output tokens: %d, response length: %d chars", output_len, len(response_text))
        logger.info("Response preview: %.200s", response_text.strip())
        logger.info("Raw output preview (with special tokens): %.200s", raw_text[:200])

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
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Run inference via HuggingFace Inference API."""
        messages = self._build_messages(session, user_message, selected_slices, history)

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
        history: list[dict[str, str]],
        use_pil: bool = False,
    ) -> list[dict[str, Any]]:
        """Build the chat message list with interleaved slices.

        Format follows the notebook pattern:
          instruction, [image, "SLICE N"]*, query

        All messages use the list-of-dicts content format for consistency
        with the Gemma3 chat template.

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
                if use_pil:
                    content.append({"type": "image", "image": img})
                else:
                    data_uri = _encode_pil_to_data_uri(img)
                    content.append({"type": "image", "image": data_uri})
                content.append({"type": "text", "text": f"SLICE {slice_idx + 1}"})

        # Add user question
        content.append({"type": "text", "text": f"\n\n{user_message}"})

        messages.append({"role": "user", "content": content})
        return messages
