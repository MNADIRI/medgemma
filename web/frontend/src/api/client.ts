import type { AnalysisResult, AnomalyResult, ChatMessage, ChatResponse, ROI, UploadResponse } from "../types";

// Backend URL — defaults to localhost, override with VITE_BACKEND_URL for remote (e.g. Colab)
const BASE = import.meta.env.VITE_BACKEND_URL || "http://localhost:8001/api";

export async function uploadDicom(files: File[]): Promise<UploadResponse> {
  const form = new FormData();
  for (const f of files) {
    form.append("files", f);
  }

  let res: Response;
  try {
    res = await fetch(`${BASE}/upload-dicom`, { method: "POST", body: form });
  } catch (e) {
    throw new Error(
      "Cannot connect to backend. Make sure the backend is running: cd ~/medgemma/web/backend && python3 main.py"
    );
  }

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      detail = res.statusText || detail;
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function sendChat(
  sessionId: string,
  message: string,
  selectedSlices: number[],
  history: ChatMessage[],
  rois: Record<string, ROI> = {}
): Promise<ChatResponse> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        message,
        selected_slices: selectedSlices,
        rois,
        history,
      }),
    });
  } catch (e) {
    throw new Error(
      "Cannot connect to backend. Make sure the backend is running: cd ~/medgemma/web/backend && python3 main.py"
    );
  }

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      detail = res.statusText || detail;
    }
    throw new Error(detail);
  }
  return res.json();
}

export function sliceUrl(sessionId: string, index: number): string {
  return `${BASE}/slices/${sessionId}/${index}`;
}

/**
 * Run structured lesion analysis on a single slice with ROI.
 * Returns parsed analysis blocks (report, diagnosis, chain_of_thought).
 */
export async function analyzeRoi(
  sessionId: string,
  sliceIndex: number,
  roi: ROI
): Promise<AnalysisResult> {
  // Analysis can take 30-120s on T4 — use a generous timeout
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 300_000); // 5 min

  let res: Response;
  try {
    res = await fetch(`${BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        slice_index: sliceIndex,
        roi,
      }),
      signal: controller.signal,
    });
  } catch (e) {
    clearTimeout(timeout);
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new Error("Analysis timed out (>5 min). The model may be overloaded.");
    }
    throw new Error("Cannot connect to backend for analysis. Check that the backend is running.");
  } finally {
    clearTimeout(timeout);
  }

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      detail = res.statusText || detail;
    }
    throw new Error(detail);
  }
  return res.json();
}

/**
 * Request MedSAM2 segmentation for a ROI on a slice.
 * Returns a blob URL for the semi-transparent PNG mask overlay, or null if unavailable.
 */
export async function segmentRoi(
  sessionId: string,
  sliceIndex: number,
  roi: ROI
): Promise<string | null> {
  try {
    const res = await fetch(`${BASE}/segment`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        slice_index: sliceIndex,
        roi,
      }),
    });
    if (!res.ok) return null; // 503 = MedSAM2 not loaded, graceful fallback
    const blob = await res.blob();
    return URL.createObjectURL(blob);
  } catch {
    return null; // Network error, graceful fallback
  }
}

/**
 * Run DINOv2 + CoDeGraph3D anomaly detection on the full CT volume.
 * Returns top anomaly slices and auto-generated ROIs.
 */
export async function detectAnomaly(sessionId: string): Promise<AnomalyResult> {
  // Anomaly detection can take 15-60s — use a generous timeout
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 300_000); // 5 min

  let res: Response;
  try {
    res = await fetch(`${BASE}/detect-anomaly`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
      signal: controller.signal,
    });
  } catch (e) {
    clearTimeout(timeout);
    if (e instanceof DOMException && e.name === "AbortError") {
      throw new Error("Anomaly detection timed out (>5 min).");
    }
    throw new Error("Cannot connect to backend for anomaly detection.");
  } finally {
    clearTimeout(timeout);
  }

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      detail = body.detail || detail;
    } catch {
      detail = res.statusText || detail;
    }
    throw new Error(detail);
  }
  return res.json();
}

/** URL for the anomaly heatmap overlay PNG for a specific slice. */
export function anomalyHeatmapUrl(sessionId: string, index: number): string {
  return `${BASE}/anomaly-heatmap/${sessionId}/${index}`;
}
