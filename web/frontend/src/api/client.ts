import type { ChatMessage, ChatResponse, ROI, UploadResponse } from "../types";

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
