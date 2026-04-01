import type { ChatMessage, ChatResponse, UploadResponse } from "../types";

// In dev, Vite proxies /api -> localhost:8000. If proxy fails on large
// uploads, fall back to calling the backend directly.
const PROXY_BASE = "/api";
const DIRECT_BASE = "http://localhost:8000/api";

async function fetchWithFallback(
  path: string,
  init: RequestInit
): Promise<Response> {
  // Try through Vite proxy first
  try {
    const res = await fetch(`${PROXY_BASE}${path}`, init);
    // 502 = proxy error → retry direct
    if (res.status === 502) throw new Error("proxy 502");
    return res;
  } catch {
    // Fallback: call backend directly (bypasses proxy)
    console.warn("Proxy failed, calling backend directly");
    return fetch(`${DIRECT_BASE}${path}`, init);
  }
}

export async function uploadDicom(files: File[]): Promise<UploadResponse> {
  const form = new FormData();
  for (const f of files) {
    form.append("files", f);
  }

  let res: Response;
  try {
    res = await fetchWithFallback("/upload-dicom", { method: "POST", body: form });
  } catch (e) {
    throw new Error(
      "Cannot connect to backend. Make sure the backend is running: python3 main.py"
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
  history: ChatMessage[]
): Promise<ChatResponse> {
  let res: Response;
  try {
    res = await fetchWithFallback("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        message,
        selected_slices: selectedSlices,
        history,
      }),
    });
  } catch (e) {
    throw new Error(
      "Cannot connect to backend. Make sure the backend is running: python3 main.py"
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
  return `${PROXY_BASE}/slices/${sessionId}/${index}`;
}

export function sliceUrlDirect(sessionId: string, index: number): string {
  return `${DIRECT_BASE}/slices/${sessionId}/${index}`;
}
