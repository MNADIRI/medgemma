import type { ChatMessage, ChatResponse, UploadResponse } from "../types";

const BASE = "/api";

export async function uploadDicom(files: File[]): Promise<UploadResponse> {
  const form = new FormData();
  for (const f of files) {
    form.append("files", f);
  }
  const res = await fetch(`${BASE}/upload-dicom`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Upload failed");
  }
  return res.json();
}

export async function sendChat(
  sessionId: string,
  message: string,
  selectedSlices: number[],
  history: ChatMessage[]
): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      message,
      selected_slices: selectedSlices,
      history,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Chat failed");
  }
  return res.json();
}

export function sliceUrl(sessionId: string, index: number): string {
  return `${BASE}/slices/${sessionId}/${index}`;
}
