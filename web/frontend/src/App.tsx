import { useCallback, useEffect, useState } from "react";
import { uploadDicom } from "./api/client";
import ChatPanel from "./components/ChatPanel";
import DicomDropZone from "./components/DicomDropZone";
import SliceViewer from "./components/SliceViewer";
import { useChat } from "./hooks/useChat";
import type { SeriesMetadata, SliceInfo } from "./types";

export default function App() {
  // Session state
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [slices, setSlices] = useState<SliceInfo[]>([]);
  const [metadata, setMetadata] = useState<SeriesMetadata | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Viewer state
  const [currentIndex, setCurrentIndex] = useState(0);
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set());

  // Chat
  const { messages, isLoading, send, reset } = useChat(sessionId, selectedIndices);

  // Prevent browser default file-open on drop ANYWHERE on the page
  useEffect(() => {
    const prevent = (e: DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
    };
    document.addEventListener("dragover", prevent);
    document.addEventListener("drop", prevent);
    return () => {
      document.removeEventListener("dragover", prevent);
      document.removeEventListener("drop", prevent);
    };
  }, []);

  const handleUpload = useCallback(async (files: File[]) => {
    setIsUploading(true);
    setUploadError(null);
    console.log(`Uploading ${files.length} files:`, files.map((f) => f.name));
    try {
      const res = await uploadDicom(files);
      setSessionId(res.session_id);
      setSlices(res.slices);
      setMetadata(res.metadata);
      setCurrentIndex(0);
      setSelectedIndices(new Set());
      reset();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Upload failed";
      console.error("Upload error:", msg);
      if (msg === "Not Found" || msg === "Not Allowed") {
        setUploadError(
          "Cannot reach backend server. Make sure the backend is running on port 8000: python3 main.py"
        );
      } else {
        setUploadError(msg);
      }
    } finally {
      setIsUploading(false);
    }
  }, [reset]);

  const handleToggleSelect = useCallback((index: number) => {
    setSelectedIndices((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  }, []);

  const handleSelectAll = useCallback(() => {
    setSelectedIndices(new Set(slices.map((_, i) => i)));
  }, [slices]);

  const handleDeselectAll = useCallback(() => {
    setSelectedIndices(new Set());
  }, []);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        background: "#0a0a0a",
        color: "#eee",
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      }}
    >
      {/* Header */}
      <header
        style={{
          padding: "12px 24px",
          borderBottom: "1px solid #222",
          display: "flex",
          alignItems: "center",
          gap: 12,
        }}
      >
        <h1 style={{ fontSize: 18, margin: 0, fontWeight: 700 }}>MedGemma CT Chat</h1>
        {metadata && (
          <span style={{ color: "#888", fontSize: 13 }}>
            {metadata.modality} | {metadata.series_description || metadata.study_description || "CT Series"}
            {metadata.study_date && ` | ${metadata.study_date}`}
            {` | ${slices.length} slices`}
          </span>
        )}
      </header>

      {/* Main content */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        {/* Left panel: DICOM viewer */}
        <div
          style={{
            width: "42%",
            padding: 16,
            display: "flex",
            flexDirection: "column",
            gap: 16,
            overflowY: "auto",
            borderRight: "1px solid #222",
          }}
        >
          {!sessionId ? (
            <>
              <DicomDropZone onUpload={handleUpload} isUploading={isUploading} />
              {uploadError && (
                <div style={{ color: "#e74c3c", fontSize: 14, padding: 8, background: "#1a0000", borderRadius: 8, border: "1px solid #3a0000" }}>
                  {uploadError}
                </div>
              )}
            </>
          ) : (
            <>
              <SliceViewer
                sessionId={sessionId}
                slices={slices}
                currentIndex={currentIndex}
                selectedIndices={selectedIndices}
                onCurrentChange={setCurrentIndex}
                onToggleSelect={handleToggleSelect}
                onSelectAll={handleSelectAll}
                onDeselectAll={handleDeselectAll}
              />
              {/* Re-upload button */}
              <DicomDropZone onUpload={handleUpload} isUploading={isUploading} />
            </>
          )}
        </div>

        {/* Right panel: Chat */}
        <div style={{ flex: 1, padding: 16, display: "flex", flexDirection: "column" }}>
          <ChatPanel
            messages={messages}
            onSend={send}
            isLoading={isLoading}
            disabled={!sessionId}
            selectedCount={selectedIndices.size}
          />
        </div>
      </div>
    </div>
  );
}
