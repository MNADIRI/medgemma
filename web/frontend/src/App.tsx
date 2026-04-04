import { useCallback, useEffect, useMemo, useState } from "react";
import { analyzeRoi, segmentRoi, uploadDicom } from "./api/client";
import AnalysisPanel from "./components/AnalysisPanel";
import ChatPanel from "./components/ChatPanel";
import DicomDropZone from "./components/DicomDropZone";
import SliceViewer from "./components/SliceViewer";
import { useChat } from "./hooks/useChat";
import type { AnalysisResult, ROI, SeriesMetadata, SliceInfo } from "./types";

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
  const [roiMap, setRoiMap] = useState<Map<number, ROI>>(new Map());
  const [maskMap, setMaskMap] = useState<Map<number, string>>(new Map());

  // Analysis state
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState<string | null>(null);

  // Chat — carries analysis context for follow-up questions
  const { messages, isLoading, send, reset } = useChat(sessionId, selectedIndices, roiMap);

  // Total images for budget warning
  const totalImageCount = useMemo(() => {
    let count = selectedIndices.size;
    for (const idx of selectedIndices) {
      if (roiMap.has(idx)) count++;
    }
    return count;
  }, [selectedIndices, roiMap]);

  // Prevent browser default file-open on drop
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
    try {
      const res = await uploadDicom(files);
      setSessionId(res.session_id);
      setSlices(res.slices);
      setMetadata(res.metadata);
      setCurrentIndex(0);
      setSelectedIndices(new Set());
      setRoiMap(new Map());
      setMaskMap((prev) => {
        for (const url of prev.values()) URL.revokeObjectURL(url);
        return new Map();
      });
      setAnalysisResult(null);
      setAnalysisError(null);
      reset();
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Upload failed";
      if (msg === "Not Found" || msg === "Not Allowed") {
        setUploadError("Cannot reach backend server. Make sure the backend is running on port 8000.");
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

  const handleSetRoi = useCallback((index: number, roi: ROI | null) => {
    setRoiMap((prev) => {
      const next = new Map(prev);
      if (roi) next.set(index, roi);
      else next.delete(index);
      return next;
    });

    // Revoke old mask
    setMaskMap((prev) => {
      const old = prev.get(index);
      if (old) URL.revokeObjectURL(old);
      const next = new Map(prev);
      next.delete(index);
      return next;
    });

    // Clear analysis when ROI changes
    setAnalysisResult(null);
    setAnalysisError(null);

    // Request segmentation
    if (roi && sessionId) {
      segmentRoi(sessionId, index, roi).then((maskUrl) => {
        if (maskUrl) {
          setMaskMap((prev) => {
            const next = new Map(prev);
            next.set(index, maskUrl);
            return next;
          });
        }
      });
    }
  }, [sessionId]);

  const handleAnalyze = useCallback(async (sliceIndex: number) => {
    if (!sessionId) return;
    const roi = roiMap.get(sliceIndex);
    if (!roi) return;

    setIsAnalyzing(true);
    setAnalysisError(null);
    setAnalysisResult(null);

    // Ensure this slice is selected for follow-up chat
    setSelectedIndices((prev) => {
      if (prev.has(sliceIndex)) return prev;
      const next = new Set(prev);
      next.add(sliceIndex);
      return next;
    });

    try {
      const result = await analyzeRoi(sessionId, sliceIndex, roi);
      setAnalysisResult(result);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Analysis failed";
      setAnalysisError(msg);
    } finally {
      setIsAnalyzing(false);
    }
  }, [sessionId, roiMap]);

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
                roiMap={roiMap}
                maskMap={maskMap}
                onCurrentChange={setCurrentIndex}
                onToggleSelect={handleToggleSelect}
                onSelectAll={handleSelectAll}
                onDeselectAll={handleDeselectAll}
                onSetRoi={handleSetRoi}
                onAnalyze={handleAnalyze}
                isAnalyzing={isAnalyzing}
              />
              <DicomDropZone onUpload={handleUpload} isUploading={isUploading} />
            </>
          )}
        </div>

        {/* Right panel: Analysis + Chat stacked */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
          {/* Upper right: Analysis panel (only visible after analysis) */}
          {(analysisResult || isAnalyzing || analysisError) && (
            <div style={{ flex: "0 1 55%", padding: "16px 16px 8px 16px", overflow: "hidden", display: "flex", flexDirection: "column" }}>
              {isAnalyzing ? (
                <div
                  style={{
                    flex: 1,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    background: "#111",
                    borderRadius: 8,
                    border: "1px solid #333",
                    gap: 12,
                  }}
                >
                  <div style={{ fontSize: 14, color: "#888" }}>
                    <span className="dots">Analyzing lesion</span>
                    <style>{`
                      .dots::after {
                        content: '';
                        animation: dots 1.5s steps(4, end) infinite;
                      }
                      @keyframes dots {
                        0% { content: ''; }
                        25% { content: '.'; }
                        50% { content: '..'; }
                        75% { content: '...'; }
                      }
                    `}</style>
                  </div>
                  <div style={{ fontSize: 12, color: "#555" }}>
                    MedGemma is performing structured lesion analysis...
                  </div>
                </div>
              ) : analysisError ? (
                <div
                  style={{
                    flex: 1,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    background: "#1a0000",
                    borderRadius: 8,
                    border: "1px solid #3a0000",
                    color: "#e74c3c",
                    fontSize: 14,
                    padding: 16,
                  }}
                >
                  Analysis failed: {analysisError}
                </div>
              ) : analysisResult ? (
                <AnalysisPanel analysis={analysisResult} />
              ) : null}
            </div>
          )}

          {/* Lower right: Chat panel */}
          <div
            style={{
              flex: analysisResult || isAnalyzing || analysisError ? "1 1 45%" : "1",
              padding: analysisResult || isAnalyzing || analysisError ? "8px 16px 16px 16px" : 16,
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
            }}
          >
            <ChatPanel
              messages={messages}
              onSend={send}
              isLoading={isLoading}
              disabled={!sessionId}
              selectedCount={selectedIndices.size}
              totalImageCount={totalImageCount}
              hasAnalysis={!!analysisResult}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
