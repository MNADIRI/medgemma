import React, { useCallback } from "react";
import { anomalyHeatmapUrl, sliceUrl } from "../api/client";
import type { AnomalyResult, ROI, SliceInfo } from "../types";
import RoiCanvas from "./RoiCanvas";

interface Props {
  sessionId: string;
  slices: SliceInfo[];
  currentIndex: number;
  selectedIndices: Set<number>;
  roiMap: Map<number, ROI>;
  maskMap: Map<number, string>;
  onCurrentChange: (index: number) => void;
  onToggleSelect: (index: number) => void;
  onSelectAll: () => void;
  onDeselectAll: () => void;
  onSetRoi: (index: number, roi: ROI | null) => void;
  onAnalyze?: (index: number) => void;
  isAnalyzing?: boolean;
  // Anomaly detection props
  anomalyResult?: AnomalyResult | null;
  isDetecting?: boolean;
  detectError?: string | null;
  showHeatmap?: boolean;
  onDetectAnomaly?: () => void;
  onToggleHeatmap?: () => void;
  onAcceptAutoRoi?: (sliceIndex: number) => void;
}

export default function SliceViewer({
  sessionId,
  slices,
  currentIndex,
  selectedIndices,
  roiMap,
  maskMap,
  onCurrentChange,
  onToggleSelect,
  onSelectAll,
  onDeselectAll,
  onSetRoi,
  onAnalyze,
  isAnalyzing = false,
  anomalyResult = null,
  isDetecting = false,
  detectError = null,
  showHeatmap = true,
  onDetectAnomaly,
  onToggleHeatmap,
  onAcceptAutoRoi,
}: Props) {
  const current = slices[currentIndex];
  const isSelected = selectedIndices.has(currentIndex);

  // Check if current slice has an auto-detected ROI (not yet accepted)
  const hasAutoRoi =
    anomalyResult?.auto_rois?.[String(currentIndex)] != null &&
    !roiMap.has(currentIndex);

  const handleSlider = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      onCurrentChange(Number(e.target.value));
    },
    [onCurrentChange]
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {/* Main slice image */}
      <div
        style={{
          position: "relative",
          background: "#000",
          borderRadius: 8,
          overflow: "hidden",
          border: isSelected ? "2px solid #4a90d9" : "2px solid #333",
        }}
      >
        <img
          src={sliceUrl(sessionId, currentIndex)}
          alt={`Slice ${currentIndex + 1}`}
          style={{ width: "100%", display: "block", imageRendering: "auto" }}
        />
        {/* Anomaly heatmap overlay */}
        {anomalyResult && showHeatmap && (
          <img
            src={anomalyHeatmapUrl(sessionId, currentIndex)}
            alt="Anomaly heatmap"
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              height: "100%",
              pointerEvents: "none",
              opacity: 0.8,
            }}
          />
        )}
        {/* MedSAM2 segmentation mask overlay */}
        {maskMap.has(currentIndex) && (
          <img
            src={maskMap.get(currentIndex)}
            alt="Segmentation mask"
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              height: "100%",
              pointerEvents: "none",
            }}
          />
        )}
        <RoiCanvas
          currentRoi={roiMap.get(currentIndex) ?? null}
          onRoiChange={(roi) => onSetRoi(currentIndex, roi)}
          disabled={!isSelected}
        />
        {/* Slice info overlay */}
        <div
          style={{
            position: "absolute",
            top: 8,
            left: 8,
            color: "#0f0",
            fontSize: 12,
            fontFamily: "monospace",
            textShadow: "0 0 4px #000",
          }}
        >
          Slice {currentIndex + 1}/{slices.length}
          {current?.position != null && ` | z=${current.position.toFixed(1)}`}
          {anomalyResult && (
            <span style={{ color: "#ff6b6b" }}>
              {" "}| anom={anomalyResult.slice_scores[currentIndex]?.toFixed(2) ?? "?"}
            </span>
          )}
        </div>
        {isSelected && (
          <div style={{ position: "absolute", top: 8, right: 8, display: "flex", gap: 4 }}>
            {roiMap.has(currentIndex) && (
              <div
                style={{
                  background: "#e67e22",
                  color: "#fff",
                  padding: "2px 8px",
                  borderRadius: 4,
                  fontSize: 11,
                  fontWeight: 600,
                }}
              >
                ROI
              </div>
            )}
            <div
              style={{
                background: "#4a90d9",
                color: "#fff",
                padding: "2px 8px",
                borderRadius: 4,
                fontSize: 11,
                fontWeight: 600,
              }}
            >
              SELECTED
            </div>
          </div>
        )}
      </div>

      {/* Auto-ROI confirmation banner */}
      {hasAutoRoi && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "8px 12px",
            background: "#1a1030",
            border: "1px solid #7c3aed",
            borderRadius: 6,
            fontSize: 13,
          }}
        >
          <span style={{ color: "#c4b5fd", flex: 1 }}>
            Anomaly detected on this slice
          </span>
          <button
            onClick={() => onAcceptAutoRoi?.(currentIndex)}
            style={{
              padding: "5px 14px",
              borderRadius: 5,
              border: "none",
              background: "#7c3aed",
              color: "#fff",
              cursor: "pointer",
              fontSize: 12,
              fontWeight: 600,
            }}
          >
            Accept ROI
          </button>
          <button
            onClick={() => {
              // Select the slice so user can draw manually
              if (!isSelected) onToggleSelect(currentIndex);
            }}
            style={{
              padding: "5px 14px",
              borderRadius: 5,
              border: "1px solid #555",
              background: "transparent",
              color: "#aaa",
              cursor: "pointer",
              fontSize: 12,
            }}
          >
            Adjust manually
          </button>
        </div>
      )}

      {/* Detect error */}
      {detectError && (
        <div
          style={{
            padding: "6px 12px",
            background: "#1a0000",
            border: "1px solid #3a0000",
            borderRadius: 6,
            color: "#e74c3c",
            fontSize: 12,
          }}
        >
          {detectError}
        </div>
      )}

      {/* Redo / Analyze buttons — shown when mask is visible */}
      {maskMap.has(currentIndex) && roiMap.has(currentIndex) && (
        <div style={{ display: "flex", gap: 8, justifyContent: "center" }}>
          <button
            onClick={() => onSetRoi(currentIndex, null)}
            disabled={isAnalyzing}
            style={{
              padding: "8px 20px",
              borderRadius: 6,
              border: "1px solid #555",
              background: "#222",
              color: "#ccc",
              cursor: isAnalyzing ? "not-allowed" : "pointer",
              fontSize: 13,
              fontWeight: 600,
            }}
          >
            Redo
          </button>
          <button
            onClick={() => onAnalyze?.(currentIndex)}
            disabled={isAnalyzing || !onAnalyze}
            style={{
              padding: "8px 24px",
              borderRadius: 6,
              border: "none",
              background: isAnalyzing ? "#333" : "#2d6a2d",
              color: "#fff",
              cursor: isAnalyzing ? "not-allowed" : "pointer",
              fontSize: 13,
              fontWeight: 600,
            }}
          >
            {isAnalyzing ? "Analyzing..." : "Analyze"}
          </button>
        </div>
      )}

      {/* Slider */}
      <input
        type="range"
        min={0}
        max={slices.length - 1}
        value={currentIndex}
        onChange={handleSlider}
        style={{ width: "100%" }}
      />

      {/* Selection controls + Auto-detect button */}
      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <button
          onClick={() => onToggleSelect(currentIndex)}
          style={{
            padding: "6px 14px",
            borderRadius: 6,
            border: "none",
            background: isSelected ? "#c0392b" : "#4a90d9",
            color: "#fff",
            cursor: "pointer",
            fontSize: 13,
          }}
        >
          {isSelected ? "Deselect" : "Select"} slice {currentIndex + 1}
        </button>
        <button
          onClick={onSelectAll}
          style={{
            padding: "6px 12px",
            borderRadius: 6,
            border: "1px solid #555",
            background: "transparent",
            color: "#aaa",
            cursor: "pointer",
            fontSize: 12,
          }}
        >
          Select all
        </button>
        <button
          onClick={onDeselectAll}
          style={{
            padding: "6px 12px",
            borderRadius: 6,
            border: "1px solid #555",
            background: "transparent",
            color: "#aaa",
            cursor: "pointer",
            fontSize: 12,
          }}
        >
          Clear
        </button>

        {/* Auto-detect anomaly button */}
        {onDetectAnomaly && (
          <button
            onClick={onDetectAnomaly}
            disabled={isDetecting}
            style={{
              padding: "6px 14px",
              borderRadius: 6,
              border: "none",
              background: isDetecting ? "#333" : "#7c3aed",
              color: "#fff",
              cursor: isDetecting ? "not-allowed" : "pointer",
              fontSize: 12,
              fontWeight: 600,
            }}
          >
            {isDetecting ? "Detecting..." : "Auto-detect"}
          </button>
        )}

        {/* Heatmap toggle */}
        {anomalyResult && onToggleHeatmap && (
          <button
            onClick={onToggleHeatmap}
            style={{
              padding: "6px 12px",
              borderRadius: 6,
              border: "1px solid #555",
              background: showHeatmap ? "#4a2080" : "transparent",
              color: showHeatmap ? "#c4b5fd" : "#888",
              cursor: "pointer",
              fontSize: 12,
            }}
          >
            {showHeatmap ? "Hide heatmap" : "Show heatmap"}
          </button>
        )}

        <span style={{ color: "#888", fontSize: 12, marginLeft: "auto" }}>
          {selectedIndices.size} slice{selectedIndices.size !== 1 ? "s" : ""} selected
          {roiMap.size > 0 && ` (${roiMap.size} ROI)`}
        </span>
      </div>

      {/* Thumbnail strip with selection and anomaly indicators */}
      <div
        style={{
          display: "flex",
          gap: 3,
          overflowX: "auto",
          padding: "4px 0",
          maxHeight: 80,
        }}
      >
        {slices.map((_, i) => {
          const anomalyScore = anomalyResult?.slice_scores?.[i] ?? 0;
          const hasAnomaly = anomalyResult?.auto_rois?.[String(i)] != null;

          return (
            <div
              key={i}
              onClick={() => onCurrentChange(i)}
              style={{
                position: "relative",
                minWidth: 6,
                height: 40,
                background: i === currentIndex ? "#4a90d9" : selectedIndices.has(i) ? "#2d6aa0" : "#333",
                borderRadius: 2,
                cursor: "pointer",
                border: selectedIndices.has(i) ? "1px solid #4a90d9" : "1px solid transparent",
              }}
              title={`Slice ${i + 1}${selectedIndices.has(i) ? " (selected)" : ""}${roiMap.has(i) ? " (ROI)" : ""}${hasAnomaly ? ` (anomaly: ${anomalyScore.toFixed(2)})` : ""}`}
            >
              {roiMap.has(i) && (
                <div
                  style={{
                    position: "absolute",
                    top: -2,
                    right: -2,
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: "#e67e22",
                    border: "1px solid #111",
                  }}
                />
              )}
              {/* Anomaly indicator dot (purple) */}
              {hasAnomaly && !roiMap.has(i) && (
                <div
                  style={{
                    position: "absolute",
                    top: -2,
                    right: -2,
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: "#7c3aed",
                    border: "1px solid #111",
                  }}
                />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
