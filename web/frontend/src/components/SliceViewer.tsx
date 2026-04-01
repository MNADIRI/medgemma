import React, { useCallback } from "react";
import { sliceUrl } from "../api/client";
import type { SliceInfo } from "../types";

interface Props {
  sessionId: string;
  slices: SliceInfo[];
  currentIndex: number;
  selectedIndices: Set<number>;
  onCurrentChange: (index: number) => void;
  onToggleSelect: (index: number) => void;
  onSelectAll: () => void;
  onDeselectAll: () => void;
}

export default function SliceViewer({
  sessionId,
  slices,
  currentIndex,
  selectedIndices,
  onCurrentChange,
  onToggleSelect,
  onSelectAll,
  onDeselectAll,
}: Props) {
  const current = slices[currentIndex];
  const isSelected = selectedIndices.has(currentIndex);

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
        </div>
        {isSelected && (
          <div
            style={{
              position: "absolute",
              top: 8,
              right: 8,
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
        )}
      </div>

      {/* Slider */}
      <input
        type="range"
        min={0}
        max={slices.length - 1}
        value={currentIndex}
        onChange={handleSlider}
        style={{ width: "100%" }}
      />

      {/* Selection controls */}
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
        <span style={{ color: "#888", fontSize: 12, marginLeft: "auto" }}>
          {selectedIndices.size} slice{selectedIndices.size !== 1 ? "s" : ""} selected
        </span>
      </div>

      {/* Thumbnail strip with selection indicators */}
      <div
        style={{
          display: "flex",
          gap: 3,
          overflowX: "auto",
          padding: "4px 0",
          maxHeight: 80,
        }}
      >
        {slices.map((_, i) => (
          <div
            key={i}
            onClick={() => onCurrentChange(i)}
            style={{
              minWidth: 6,
              height: 40,
              background: i === currentIndex ? "#4a90d9" : selectedIndices.has(i) ? "#2d6aa0" : "#333",
              borderRadius: 2,
              cursor: "pointer",
              border: selectedIndices.has(i) ? "1px solid #4a90d9" : "1px solid transparent",
            }}
            title={`Slice ${i + 1}${selectedIndices.has(i) ? " (selected)" : ""}`}
          />
        ))}
      </div>
    </div>
  );
}
