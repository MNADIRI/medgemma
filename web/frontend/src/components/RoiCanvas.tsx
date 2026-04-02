import React, { useCallback, useEffect, useRef, useState } from "react";
import type { ROI } from "../types";

interface Props {
  currentRoi: ROI | null;
  onRoiChange: (roi: ROI | null) => void;
  disabled: boolean;
}

const MIN_SIZE = 0.05; // minimum 5% of image dimension

export default function RoiCanvas({ currentRoi, onRoiChange, disabled }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [drawing, setDrawing] = useState(false);
  const startRef = useRef<{ x: number; y: number } | null>(null);
  const [dragRect, setDragRect] = useState<{
    x: number;
    y: number;
    w: number;
    h: number;
  } | null>(null);

  // Sync canvas size with container
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        canvas.width = width;
        canvas.height = height;
      }
    });
    observer.observe(container);

    // Initial size
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;

    return () => observer.disconnect();
  }, []);

  // Draw overlay
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw drag rectangle while dragging
    if (dragRect) {
      ctx.strokeStyle = "#4a90d9";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 3]);
      ctx.strokeRect(dragRect.x, dragRect.y, dragRect.w, dragRect.h);
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(74, 144, 217, 0.15)";
      ctx.fillRect(dragRect.x, dragRect.y, dragRect.w, dragRect.h);
      return;
    }

    // Draw saved ROI
    if (currentRoi && !disabled) {
      const x = currentRoi.x * canvas.width;
      const y = currentRoi.y * canvas.height;
      const w = currentRoi.width * canvas.width;
      const h = currentRoi.height * canvas.height;

      // Fill
      ctx.fillStyle = "rgba(74, 144, 217, 0.2)";
      ctx.fillRect(x, y, w, h);

      // Border
      ctx.strokeStyle = "#4a90d9";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      // "ROI" label
      ctx.fillStyle = "#4a90d9";
      ctx.font = "bold 11px monospace";
      ctx.fillText("ROI", x + 4, y + 13);

      // Clear button (top-right corner of ROI)
      const btnSize = 18;
      const bx = x + w - btnSize - 2;
      const by = y + 2;
      ctx.fillStyle = "rgba(192, 57, 43, 0.85)";
      ctx.beginPath();
      ctx.arc(bx + btnSize / 2, by + btnSize / 2, btnSize / 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "bold 13px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("\u00d7", bx + btnSize / 2, by + btnSize / 2 + 1);
      ctx.textAlign = "start";
      ctx.textBaseline = "alphabetic";
    }
  }, [currentRoi, dragRect, disabled]);

  const getCanvasPos = useCallback(
    (e: React.MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return { x: 0, y: 0 };
      const rect = canvas.getBoundingClientRect();
      return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    },
    []
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (disabled) return;

      const canvas = canvasRef.current;
      if (!canvas) return;
      const pos = getCanvasPos(e);

      // Check if clicking the clear button on existing ROI
      if (currentRoi) {
        const rx = currentRoi.x * canvas.width;
        const ry = currentRoi.y * canvas.height;
        const rw = currentRoi.width * canvas.width;
        const rh = currentRoi.height * canvas.height;
        const btnSize = 18;
        const bx = rx + rw - btnSize - 2 + btnSize / 2;
        const by = ry + 2 + btnSize / 2;
        const dist = Math.sqrt((pos.x - bx) ** 2 + (pos.y - by) ** 2);
        if (dist <= btnSize / 2 + 2) {
          onRoiChange(null);
          return;
        }
      }

      startRef.current = pos;
      setDrawing(true);
    },
    [disabled, currentRoi, getCanvasPos, onRoiChange]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!drawing || !startRef.current) return;
      const pos = getCanvasPos(e);
      const start = startRef.current;
      setDragRect({
        x: Math.min(start.x, pos.x),
        y: Math.min(start.y, pos.y),
        w: Math.abs(pos.x - start.x),
        h: Math.abs(pos.y - start.y),
      });
    },
    [drawing, getCanvasPos]
  );

  const handleMouseUp = useCallback(
    (e: React.MouseEvent) => {
      if (!drawing || !startRef.current) return;
      const canvas = canvasRef.current;
      if (!canvas) return;

      const pos = getCanvasPos(e);
      const start = startRef.current;

      const x = Math.min(start.x, pos.x) / canvas.width;
      const y = Math.min(start.y, pos.y) / canvas.height;
      const w = Math.abs(pos.x - start.x) / canvas.width;
      const h = Math.abs(pos.y - start.y) / canvas.height;

      setDrawing(false);
      startRef.current = null;
      setDragRect(null);

      // Enforce minimum size
      if (w >= MIN_SIZE && h >= MIN_SIZE) {
        onRoiChange({ x, y, width: w, height: h });
      }
    },
    [drawing, getCanvasPos, onRoiChange]
  );

  return (
    <div
      ref={containerRef}
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width: "100%",
        height: "100%",
        cursor: disabled ? "default" : "crosshair",
      }}
    >
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => {
          if (drawing) {
            setDrawing(false);
            startRef.current = null;
            setDragRect(null);
          }
        }}
        style={{ display: "block", width: "100%", height: "100%" }}
      />
    </div>
  );
}
