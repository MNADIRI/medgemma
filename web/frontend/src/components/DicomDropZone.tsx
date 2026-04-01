import React, { useCallback, useState } from "react";

interface Props {
  onUpload: (files: File[]) => void;
  isUploading: boolean;
}

export default function DicomDropZone({ onUpload, isUploading }: Props) {
  const [isDragging, setIsDragging] = useState(false);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const items = e.dataTransfer.items;
      const files: File[] = [];

      // Collect files from drag (supports folder drops)
      const entries: FileSystemEntry[] = [];
      for (let i = 0; i < items.length; i++) {
        const entry = items[i].webkitGetAsEntry?.();
        if (entry) entries.push(entry);
      }

      if (entries.length > 0) {
        await collectFiles(entries, files);
      } else {
        // Fallback: just use the file list
        for (let i = 0; i < e.dataTransfer.files.length; i++) {
          files.push(e.dataTransfer.files[i]);
        }
      }

      // Filter to .dcm files (or files without extension which are often DICOM)
      const dicomFiles = files.filter(
        (f) => f.name.endsWith(".dcm") || f.name.endsWith(".DCM") || !f.name.includes(".")
      );

      if (dicomFiles.length > 0) {
        onUpload(dicomFiles);
      }
    },
    [onUpload]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files) {
        const files = Array.from(e.target.files);
        onUpload(files);
      }
    },
    [onUpload]
  );

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      style={{
        border: `2px dashed ${isDragging ? "#4a90d9" : "#555"}`,
        borderRadius: 12,
        padding: 40,
        textAlign: "center",
        background: isDragging ? "#1a2a3a" : "#111",
        transition: "all 0.2s",
        cursor: "pointer",
      }}
      onClick={() => document.getElementById("file-input")?.click()}
    >
      <input
        id="file-input"
        type="file"
        multiple
        accept=".dcm,.DCM"
        style={{ display: "none" }}
        onChange={handleFileInput}
      />
      {isUploading ? (
        <p style={{ color: "#888", fontSize: 16 }}>Processing DICOM files...</p>
      ) : (
        <>
          <p style={{ color: "#ccc", fontSize: 18, margin: 0 }}>
            Drop DICOM CT files or folder here
          </p>
          <p style={{ color: "#666", fontSize: 14, marginTop: 8 }}>
            or click to browse
          </p>
        </>
      )}
    </div>
  );
}

async function collectFiles(entries: FileSystemEntry[], result: File[]): Promise<void> {
  for (const entry of entries) {
    if (entry.isFile) {
      const file = await new Promise<File>((resolve) =>
        (entry as FileSystemFileEntry).file(resolve)
      );
      result.push(file);
    } else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      const children = await new Promise<FileSystemEntry[]>((resolve) =>
        reader.readEntries(resolve)
      );
      await collectFiles(children, result);
    }
  }
}
