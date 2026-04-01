import React, { useCallback, useState } from "react";

interface Props {
  onUpload: (files: File[]) => void;
  isUploading: boolean;
}

// Files to always skip (system / non-medical)
const IGNORED_NAMES = new Set([
  ".ds_store",
  "thumbs.db",
  "desktop.ini",
  "._.ds_store",
  "dicomdir",
]);

const IGNORED_EXTENSIONS = new Set([
  ".txt",
  ".json",
  ".xml",
  ".html",
  ".css",
  ".js",
  ".md",
  ".csv",
  ".log",
  ".py",
  ".sh",
  ".bat",
  ".exe",
  ".dll",
  ".so",
  ".dylib",
]);

function shouldIncludeFile(f: File): boolean {
  const name = f.name.toLowerCase();
  if (IGNORED_NAMES.has(name)) return false;
  const ext = name.includes(".") ? name.slice(name.lastIndexOf(".")) : "";
  if (IGNORED_EXTENSIONS.has(ext)) return false;
  // Accept everything else: .dcm, .DCM, .ima, .nii, .nii.gz, no extension, .zip, etc.
  return true;
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

      // Collect files from drag (supports folder drops via webkitGetAsEntry)
      const entries: FileSystemEntry[] = [];
      for (let i = 0; i < items.length; i++) {
        const entry = items[i].webkitGetAsEntry?.();
        if (entry) entries.push(entry);
      }

      if (entries.length > 0) {
        await collectFilesRecursive(entries, files);
      } else {
        for (let i = 0; i < e.dataTransfer.files.length; i++) {
          files.push(e.dataTransfer.files[i]);
        }
      }

      const filtered = files.filter(shouldIncludeFile);
      if (filtered.length > 0) {
        onUpload(filtered);
      }
    },
    [onUpload]
  );

  const handleFileInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files) {
        const files = Array.from(e.target.files).filter(shouldIncludeFile);
        if (files.length > 0) onUpload(files);
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
        // @ts-expect-error webkitdirectory is non-standard but works in Chrome/Safari/Edge
        webkitdirectory=""
        directory=""
        style={{ display: "none" }}
        onChange={handleFileInput}
      />
      {isUploading ? (
        <p style={{ color: "#888", fontSize: 16 }}>Processing files...</p>
      ) : (
        <>
          <p style={{ color: "#ccc", fontSize: 18, margin: 0 }}>
            Drop DICOM files, folder, or ZIP here
          </p>
          <p style={{ color: "#666", fontSize: 14, marginTop: 8 }}>
            .dcm, .ima, no extension, .zip — or click to browse
          </p>
        </>
      )}
    </div>
  );
}

async function collectFilesRecursive(
  entries: FileSystemEntry[],
  result: File[]
): Promise<void> {
  for (const entry of entries) {
    if (entry.isFile) {
      const file = await new Promise<File>((resolve) =>
        (entry as FileSystemFileEntry).file(resolve)
      );
      result.push(file);
    } else if (entry.isDirectory) {
      // readEntries may not return all entries in one call — loop until empty
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      let batch: FileSystemEntry[];
      do {
        batch = await new Promise<FileSystemEntry[]>((resolve) =>
          reader.readEntries(resolve)
        );
        await collectFilesRecursive(batch, result);
      } while (batch.length > 0);
    }
  }
}
