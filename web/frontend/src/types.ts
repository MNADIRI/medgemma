export interface SliceInfo {
  index: number;
  position: number;
  preview_url: string;
  instance_number: number | null;
  slice_location: string | null;
}

export interface SeriesMetadata {
  patient_id: string | null;
  patient_name: string | null;
  study_description: string | null;
  series_description: string | null;
  modality: string | null;
  slice_thickness: string | null;
  study_date: string | null;
  rows: number | null;
  columns: number | null;
}

export interface UploadResponse {
  session_id: string;
  num_slices: number;
  slices: SliceInfo[];
  metadata: SeriesMetadata;
}

export interface ROI {
  x: number;      // normalized 0-1, left edge
  y: number;      // normalized 0-1, top edge
  width: number;  // normalized 0-1
  height: number; // normalized 0-1
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  response: string;
  usage: { input_tokens: number | null; output_tokens: number | null } | null;
}

// ── Structured analysis types ──────────────────────────────────────────

export interface DiagnosisEntry {
  tier: string;
  label: string;
  supporting: string;
  against: string;
  raw: string;
}

/** ROI quantitative data from the pipeline (always reliable) */
export interface RoiData {
  density: {
    mean: number;
    median: number;
    sd: number;
    min: number;
    max: number;
    deciles: number[];
    density_asymmetry: number;
  };
  peri_lesional: {
    mean: number | null;
    sd: number | null;
    delta_hu_median_parenchyma: number | null;
  };
  morphometry: {
    major_axis_mm: number;
    minor_axis_mm: number;
    aspect_ratio: number;
    area_mm2: number;
    perimeter_mm: number;
    compactness: number;
    solidity: number;
    eroded_area_fraction: number;
  };
  spatial: {
    laterality: string;
    antero_posterior: string;
    patient_xyz_mm: number[] | null;
  };
}

export interface AnalysisResult {
  localisation: string | null;      // model's localization text
  aspect: string | null;            // model's characterization text
  diagnosis_text: string | null;    // model's diagnosis text
  diagnosis_entries: DiagnosisEntry[];
  roi_data: RoiData | null;         // quantitative data from pipeline
  raw_response: string;
  usage: { input_tokens: number | null; output_tokens: number | null } | null;
}
