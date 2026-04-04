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

export interface AnalysisReport {
  localisation: {
    text: string;
    organ: string;
    segment: string | null;
    laterality: string;
    position: string;
  };
  aspect: {
    text: string;
    density_class: string;
    delta_hu: number;
    homogeneity: string;
    margins: string;
    shape: string;
    aspect_ratio: number;
    mass_effect: boolean;
  };
  taille: {
    text: string;
    long_axis_mm: number;
    short_axis_mm: number;
    area_mm2: number;
  };
}

export interface DiagnosisEntry {
  rank: number;
  tier: "likely" | "possible" | "unlikely_but_to_exclude";
  label: string;
  supporting_features: string[];
  against_features: string[];
  confidence_rationale: string;
}

export interface AnalysisDiagnosis {
  diagnostics: DiagnosisEntry[];
}

export interface AnalysisResult {
  chain_of_thought: Record<string, unknown> | null;
  report: AnalysisReport | null;
  diagnosis: AnalysisDiagnosis | null;
  raw_response: string;
  usage: { input_tokens: number | null; output_tokens: number | null } | null;
}
