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

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatResponse {
  response: string;
  usage: { input_tokens: number | null; output_tokens: number | null } | null;
}
