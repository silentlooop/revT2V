export type MethodId = 'attn_injection' | 'conv_oracle' | 'conv_lora' | (string & {});

export type MethodKind = 'main' | 'trained' | 'training_free';

export interface Method {
  id: MethodId;
  label: string;
  kind: MethodKind;
  description: string;
  trained: boolean;
  available: boolean;
  checkpoint?: string;
}

export interface GenerateRequest {
  prompt: string;
  student_method: MethodId;
  seed: number;
  steps: number;
  cfg: number;
  num_frames: number;
  negative_prompt: string;
}

export type JobStatus = 'queued' | 'running' | 'done' | 'error';

export interface Job {
  id: string;
  status: JobStatus;
  queue_pos: number;
  current_model: 'teacher' | 'student';
  step: number;
  total_steps: number;
  elapsed_s: number;
  created_at: string;
  request: GenerateRequest;
  error?: string;
}

/** Sparse optical-flow field, sampled on a grid_w × grid_h grid for every frame transition. */
export interface FlowField {
  grid_w: number;
  grid_h: number;
  /** frames[t][j*grid_w+i] = [dx, dy] in pixels at 256×256, between frame t and t+1 */
  frames: [number, number][][];
}

export interface VideoOut {
  video_url: string;
  frames: string[];
  frame_indices: number[];
  flow: FlowField;
  fps: number;
  width: number;
  height: number;
  num_frames: number;
}

export interface MetricNotes {
  fvd?: string;
  clip?: string;
  dynamic_degree?: string;
  optical_flow?: string;
}

export interface PairMetrics {
  fvd: number;
  clip: { teacher: number; student: number };
  dynamic_degree: { teacher: number; student: number };
  optical_flow: {
    teacher_mag: number;
    student_mag: number;
    direction_cosine: number;
    frozen: boolean;
  };
  notes: MetricNotes;
}

export interface PairResult {
  job_id: string;
  request: GenerateRequest;
  teacher: VideoOut;
  student: VideoOut;
  teacher_reversed?: VideoOut;
  metrics: PairMetrics;
  time_taken_s: number;
}

export interface HealthStatus {
  online: boolean;
  gpu: string | null;
  model_loaded: boolean;
  queue_length: number;
  version: string;
}

/* ---------- set-level results (Section 3) ---------- */

export interface MetricColumn {
  key: 'fvd' | 'clip' | 'dynamic_degree' | 'flow_cosine';
  label: string;
  /** 'one' = best when closest to 1.0 */
  better: 'higher' | 'lower' | 'one';
  unit?: string;
  hint: string;
}

export interface LeaderboardRow {
  method: MethodId;
  fvd: number | null;
  clip: number | null;
  dynamic_degree: number | null;
  flow_cosine: number | null;
  todo?: boolean;
  note?: string;
}

export interface LossCurve {
  method: MethodId;
  label: string;
  steps: number[];
  total: number[];
  components?: { name: string; values: number[] }[];
  todo?: boolean;
}

export interface QualitativeEntry {
  prompt: string;
  seed: number;
  /** method -> video url (mock:// or http) */
  videos: Record<string, string>;
}

export interface Finding {
  id: string;
  method: MethodId;
  title: string;
  body: string;
  verdict: 'negative' | 'positive' | 'mixed' | 'todo';
  todo?: boolean;
}

export interface Results {
  test_set: { name: string; num_prompts: number; seeds_per_prompt: number; todo?: boolean };
  columns: MetricColumn[];
  leaderboard: LeaderboardRow[];
  loss_curves: LossCurve[];
  qualitative: QualitativeEntry[];
  findings: Finding[];
}
