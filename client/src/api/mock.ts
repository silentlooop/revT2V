import type { GenerateRequest, HealthStatus, Job, Method, PairMetrics, PairResult, Results, VideoOut } from './types';
import { RESULTS } from '../content/results';
import { flowStats, frameUrl, mockClipUrl, mockFlow, reversedCosine, sceneForPrompt, type Variant } from '../lib/mockClips';
import { hashString, mulberry32 } from '../lib/rng';

const JOBS_KEY = 'revt2v.mock.jobs';
const OFFLINE_KEY = 'revt2v.mock.offline';
const STEP_SEC = 0.3;
const QUEUE_SEC = 2.2;

export const delay = (ms: number) => new Promise((r) => setTimeout(r, ms));
const jitter = () => delay(120 + Math.random() * 260);

export const MOCK_METHODS: Method[] = [
  {
    id: 'attn_injection',
    label: 'attn_injection',
    kind: 'trained',
    description: 'Inject the base U-Net\'s own attention, rotated 180°, + LoRA',
    trained: true,
    available: true,
    checkpoint: 'silentlooop/revt2v-ckpt/attn_injection',
  },
  {
    id: 'conv_oracle',
    label: 'conv_oracle',
    kind: 'training_free',
    description: 'Temporal conv kernels flipped in time',
    trained: false,
    available: true,
  },
  {
    id: 'conv_lora',
    label: 'conv_lora',
    kind: 'main',
    description: 'LoRA on temporal convs + attention',
    trained: true,
    available: true,
    checkpoint: 'silentlooop/revt2v-ckpt/conv_lora',
  },
];

interface StoredJob {
  id: string;
  createdAt: number;
  queueAhead: number;
  request: GenerateRequest;
}

function loadJobs(): Record<string, StoredJob> {
  try {
    return JSON.parse(localStorage.getItem(JOBS_KEY) ?? '{}');
  } catch {
    return {};
  }
}

function saveJobs(jobs: Record<string, StoredJob>) {
  try {
    localStorage.setItem(JOBS_KEY, JSON.stringify(jobs));
  } catch {
    /* storage unavailable: jobs just won't survive a refresh */
  }
}

export function isMockOffline() {
  try {
    return localStorage.getItem(OFFLINE_KEY) === '1' || new URLSearchParams(location.search).has('offline');
  } catch {
    return false;
  }
}

export function setMockOffline(v: boolean) {
  try {
    localStorage.setItem(OFFLINE_KEY, v ? '1' : '0');
  } catch {
    /* ignore */
  }
}

function studentSteps(req: GenerateRequest) {
  return req.steps;
}

function jobState(j: StoredJob): Job {
  const elapsed = (Date.now() - j.createdAt) / 1000;
  const q = j.queueAhead * QUEUE_SEC;
  const tSteps = j.request.steps;
  const sSteps = studentSteps(j.request);
  const base = { id: j.id, created_at: new Date(j.createdAt).toISOString(), request: j.request, elapsed_s: elapsed };
  if (elapsed < q) {
    return {
      ...base,
      status: 'queued',
      queue_pos: Math.ceil((q - elapsed) / QUEUE_SEC),
      current_model: 'teacher',
      step: 0,
      total_steps: tSteps,
    };
  }
  const run = elapsed - q;
  const tDur = tSteps * STEP_SEC;
  if (run < tDur) {
    return { ...base, status: 'running', queue_pos: 0, current_model: 'teacher', step: Math.floor(run / STEP_SEC), total_steps: tSteps };
  }
  const sRun = run - tDur;
  if (sRun < sSteps * STEP_SEC) {
    return { ...base, status: 'running', queue_pos: 0, current_model: 'student', step: Math.floor(sRun / STEP_SEC), total_steps: sSteps };
  }
  return {
    ...base,
    status: 'done',
    queue_pos: 0,
    current_model: 'student',
    step: Math.max(sSteps, 1),
    total_steps: Math.max(sSteps, 1),
    elapsed_s: q + tDur + sSteps * STEP_SEC,
  };
}

function videoOut(url: string, frames: number): VideoOut {
  const idx = [0, 5, 10, 15].filter((i) => i < frames);
  return {
    video_url: url,
    frames: idx.map((i) => frameUrl(url, i)),
    frame_indices: idx,
    flow: mockFlow(url),
    fps: 8,
    width: 256,
    height: 256,
    num_frames: frames,
  };
}

const METHOD_PROFILE: Record<string, { fvd: [number, number]; clipDelta: number }> = {
  attn_injection: { fvd: [380, 620], clipDelta: -1.2 },
  conv_oracle: { fvd: [8, 30], clipDelta: 0 },
  conv_lora: { fvd: [180, 320], clipDelta: -0.4 },
};

export function buildPair(jobId: string, req: GenerateRequest, timeTaken: number): PairResult {
  const scene = sceneForPrompt(req.prompt);
  const mk = (variant: Variant) =>
    mockClipUrl({ scene, seed: req.seed, variant, method: req.student_method, frames: req.num_frames });
  const teacher = videoOut(mk('teacher'), req.num_frames);
  const student = videoOut(mk('student'), req.num_frames);
  const teacher_reversed = videoOut(mk('teacher_reversed'), req.num_frames);

  const r = mulberry32(req.seed ^ hashString(req.prompt + req.student_method));
  const prof = METHOD_PROFILE[req.student_method] ?? METHOD_PROFILE.conv_lora;
  const tMag = flowStats(teacher.flow);
  const sMag = flowStats(student.flow);
  const cos = reversedCosine(student.flow, teacher.flow);
  const clipT = 30.5 + r() * 3;
  const dynT = Math.min(1, 0.55 + tMag / 12);
  const metrics: PairMetrics = {
    fvd: Math.round(prof.fvd[0] + r() * (prof.fvd[1] - prof.fvd[0])),
    clip: { teacher: +clipT.toFixed(2), student: +(clipT + prof.clipDelta + (r() - 0.5) * 0.6).toFixed(2) },
    dynamic_degree: { teacher: +dynT.toFixed(2), student: +Math.min(1, dynT * (sMag / Math.max(tMag, 1e-6))).toFixed(2) },
    optical_flow: {
      teacher_mag: +tMag.toFixed(2),
      student_mag: +sMag.toFixed(2),
      direction_cosine: +cos.toFixed(3),
      frozen: sMag < 0.3 * tMag,
    },
    notes: {
      fvd: 'Single-pair estimate; set-level FVD is in Findings.',
    },
  };
  return { job_id: jobId, request: req, teacher, student, teacher_reversed, metrics, time_taken_s: +timeTaken.toFixed(1) };
}

export const mock = {
  async getHealth(): Promise<HealthStatus> {
    await jitter();
    if (isMockOffline()) throw new Error('backend unreachable (mock offline mode)');
    const jobs = Object.values(loadJobs()).map(jobState);
    return {
      online: true,
      gpu: 'Tesla T4 (mock)',
      model_loaded: true,
      queue_length: jobs.filter((j) => j.status !== 'done').length,
      version: 'mock-0.1',
    };
  },
  async getMethods(): Promise<Method[]> {
    await jitter();
    return MOCK_METHODS;
  },
  async generate(req: GenerateRequest): Promise<{ job_id: string }> {
    await jitter();
    if (isMockOffline()) throw new Error('backend unreachable (mock offline mode)');
    const jobs = loadJobs();
    const active = Object.values(jobs).filter((j) => jobState(j).status !== 'done').length;
    const id = `job_${Date.now().toString(36)}${Math.floor(Math.random() * 1e4).toString(36)}`;
    jobs[id] = { id, createdAt: Date.now(), queueAhead: active + (Math.random() < 0.4 ? 1 : 0), request: req };
    saveJobs(jobs);
    return { job_id: id };
  },
  async getJob(id: string): Promise<Job> {
    await delay(60);
    const j = loadJobs()[id];
    if (!j) throw new Error(`unknown job ${id}`);
    return jobState(j);
  },
  async getResult(id: string): Promise<PairResult> {
    await jitter();
    const j = loadJobs()[id];
    if (!j) throw new Error(`unknown job ${id}`);
    const st = jobState(j);
    if (st.status !== 'done') throw new Error('job not finished');
    return buildPair(id, j.request, st.elapsed_s);
  },
  async getResults(): Promise<Results> {
    await jitter();
    return RESULTS;
  },
};
