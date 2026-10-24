import type { GenerateRequest, HealthStatus, Job, Method, PairResult, Results, VideoOut } from './types';
import { RESULTS } from '../content/results';
import { mock } from './mock';

/**
 * The ONLY file that knows whether we talk to the mock or to FastAPI.
 * Flip VITE_USE_MOCK=false and set VITE_API_BASE_URL (and, if the server
 * sets one, VITE_API_KEY — printed by server/colab_server.ipynb) to go live.
 */
export const USE_MOCK = import.meta.env.VITE_USE_MOCK !== 'false';
export const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000').replace(/\/$/, '');
const API_KEY = import.meta.env.VITE_API_KEY;

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(API_KEY ? { 'X-API-Key': API_KEY } : {}),
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json() as Promise<T>;
}

/** Backend may return media paths relative to the API (e.g. /media/job_x/teacher.mp4). */
const abs = (u: string) => (/^(https?:|mock:|data:|blob:)/.test(u) ? u : `${API_BASE}${u.startsWith('/') ? '' : '/'}${u}`);

function absVideo(v: VideoOut): VideoOut {
  return { ...v, video_url: abs(v.video_url), frames: v.frames.map(abs) };
}

function absPair(p: PairResult): PairResult {
  return {
    ...p,
    teacher: absVideo(p.teacher),
    student: absVideo(p.student),
    teacher_reversed: p.teacher_reversed ? absVideo(p.teacher_reversed) : undefined,
  };
}

export const api = {
  getHealth(): Promise<HealthStatus> {
    return USE_MOCK ? mock.getHealth() : http('/health');
  },
  getMethods(): Promise<Method[]> {
    return USE_MOCK ? mock.getMethods() : http('/methods');
  },
  generate(req: GenerateRequest): Promise<{ job_id: string }> {
    return USE_MOCK ? mock.generate(req) : http('/generate', { method: 'POST', body: JSON.stringify(req) });
  },
  getJob(id: string): Promise<Job> {
    return USE_MOCK ? mock.getJob(id) : http(`/jobs/${encodeURIComponent(id)}`);
  },
  async getResult(id: string): Promise<PairResult> {
    if (USE_MOCK) return mock.getResult(id);
    return absPair(await http<PairResult>(`/jobs/${encodeURIComponent(id)}/result`));
  },
  getResults(): Promise<Results> {
    // Findings/leaderboard numbers are curated, not live — see CONTRACT.md's
    // own recommendation. Served from src/content/results.ts either way;
    // there is no GET /results on the server.
    return USE_MOCK ? mock.getResults() : Promise.resolve(RESULTS);
  },
};
