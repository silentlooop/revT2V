import type { FlowField } from '../api/types';
import { clamp, hashString, mulberry32 } from './rng';

/**
 * Procedural stand-ins for generated videos. A mock URL looks like
 *   mock://clip?scene=smoke&seed=42&variant=student&method=conv_lora
 * Frames are rendered once into canvases and cached, so playback is just drawImage.
 */

export type SceneId = 'smoke' | 'ink' | 'juice' | 'ball' | 'corridor' | 'abstract';
export type Variant = 'teacher' | 'student' | 'teacher_reversed';

export interface ClipSpec {
  scene: SceneId;
  seed: number;
  variant: Variant;
  method: string;
  frames: number;
  size: number;
}

export const MOCK_SIZE = 256;

export function sceneForPrompt(prompt: string): SceneId {
  const p = prompt.toLowerCase();
  if (/smoke|incense|steam|fog/.test(p)) return 'smoke';
  if (/ink|dye|water/.test(p)) return 'ink';
  if (/pour|juice|milk|glass|coffee/.test(p)) return 'juice';
  if (/ball|fall|drop|bounce/.test(p)) return 'ball';
  if (/walk|corridor|man|woman|person|run/.test(p)) return 'corridor';
  return 'abstract';
}

export function mockClipUrl(spec: Omit<ClipSpec, 'frames' | 'size'> & { frames?: number }): string {
  const q = new URLSearchParams({
    scene: spec.scene,
    seed: String(spec.seed),
    variant: spec.variant,
    method: spec.method,
    frames: String(spec.frames ?? 16),
  });
  return `mock://clip?${q.toString()}`;
}

export function isMockUrl(url: string) {
  return url.startsWith('mock://');
}

export function parseMockUrl(url: string): ClipSpec & { frame?: number } {
  const q = new URLSearchParams(url.split('?')[1] ?? '');
  const frame = q.get('frame');
  return {
    scene: (q.get('scene') as SceneId) ?? 'abstract',
    seed: Number(q.get('seed') ?? 0),
    variant: (q.get('variant') as Variant) ?? 'teacher',
    method: q.get('method') ?? 'conv_lora',
    frames: Number(q.get('frames') ?? 16),
    size: MOCK_SIZE,
    frame: frame == null ? undefined : Number(frame),
  };
}

export function frameUrl(clipUrl: string, frame: number) {
  return `${clipUrl}&frame=${frame}`;
}

/* ---------- time mapping: how each method's output relates to the teacher's forward time ---------- */

/** Returns forward-time position u∈[0,1] of the underlying physical process at normalised clip time t. */
function timeMap(spec: ClipSpec, t: number): number {
  const { variant, method, seed } = spec;
  if (variant === 'teacher') return t;
  if (variant === 'teacher_reversed') return 1 - t;
  const r = mulberry32(seed ^ hashString(method));
  const wob = r() * 6.28;
  switch (method) {
    case 'conv_oracle':
      return 1 - t;
    case 'conv_lora':
      return clamp(1 - t + 0.025 * Math.sin(t * 9 + wob), 0, 1);
    case 'attn_injection':
      // damped, partly reversed motion with a direction wobble
      return clamp(0.75 - 0.4 * t + 0.06 * Math.sin(t * Math.PI * 2 + wob), 0, 1);
    default:
      return 1 - t;
  }
}

/* ---------- scenes ---------- */

type Ctx = CanvasRenderingContext2D;

interface Scene {
  draw(ctx: Ctx, u: number, rnd: () => number, s: number): void;
  /** velocity (pixels per unit u) at (x,y) */
  vel(x: number, y: number, u: number, s: number): [number, number];
}

const smoke: Scene = {
  draw(ctx, u, rnd, s) {
    ctx.fillStyle = '#0d0d0d';
    ctx.fillRect(0, 0, s, s);
    const cx = s * (0.45 + rnd() * 0.1);
    ctx.fillStyle = '#5a4a3a';
    ctx.fillRect(cx - 2, s * 0.72, 4, s * 0.3);
    ctx.fillStyle = '#ff7a3a';
    ctx.beginPath();
    ctx.arc(cx, s * 0.72, 3, 0, Math.PI * 2);
    ctx.fill();
    const n = 140;
    for (let i = 0; i < n; i++) {
      const birth = -1 + (i / n) * 2;
      const phase = rnd() * 6.28;
      const drift = (rnd() - 0.5) * 40;
      const age = u * 1.1 - birth;
      if (age <= 0 || age > 1.25) continue;
      const x = cx + Math.sin(age * 4 + phase) * age * 28 + drift * age * age;
      const y = s * 0.71 - age * s * 0.62;
      const r = 3 + age * 22;
      const a = Math.max(0, 0.22 * (1 - age / 1.25));
      ctx.fillStyle = `rgba(235,235,235,${a})`;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
  },
  vel(x, y, u, s) {
    const top = s * 0.71 - Math.min(1.25, u * 1.1 + 1) * s * 0.62;
    if (y > s * 0.72 || y < top) return [0, 0];
    const spread = 10 + (s * 0.72 - y) * 0.35;
    if (Math.abs(x - s / 2) > spread) return [0, 0];
    return [Math.sin(y * 0.05) * 18, -s * 0.56];
  },
};

const ink: Scene = {
  draw(ctx, u, rnd, s) {
    ctx.fillStyle = '#e9e9e6';
    ctx.fillRect(0, 0, s, s);
    ctx.strokeStyle = '#9a9a9a';
    ctx.lineWidth = 3;
    ctx.strokeRect(s * 0.18, s * 0.12, s * 0.64, s * 0.8);
    ctx.fillStyle = '#d4dbe0';
    ctx.fillRect(s * 0.19, s * 0.3, s * 0.62, s * 0.61);
    const drop = clamp(u / 0.18, 0, 1);
    if (u < 0.2) {
      ctx.fillStyle = '#2a2a2a';
      ctx.beginPath();
      ctx.arc(s / 2, s * 0.05 + drop * s * 0.27, 6, 0, Math.PI * 2);
      ctx.fill();
    }
    const spread = clamp((u - 0.15) / 0.85, 0, 1);
    const blobs = 26;
    for (let i = 0; i < blobs; i++) {
      const ang = rnd() * Math.PI * 2;
      const dist = rnd() * 80 * spread;
      const sink = rnd() * 120 * spread;
      const x = s / 2 + Math.cos(ang) * dist;
      const y = s * 0.33 + sink + Math.sin(ang) * dist * 0.4;
      const r = 4 + spread * (14 + rnd() * 26);
      const a = spread > 0 ? 0.55 * (1 - spread * 0.55) : 0;
      ctx.fillStyle = `rgba(30,30,30,${a})`;
      ctx.beginPath();
      ctx.ellipse(x, y, r, r * 0.8, ang, 0, Math.PI * 2);
      ctx.fill();
    }
  },
  vel(x, y, u, s) {
    if (u < 0.15) return Math.abs(x - s / 2) < 10 && y < s * 0.35 ? [0, s * 1.5] : [0, 0];
    const dx = x - s / 2;
    const dy = y - s * 0.4;
    const d = Math.hypot(dx, dy);
    const R = 30 + 110 * u;
    if (d > R || y < s * 0.3) return [0, 0];
    return [(dx / (d + 1)) * 70, (dy / (d + 1)) * 70 + 60];
  },
};

const juice: Scene = {
  draw(ctx, u, rnd, s) {
    ctx.fillStyle = '#efeee9';
    ctx.fillRect(0, 0, s, s);
    const gx = s * 0.3;
    const gw = s * 0.4;
    const gTop = s * 0.35;
    const gBot = s * 0.9;
    const level = gBot - (gBot - gTop) * 0.92 * u;
    ctx.fillStyle = '#7a7a7a';
    ctx.fillRect(gx, level, gw, gBot - level);
    for (let i = 0; i < 18; i++) {
      ctx.fillStyle = 'rgba(255,255,255,0.35)';
      ctx.beginPath();
      ctx.arc(gx + rnd() * gw, level + rnd() * (gBot - level), 1 + rnd() * 2, 0, Math.PI * 2);
      ctx.fill();
    }
    if (u < 0.97) {
      ctx.fillStyle = '#5c5c5c';
      const w = 7 + Math.sin(u * 40) * 1.5;
      ctx.fillRect(s / 2 - w / 2, 0, w, level);
    }
    ctx.strokeStyle = '#222';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(gx - 4, gTop - 10);
    ctx.lineTo(gx, gBot);
    ctx.lineTo(gx + gw, gBot);
    ctx.lineTo(gx + gw + 4, gTop - 10);
    ctx.stroke();
  },
  vel(x, y, u, s) {
    const level = s * 0.9 - s * 0.55 * 0.92 * u;
    if (Math.abs(x - s / 2) < 8 && y < level) return [0, s * 2.2];
    if (x > s * 0.3 && x < s * 0.7 && Math.abs(y - level) < 14) return [0, -s * 0.5];
    return [0, 0];
  },
};

function ballY(u: number, s: number) {
  // two bounces with decay
  const floor = s * 0.78;
  const top = s * 0.1;
  const segs = [
    { t0: 0, t1: 0.45, h: floor - top, start: true },
    { t0: 0.45, t1: 0.75, h: (floor - top) * 0.4 },
    { t0: 0.75, t1: 0.92, h: (floor - top) * 0.14 },
  ];
  for (const g of segs) {
    if (u >= g.t0 && u < g.t1) {
      const p = (u - g.t0) / (g.t1 - g.t0);
      if (g.start) return top + (floor - top) * p * p;
      return floor - g.h * 4 * p * (1 - p);
    }
  }
  return floor;
}

const ball: Scene = {
  draw(ctx, u, _rnd, s) {
    ctx.fillStyle = '#e8e6e0';
    ctx.fillRect(0, 0, s, s);
    ctx.fillStyle = '#b9b3a6';
    ctx.fillRect(0, s * 0.82, s, s * 0.18);
    ctx.strokeStyle = 'rgba(0,0,0,0.18)';
    for (let i = 0; i < 9; i++) {
      ctx.beginPath();
      ctx.moveTo(i * 34 - 20, s);
      ctx.lineTo(i * 34 + 10, s * 0.82);
      ctx.stroke();
    }
    const y = ballY(u, s);
    const r = 18;
    const shadowW = 10 + (y / (s * 0.78)) * 20;
    ctx.fillStyle = 'rgba(0,0,0,0.25)';
    ctx.beginPath();
    ctx.ellipse(s / 2, s * 0.84, shadowW, 4, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#3a3a3a';
    ctx.beginPath();
    ctx.arc(s / 2, y + r * 0.1, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.beginPath();
    ctx.arc(s / 2 - 6, y - 5, 5, 0, Math.PI * 2);
    ctx.fill();
  },
  vel(x, y, u, s) {
    const by = ballY(u, s);
    if (Math.hypot(x - s / 2, y - by) > 26) return [0, 0];
    const dy = (ballY(Math.min(1, u + 0.01), s) - by) / 0.01;
    return [0, dy];
  },
};

const corridor: Scene = {
  draw(ctx, u, _rnd, s) {
    ctx.fillStyle = '#d9d9d6';
    ctx.fillRect(0, 0, s, s);
    const vx = s / 2;
    const vy = s * 0.42;
    ctx.strokeStyle = '#7d7d7d';
    ctx.lineWidth = 1.5;
    for (const [x, y] of [
      [0, 0],
      [s, 0],
      [0, s],
      [s, s],
    ]) {
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(vx, vy);
      ctx.stroke();
    }
    ctx.fillStyle = '#bdbdb8';
    ctx.fillRect(vx - 14, vy - 18, 28, 36);
    for (let i = 1; i < 6; i++) {
      const k = i / 6;
      ctx.strokeStyle = 'rgba(0,0,0,0.12)';
      ctx.strokeRect(vx - k * vx, vy - k * vy, k * s, k * (s - vy) + k * vy);
    }
    const scale = 1 - 0.78 * u;
    const fy = vy + (s * 0.95 - vy) * scale;
    const h = 130 * scale;
    const leg = Math.sin(u * 40) * 0.35;
    ctx.strokeStyle = '#1b1b1b';
    ctx.fillStyle = '#1b1b1b';
    ctx.lineWidth = Math.max(1.5, 9 * scale);
    ctx.lineCap = 'round';
    const hip = fy - h * 0.45;
    ctx.beginPath();
    ctx.moveTo(vx, hip);
    ctx.lineTo(vx + Math.sin(leg) * h * 0.4, fy);
    ctx.moveTo(vx, hip);
    ctx.lineTo(vx - Math.sin(leg) * h * 0.4, fy);
    ctx.moveTo(vx, hip);
    ctx.lineTo(vx, fy - h * 0.85);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(vx, fy - h * 0.93, h * 0.09, 0, Math.PI * 2);
    ctx.fill();
  },
  vel(x, y, u, s) {
    const scale = 1 - 0.78 * u;
    const vy = s * 0.42;
    const fy = vy + (s * 0.95 - vy) * scale;
    const h = 130 * scale;
    if (Math.abs(x - s / 2) > h * 0.35 || y > fy || y < fy - h) return [0, 0];
    return [(s / 2 - x) * 0.8, (vy - y) * 0.8];
  },
};

const abstract: Scene = {
  draw(ctx, u, rnd, s) {
    ctx.fillStyle = '#e6e6e2';
    ctx.fillRect(0, 0, s, s);
    const n = 9;
    for (let i = 0; i < n; i++) {
      const ox = rnd() * s;
      const oy = rnd() * s;
      const vx = (rnd() - 0.5) * s * 0.8;
      const vy = (rnd() - 0.5) * s * 0.8;
      const r = 10 + rnd() * 26;
      const g = Math.floor(30 + rnd() * 140);
      ctx.fillStyle = `rgb(${g},${g},${g})`;
      ctx.save();
      ctx.translate(ox + vx * (u - 0.5), oy + vy * (u - 0.5));
      ctx.rotate(u * 3 + i);
      if (i % 2) ctx.fillRect(-r, -r, r * 2, r * 2);
      else {
        ctx.beginPath();
        ctx.arc(0, 0, r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }
  },
  vel(x, y, _u, s) {
    return [Math.sin(y / s * 6) * 60, Math.cos(x / s * 6) * 60];
  },
};

const SCENES: Record<SceneId, Scene> = { smoke, ink, juice, ball, corridor, abstract };

/* ---------- rendering + cache ---------- */

const cache = new Map<string, HTMLCanvasElement[]>();

function clipKey(spec: ClipSpec) {
  return `${spec.scene}|${spec.seed}|${spec.variant}|${spec.method}|${spec.frames}`;
}

function photocopy(ctx: Ctx, s: number, seed: number, block: number) {
  const img = ctx.getImageData(0, 0, s, s);
  const d = img.data;
  const r = mulberry32(seed);
  for (let i = 0; i < d.length; i += 4) {
    const l = 0.3 * d[i] + 0.59 * d[i + 1] + 0.11 * d[i + 2];
    const c = clamp((l - 128) * 1.18 + 128 + (r() - 0.5) * 34, 0, 255);
    d[i] = d[i + 1] = d[i + 2] = c;
  }
  ctx.putImageData(img, 0, 0);
  if (block > 1) {
    const tmp = document.createElement('canvas');
    tmp.width = tmp.height = Math.ceil(s / block);
    const t = tmp.getContext('2d')!;
    t.drawImage(ctx.canvas, 0, 0, tmp.width, tmp.height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, s, s);
    ctx.imageSmoothingEnabled = true;
  }
}

export function getClipFrames(url: string): HTMLCanvasElement[] {
  const spec = parseMockUrl(url);
  const key = clipKey(spec);
  const hit = cache.get(key);
  if (hit) return hit;
  const scene = SCENES[spec.scene] ?? abstract;
  const frames: HTMLCanvasElement[] = [];
  for (let f = 0; f < spec.frames; f++) {
    const c = document.createElement('canvas');
    c.width = c.height = spec.size;
    const ctx = c.getContext('2d', { willReadFrequently: true })!;
    const t = spec.frames === 1 ? 0 : f / (spec.frames - 1);
    const u = timeMap(spec, t);
    scene.draw(ctx, u, mulberry32(spec.seed * 7919 + 1), spec.size);
    photocopy(ctx, spec.size, spec.seed * 31 + f, 0);
    frames.push(c);
  }
  cache.set(key, frames);
  return frames;
}

/* ---------- flow ---------- */

export function mockFlow(url: string, grid = 8): FlowField {
  const spec = parseMockUrl(url);
  const scene = SCENES[spec.scene] ?? abstract;
  const s = spec.size;
  const out: [number, number][][] = [];
  for (let f = 0; f < spec.frames - 1; f++) {
    const t0 = f / (spec.frames - 1);
    const t1 = (f + 1) / (spec.frames - 1);
    const u0 = timeMap(spec, t0);
    const du = timeMap(spec, t1) - u0;
    const field: [number, number][] = [];
    for (let j = 0; j < grid; j++) {
      for (let i = 0; i < grid; i++) {
        const x = ((i + 0.5) / grid) * s;
        const y = ((j + 0.5) / grid) * s;
        const [vx, vy] = scene.vel(x, y, u0, s);
        field.push([vx * du, vy * du]);
      }
    }
    out.push(field);
  }
  return { grid_w: grid, grid_h: grid, frames: out };
}

export function flowStats(flow: FlowField) {
  let sum = 0;
  let n = 0;
  for (const fr of flow.frames)
    for (const [dx, dy] of fr) {
      sum += Math.hypot(dx, dy);
      n++;
    }
  return sum / Math.max(1, n);
}

/** cosine between flow a and the time-reversal of flow b (b reversed: frame order flipped, vectors negated) */
export function reversedCosine(a: FlowField, b: FlowField) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  const T = Math.min(a.frames.length, b.frames.length);
  for (let t = 0; t < T; t++) {
    const fa = a.frames[t];
    const fb = b.frames[T - 1 - t];
    for (let k = 0; k < fa.length; k++) {
      const [ax, ay] = fa[k];
      const bx = -fb[k][0];
      const by = -fb[k][1];
      dot += ax * bx + ay * by;
      na += ax * ax + ay * ay;
      nb += bx * bx + by * by;
    }
  }
  if (na < 1e-9 || nb < 1e-9) return 0;
  return dot / Math.sqrt(na * nb);
}
