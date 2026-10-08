import { useEffect, useRef } from 'react';
import type { FlowField } from '../../api/types';
import { getClipFrames, isMockUrl, parseMockUrl } from '../../lib/mockClips';
import { cx } from '../../lib/format';

/** Renders one frame of a clip. Mock clips draw from a canvas cache; real clips seek a paused <video>. */
export function ClipSurface({ url, frame, fps = 8, className, label }: { url: string; frame: number; fps?: number; className?: string; label: string }) {
  if (isMockUrl(url)) return <MockSurface url={url} frame={frame} className={className} label={label} />;
  return <VideoSurface url={url} frame={frame} fps={fps} className={className} label={label} />;
}

function MockSurface({ url, frame, className, label }: { url: string; frame: number; className?: string; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const frames = getClipFrames(url);
    const f = frames[Math.min(frame, frames.length - 1)];
    const ctx = ref.current?.getContext('2d');
    if (ctx && f) ctx.drawImage(f, 0, 0);
  }, [url, frame]);
  return <canvas ref={ref} width={256} height={256} role="img" aria-label={label} className={cx('block w-full h-auto aspect-square bg-ink', className)} />;
}

function VideoSurface({ url, frame, fps, className, label }: { url: string; frame: number; fps: number; className?: string; label: string }) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const v = ref.current;
    if (!v || Number.isNaN(v.duration)) return;
    v.currentTime = Math.min(v.duration - 0.001, (frame + 0.5) / fps);
  }, [frame, fps]);
  return (
    <video
      ref={ref}
      src={url}
      muted
      playsInline
      preload="auto"
      crossOrigin="anonymous"
      aria-label={label}
      className={cx('block w-full h-auto aspect-square bg-ink object-cover', className)}
    />
  );
}

export function FlowOverlay({ flow, frame, scale = 3 }: { flow: FlowField; frame: number; scale?: number }) {
  const fr = flow.frames[Math.min(frame, flow.frames.length - 1)] ?? [];
  const cw = 256 / flow.grid_w;
  const ch = 256 / flow.grid_h;
  return (
    <svg viewBox="0 0 256 256" className="absolute inset-0 w-full h-full pointer-events-none" aria-hidden>
      <defs>
        <marker id="flow-head" viewBox="0 0 6 6" refX="3" refY="3" markerWidth="4" markerHeight="4" orient="auto">
          <path d="M0 0L6 3 0 6z" fill="var(--color-accent)" />
        </marker>
      </defs>
      {fr.map(([dx, dy], k) => {
        const i = k % flow.grid_w;
        const j = Math.floor(k / flow.grid_w);
        const x = (i + 0.5) * cw;
        const y = (j + 0.5) * ch;
        let ex = dx * scale;
        let ey = dy * scale;
        const len = Math.hypot(ex, ey);
        if (len < 1.5) return <circle key={k} cx={x} cy={y} r="1.2" fill="var(--color-accent)" opacity=".55" />;
        const max = cw * 0.9;
        if (len > max) {
          ex *= max / len;
          ey *= max / len;
        }
        return (
          <g key={k}>
            <line x1={x} y1={y} x2={x + ex} y2={y + ey} stroke="white" strokeWidth="3.4" strokeLinecap="round" />
            <line x1={x} y1={y} x2={x + ex} y2={y + ey} stroke="var(--color-accent)" strokeWidth="1.8" markerEnd="url(#flow-head)" />
          </g>
        );
      })}
    </svg>
  );
}

/** Single still frame (for strips/lightbox). */
export function FrameImage({ src, alt, className }: { src: string; alt: string; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const mock = isMockUrl(src);
  useEffect(() => {
    if (!mock) return;
    const spec = parseMockUrl(src);
    const clipUrl = src.replace(/&frame=\d+$/, '');
    const frames = getClipFrames(clipUrl);
    const f = frames[Math.min(spec.frame ?? 0, frames.length - 1)];
    const ctx = ref.current?.getContext('2d');
    if (ctx && f) ctx.drawImage(f, 0, 0);
  }, [src, mock]);
  if (mock) return <canvas ref={ref} width={256} height={256} role="img" aria-label={alt} className={cx('block w-full h-auto aspect-square', className)} />;
  return <img src={src} alt={alt} loading="lazy" className={cx('block w-full h-auto aspect-square object-cover', className)} />;
}

/** Downloads a real video, or for mock clips a PNG contact sheet of all frames. */
export function downloadClip(url: string, name: string) {
  if (!isMockUrl(url)) {
    const a = document.createElement('a');
    a.href = url;
    a.download = `${name}.mp4`;
    a.target = '_blank';
    a.rel = 'noopener';
    a.click();
    return;
  }
  const frames = getClipFrames(url);
  const c = document.createElement('canvas');
  const cols = 4;
  c.width = 256 * cols;
  c.height = 256 * Math.ceil(frames.length / cols);
  const ctx = c.getContext('2d')!;
  frames.forEach((f, i) => ctx.drawImage(f, (i % cols) * 256, Math.floor(i / cols) * 256));
  const a = document.createElement('a');
  a.href = c.toDataURL('image/png');
  a.download = `${name}_frames.png`;
  a.click();
}
