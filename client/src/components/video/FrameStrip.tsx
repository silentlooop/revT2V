import { useEffect, useState } from 'react';
import type { VideoOut } from '../../api/types';
import { FrameImage } from './ClipSurface';
import { Win98Window } from '../ui/Win98Window';

export function FrameStrip({ video, label }: { video: VideoOut; label: string }) {
  const [open, setOpen] = useState<number | null>(null);
  useEffect(() => {
    if (open == null) return;
    const k = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(null);
      if (e.key === 'ArrowRight') setOpen((o) => (o == null ? o : Math.min(video.frames.length - 1, o + 1)));
      if (e.key === 'ArrowLeft') setOpen((o) => (o == null ? o : Math.max(0, o - 1)));
    };
    window.addEventListener('keydown', k);
    return () => window.removeEventListener('keydown', k);
  }, [open, video.frames.length]);

  return (
    <>
      <div className="grid grid-cols-4 gap-1" role="list" aria-label={`${label} frames`}>
        {video.frames.map((src, i) => (
          <button
            key={src}
            type="button"
            role="listitem"
            onClick={() => setOpen(i)}
            className="relative group border-2 border-ink bg-ink hover:border-accent"
            aria-label={`Enlarge ${label} frame ${video.frame_indices[i]}`}
          >
            <FrameImage src={src} alt="" />
            <span className="absolute left-0 top-0 bg-ink text-white font-mono text-[9px] px-1 group-hover:bg-accent group-hover:text-on-accent">f{video.frame_indices[i]}</span>
          </button>
        ))}
      </div>
      {open != null && (
        <div className="fixed inset-0 z-[90] grid place-items-center bg-ink/70 p-4" onClick={() => setOpen(null)}>
          <div onClick={(e) => e.stopPropagation()} className="w-full max-w-lg">
            <Win98Window title={`${label} — frame_${String(video.frame_indices[open]).padStart(2, '0')}.png`} onClose={() => setOpen(null)}>
              <FrameImage src={video.frames[open]} alt={`${label} frame ${video.frame_indices[open]}`} className="border-2 border-ink" />
              <p className="mt-2 text-xs flex justify-between">
                <span>← → to browse · Esc to close</span>
                <span className="font-mono">
                  {video.width}×{video.height}
                </span>
              </p>
            </Win98Window>
          </div>
        </div>
      )}
    </>
  );
}
