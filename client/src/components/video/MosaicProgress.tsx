import { useEffect, useMemo, useRef } from 'react';
import { getClipFrames, mockClipUrl, sceneForPrompt } from '../../lib/mockClips';
import { mulberry32 } from '../../lib/rng';

/**
 * Generation progress as a pixel mosaic: the frame starts as coarse 32px blocks and
 * resolves block-by-block (32 → 8 → full) as diffusion steps complete.
 * The image is a schematic stand-in for the frame being generated.
 */
export function MosaicProgress({ prompt, seed, progress, className }: { prompt: string; seed: number; progress: number; className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const url = useMemo(() => mockClipUrl({ scene: sceneForPrompt(prompt), seed, variant: 'teacher', method: 'conv_lora' }), [prompt, seed]);
  const order = useMemo(() => {
    const r = mulberry32(seed);
    const cells = Array.from({ length: 64 }, (_, i) => i);
    for (let i = cells.length - 1; i > 0; i--) {
      const j = Math.floor(r() * (i + 1));
      [cells[i], cells[j]] = [cells[j], cells[i]];
    }
    const rank = new Array(64);
    cells.forEach((c, k) => (rank[c] = k));
    return rank as number[];
  }, [seed]);

  useEffect(() => {
    const ctx = ref.current?.getContext('2d');
    if (!ctx) return;
    const src = getClipFrames(url)[7];
    const lv = (block: number) => {
      const c = document.createElement('canvas');
      c.width = c.height = 256 / block;
      c.getContext('2d')!.drawImage(src, 0, 0, c.width, c.height);
      return c;
    };
    const coarse = lv(32);
    const mid = lv(8);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(coarse, 0, 0, 256, 256);
    const p = Math.max(0, Math.min(1, progress));
    for (let cell = 0; cell < 64; cell++) {
      const x = (cell % 8) * 32;
      const y = Math.floor(cell / 8) * 32;
      const rk = order[cell] / 64;
      if (rk < p * 1.6 - 0.6) ctx.drawImage(src, x, y, 32, 32, x, y, 32, 32);
      else if (rk < p * 1.6) ctx.drawImage(mid, x / 8, y / 8, 4, 4, x, y, 32, 32);
    }
  }, [url, progress, order]);

  return <canvas ref={ref} width={256} height={256} aria-hidden className={`block w-full aspect-square pixelated ${className ?? ''}`} />;
}
