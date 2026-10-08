import { motion, type Transition } from 'framer-motion';
import type { ReactNode } from 'react';

export const C = {
  ink: 'var(--color-ink)',
  accent: 'var(--color-accent)',
  soft: 'var(--color-accent-soft)',
  wash: 'var(--color-accent-wash)',
  paper: 'var(--color-paper-2)',
  white: 'var(--color-white)',
  onAccent: 'var(--color-on-accent)',
  mute: 'var(--color-mute)',
};

export const F = {
  pixel: { fontFamily: 'var(--font-pixel)' },
  mono: { fontFamily: 'var(--font-mono)' },
  sans: { fontFamily: 'var(--font-sans)' },
  cond: { fontFamily: 'var(--font-display)', fontStretch: '62%', fontWeight: 900 },
};

export const inView = { once: true, amount: 0.35 } as const;
export const ease: Transition = { duration: 0.6, ease: [0.2, 0.8, 0.2, 1] };

export function Svg({ w, h, children, label, className }: { w: number; h: number; children: ReactNode; label: string; className?: string }) {
  return (
    <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label} className={`w-full h-auto overflow-visible ${className ?? ''}`}>
      <Defs />
      {children}
    </svg>
  );
}

/** Shared patterns + markers. IDs are global but identical everywhere, so duplicates are harmless. */
export function Defs() {
  return (
    <defs>
      <pattern id="ht-ink" width="5" height="5" patternUnits="userSpaceOnUse">
        <circle cx="2.5" cy="2.5" r="1" fill={C.ink} />
      </pattern>
      <pattern id="ht-accent" width="6" height="6" patternUnits="userSpaceOnUse">
        <circle cx="3" cy="3" r="1.2" fill={C.accent} />
      </pattern>
      <pattern id="stripe" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <line x1="0" y1="0" x2="0" y2="6" stroke={C.ink} strokeWidth="1.4" />
      </pattern>
      <pattern id="noise" width="12" height="12" patternUnits="userSpaceOnUse">
        {[
          [1, 1], [5, 2], [9, 0], [3, 5], [8, 6], [0, 9], [6, 10], [10, 9], [2, 7], [11, 4],
        ].map(([x, y], i) => (
          <rect key={i} x={x} y={y} width="2" height="2" fill={C.ink} opacity={0.35 + (i % 3) * 0.2} />
        ))}
      </pattern>
      <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0 0L10 5 0 10z" fill={C.ink} />
      </marker>
      <marker id="arr-a" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0 0L10 5 0 10z" fill={C.accent} />
      </marker>
    </defs>
  );
}

/** Solid accent label tag. */
export function Tag({ x, y, children, tone = 'accent', size = 11, anchor = 'start' }: { x: number; y: number; children: string; tone?: 'accent' | 'ink' | 'soft' | 'white'; size?: number; anchor?: 'start' | 'middle' | 'end' }) {
  const w = children.length * size * 0.8 + 12;
  const h = size + 8;
  const x0 = anchor === 'middle' ? x - w / 2 : anchor === 'end' ? x - w : x;
  const fill = { accent: C.accent, ink: C.ink, soft: C.soft, white: C.white }[tone];
  const color = tone === 'soft' || tone === 'white' ? C.ink : tone === 'accent' ? C.onAccent : C.white;
  return (
    <g>
      <rect x={x0} y={y - h / 2} width={w} height={h} fill={fill} stroke={tone === 'white' ? C.ink : 'none'} strokeWidth="1.5" />
      <text x={x0 + w / 2} y={y + size * 0.36} textAnchor="middle" fontSize={size} fill={color} style={F.pixel}>
        {children}
      </text>
    </g>
  );
}

export function Num({ x, y, n }: { x: number; y: number; n: number | string }) {
  return (
    <g>
      <circle cx={x} cy={y} r="10" fill={C.white} stroke={C.accent} strokeWidth="1.5" />
      <text x={x} y={y + 4} textAnchor="middle" fontSize="11" fill={C.accent} style={F.mono} fontWeight="700">
        {n}
      </text>
    </g>
  );
}

/** Thin accent connector, optionally elbowed, drawn on scroll. */
export function Wire({ d, accent = true, delay = 0, dashed, arrow }: { d: string; accent?: boolean; delay?: number; dashed?: boolean; arrow?: boolean }) {
  return (
    <motion.path
      d={d}
      fill="none"
      stroke={accent ? C.accent : C.ink}
      strokeWidth="1.5"
      strokeDasharray={dashed ? '4 4' : undefined}
      markerEnd={arrow ? (accent ? 'url(#arr-a)' : 'url(#arr)') : undefined}
      initial={{ pathLength: 0, opacity: 0 }}
      whileInView={{ pathLength: 1, opacity: 1 }}
      viewport={inView}
      transition={{ ...ease, duration: 0.9, delay }}
    />
  );
}

/** Selection box with corner handles. */
export function Sel({ x, y, w, h, dashed }: { x: number; y: number; w: number; h: number; dashed?: boolean }) {
  const s = 6;
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} fill="none" stroke={C.accent} strokeWidth="1.3" strokeDasharray={dashed ? '4 3' : undefined} />
      {[
        [x, y],
        [x + w, y],
        [x, y + h],
        [x + w, y + h],
      ].map(([cx, cy], i) => (
        <rect key={i} x={cx - s / 2} y={cy - s / 2} width={s} height={s} fill={C.white} stroke={C.accent} strokeWidth="1.3" />
      ))}
    </g>
  );
}

/** Hard-shadow box (cut-out look). */
export function Card({ x, y, w, h, fill = C.white, stroke = C.ink, shadow = true, children }: { x: number; y: number; w: number; h: number; fill?: string; stroke?: string; shadow?: boolean; children?: ReactNode }) {
  return (
    <g>
      {shadow && <rect x={x + 4} y={y + 4} width={w} height={h} fill={C.ink} />}
      <rect x={x} y={y} width={w} height={h} fill={fill} stroke={stroke} strokeWidth="2" />
      {children}
    </g>
  );
}

/** A tiny "frame" thumbnail with an abstract dot that moves with t∈[0,1]. */
export function FrameThumb({ x, y, s, t, fill = C.white, dot = C.ink, noise = 0 }: { x: number; y: number; s: number; t: number; fill?: string; dot?: string; noise?: number }) {
  return (
    <g>
      <rect x={x} y={y} width={s} height={s} fill={fill} stroke={C.ink} strokeWidth="1.5" />
      {noise < 1 && <circle cx={x + s * 0.5} cy={y + s * (0.78 - 0.56 * t)} r={s * 0.13} fill={dot} opacity={1 - noise} />}
      {noise > 0 && <rect x={x + 1} y={y + 1} width={s - 2} height={s - 2} fill="url(#noise)" opacity={noise} />}
    </g>
  );
}
