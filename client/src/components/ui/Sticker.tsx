import type { ReactNode } from 'react';
import { cx } from '../../lib/format';

/** Small label tag. */
export function Sticker({ children, tone = 'accent', className }: { children: ReactNode; tone?: 'accent' | 'ink' | 'outline' | 'soft'; className?: string }) {
  const tones = {
    accent: 'bg-accent text-on-accent',
    ink: 'bg-ink text-white',
    outline: 'border border-ink text-ink',
    soft: 'bg-accent-wash text-accent-ink',
  };
  return <span className={cx('inline-flex items-center font-pixel text-[10px] uppercase leading-none px-1.5 py-1 whitespace-nowrap', tones[tone], className)}>{children}</span>;
}

const KIND: Record<string, { label: string; tone: 'accent' | 'ink' | 'outline' | 'soft' }> = {
  main: { label: 'main', tone: 'accent' },
  trained: { label: 'LoRA', tone: 'soft' },
  training_free: { label: 'no training', tone: 'outline' },
};

export function KindTag({ kind, className }: { kind: string; className?: string }) {
  const k = KIND[kind] ?? { label: kind, tone: 'outline' as const };
  return (
    <Sticker tone={k.tone} className={className}>
      {k.label}
    </Sticker>
  );
}

export function TodoSticker({ className }: { className?: string }) {
  return (
    <span className={cx('inline-block font-pixel text-[9px] leading-none text-bad border border-bad px-1 py-0.5', className)} title="placeholder: replace in src/content/results.ts">
      TODO
    </span>
  );
}
