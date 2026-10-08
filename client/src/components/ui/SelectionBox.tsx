import type { ReactNode } from 'react';
import { cx } from '../../lib/format';

/** Figma-style selection frame: thin accent border, square corner handles, numbered label. */
export function SelectionBox({
  children,
  label,
  num,
  className,
  labelPos = 'bottom',
  dashed,
}: {
  children: ReactNode;
  label?: ReactNode;
  num?: string;
  className?: string;
  labelPos?: 'top' | 'bottom';
  dashed?: boolean;
}) {
  const h = 'absolute size-2 bg-white border-[1.5px] border-accent';
  return (
    <div className={cx('relative border-[1.5px] border-accent', dashed && 'border-dashed', className)}>
      <span aria-hidden className={cx(h, '-left-1 -top-1')} />
      <span aria-hidden className={cx(h, '-right-1 -top-1')} />
      <span aria-hidden className={cx(h, '-left-1 -bottom-1')} />
      <span aria-hidden className={cx(h, '-right-1 -bottom-1')} />
      {children}
      {(label || num) && (
        <span
          className={cx(
            'absolute left-0 flex items-center gap-1.5 text-[0.7rem] font-sans text-ink',
            labelPos === 'bottom' ? '-bottom-6' : '-top-6',
          )}
        >
          {num && <span className="bg-accent-soft text-ink px-1 font-mono text-[0.62rem] leading-4">{num}.</span>}
          {label}
        </span>
      )}
    </div>
  );
}
