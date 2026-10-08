import { useId, useState, type ReactNode } from 'react';
import { cx } from '../../lib/format';

/** Accessible "?" tooltip: opens on hover, focus or tap; Escape closes. */
export function InfoTip({ children, label = 'What is this?', className, align = 'left' }: { children: ReactNode; label?: string; className?: string; align?: 'left' | 'right' }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <span className={cx('relative inline-flex', className)} onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <button
        type="button"
        aria-label={label}
        aria-describedby={open ? id : undefined}
        aria-expanded={open}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={(e) => e.key === 'Escape' && setOpen(false)}
        className="grid place-items-center size-[18px] rounded-full border-[1.5px] border-current font-bold text-[11px] leading-none hover:bg-accent hover:text-on-accent hover:border-accent"
      >
        ?
      </button>
      {open && (
        <span
          role="tooltip"
          id={id}
          className={cx(
            'absolute z-50 top-full mt-2 w-64 bg-ink text-white text-xs font-sans font-normal normal-case tracking-normal leading-snug p-2.5 shadow-hard-accent text-left',
            align === 'left' ? 'left-0' : 'right-0',
          )}
        >
          {children}
        </span>
      )}
    </span>
  );
}
