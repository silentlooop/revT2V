import type { ReactNode } from 'react';
import { cx } from '../../lib/format';

export function Win98Window({
  title,
  children,
  className,
  onClose,
  icon,
  bodyClassName,
}: {
  title: string;
  children: ReactNode;
  className?: string;
  onClose?: () => void;
  icon?: ReactNode;
  bodyClassName?: string;
}) {
  return (
    <div className={cx('bg-win shadow-win-out p-[3px] font-win text-[13px] text-win-darker', className)} role="dialog" aria-label={title}>
      <div className="flex items-center gap-1.5 px-1.5 py-0.5 bg-accent text-on-accent font-bold select-none">
        {icon}
        <span className="truncate flex-1">{title}</span>
        <span className="flex gap-0.5">
          <span aria-hidden className="grid place-items-center size-4 bg-win text-win-darker shadow-win-out text-[10px] leading-none">_</span>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            disabled={!onClose}
            className="grid place-items-center size-4 bg-win text-win-darker shadow-win-out text-[11px] font-bold leading-none active:shadow-win-in disabled:cursor-default"
          >
            ×
          </button>
        </span>
      </div>
      <div className={cx('p-3', bodyClassName)}>{children}</div>
    </div>
  );
}

export function Win98Button({ children, className, ...rest }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      type="button"
      className={cx('bg-win shadow-win-out px-4 py-1 min-w-20 font-win text-[13px] active:shadow-win-in focus-visible:outline-dotted focus-visible:outline-1 focus-visible:-outline-offset-4', className)}
      {...rest}
    >
      {children}
    </button>
  );
}
