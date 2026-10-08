import type { Clock } from '../../hooks/useClock';
import { cx } from '../../lib/format';

/** Compact transport: play/pause, frame scrubber, loop, speed. Drives every video bound to `clock`. */
export function Transport({ clock, className, children }: { clock: Clock; className?: string; children?: React.ReactNode }) {
  return (
    <div className={cx('flex flex-wrap items-center gap-3', className)}>
      <button
        type="button"
        onClick={clock.toggle}
        aria-label={clock.playing ? 'Pause' : 'Play'}
        className="grid place-items-center size-9 rounded-full bg-ink text-white hover:bg-accent hover:text-on-accent shrink-0"
      >
        {clock.playing ? (
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden>
            <rect x="2" y="1" width="3.5" height="12" fill="currentColor" />
            <rect x="8.5" y="1" width="3.5" height="12" fill="currentColor" />
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden>
            <path d="M3 1l10 6-10 6z" fill="currentColor" />
          </svg>
        )}
      </button>
      <input
        type="range"
        min={0}
        max={clock.frames - 1}
        value={clock.frame}
        onChange={(e) => {
          clock.pause();
          clock.seek(Number(e.target.value));
        }}
        aria-label="Frame"
        aria-valuetext={`frame ${clock.frame} of ${clock.frames - 1}`}
        className="flex-1 min-w-32 accent-[var(--color-accent)] h-1.5 cursor-pointer"
      />
      <span className="font-mono text-xs tabular-nums w-12 text-right">
        {String(clock.frame).padStart(2, '0')}/{clock.frames - 1}
      </span>
      <button
        type="button"
        onClick={() => clock.setLoop(!clock.loop)}
        aria-pressed={clock.loop}
        className={cx('font-mono text-xs px-2 py-1 border', clock.loop ? 'border-ink bg-ink text-white' : 'border-ink/30 hover:border-ink')}
      >
        loop
      </button>
      <select
        value={clock.speed}
        onChange={(e) => clock.setSpeed(Number(e.target.value))}
        aria-label="Playback speed"
        className="font-mono text-xs border border-ink/30 bg-white px-1.5 py-1"
      >
        {[0.25, 0.5, 1, 2].map((s) => (
          <option key={s} value={s}>
            {s}×
          </option>
        ))}
      </select>
      {children}
    </div>
  );
}

export function Toggle({ on, onChange, children }: { on: boolean; onChange: (v: boolean) => void; children: React.ReactNode }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      onClick={() => onChange(!on)}
      className={cx('font-mono text-xs px-2 py-1 border', on ? 'border-accent bg-accent text-on-accent' : 'border-ink/30 hover:border-ink')}
    >
      {children}
    </button>
  );
}
