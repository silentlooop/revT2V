import type { HealthStatus } from '../../api/types';
import { USE_MOCK } from '../../api/client';
import { cx } from '../../lib/format';

export function StatusBadge({ health, compact }: { health: HealthStatus | null; compact?: boolean }) {
  const online = !!health?.online;
  const label = health == null ? 'checking…' : online ? 'online' : 'offline';
  return (
    <span
      className={cx('inline-flex items-center gap-1.5 border-2 py-0.5', compact ? 'px-1.5 py-1.5' : 'px-2', ' font-mono text-[11px] whitespace-nowrap', online ? 'border-ink bg-white' : 'border-bad bg-white text-bad')}
      role="status"
      aria-label={`Backend ${label}${health?.gpu ? `, GPU ${health.gpu}` : ''}`}
    >
      <span className={cx('size-2 rounded-full', health == null ? 'bg-mute animate-blink' : online ? 'bg-good' : 'bg-bad animate-blink')} />
      <span className={compact ? 'sr-only' : ''}>{label}</span>
      {!compact && online && health?.gpu && <span className="text-mute">· {health.gpu}</span>}
      {!compact && USE_MOCK && <span className="bg-accent-soft text-ink px-1">MOCK</span>}
    </span>
  );
}
