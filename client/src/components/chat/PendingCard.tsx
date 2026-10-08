import { useEffect, useState } from 'react';
import type { GenerateRequest, Job } from '../../api/types';
import { cx, fmtTime } from '../../lib/format';
import { MosaicProgress } from '../video/MosaicProgress';

export function PendingCard({ req, job, error, onRetry, onDismiss }: { req: GenerateRequest; job: Job | null; error?: string; onRetry?: () => void; onDismiss?: () => void }) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, []);

  const total = req.steps * 2;
  const done = job ? (job.status === 'done' ? total : job.current_model === 'teacher' ? job.step : req.steps + job.step) : 0;
  const progress = done / total;
  const elapsed = job ? Math.max(job.elapsed_s, (now - Date.parse(job.created_at)) / 1000) : 0;
  const queued = job?.status === 'queued';
  const phase = error ? 'Failed' : !job || queued ? 'Queued' : job.current_model === 'teacher' ? 'Teacher' : 'Student';

  return (
    <div className="w-full max-w-lg bg-white border-2 border-ink shadow-hard flex" aria-live="polite" aria-busy={!error}>
      <div className="w-36 shrink-0 border-r-2 border-ink bg-ink">
        <MosaicProgress prompt={req.prompt} seed={req.seed} progress={progress} />
      </div>
      <div className="flex-1 min-w-0 p-3 flex flex-col gap-2">
        <div className="flex items-baseline justify-between gap-2">
          <span className={cx('condensed text-2xl uppercase leading-none', error && 'text-bad')}>{phase}</span>
          <span className="font-mono text-xs tabular-nums text-mute">{fmtTime(elapsed)}</span>
        </div>
        {error ? (
          <>
            <p className="font-mono text-xs text-bad break-words">{error}</p>
            <div className="flex gap-2 mt-auto">
              {onRetry && (
                <button type="button" onClick={onRetry} className="bg-accent text-on-accent px-3 py-1 text-xs font-bold">
                  Retry
                </button>
              )}
              {onDismiss && (
                <button type="button" onClick={onDismiss} className="border border-ink px-3 py-1 text-xs font-bold">
                  Dismiss
                </button>
              )}
            </div>
          </>
        ) : (
          <>
            <p className="font-mono text-xs text-mute">
              {queued ? `queue #${job?.queue_pos}` : job ? `step ${Math.min(job.step, job.total_steps)}/${job.total_steps}` : 'submitting…'} · {req.student_method}
            </p>
            <div className="mt-auto grid grid-cols-2 gap-1" aria-hidden>
              {(['teacher', 'student'] as const).map((m, i) => {
                const frac = Math.min(1, Math.max(0, (done - i * req.steps) / req.steps));
                return (
                  <div key={m}>
                    <div className="h-1.5 bg-paper">
                      <div className={cx('h-full transition-[width] duration-300', i ? 'bg-accent' : 'bg-ink')} style={{ width: `${frac * 100}%` }} />
                    </div>
                    <span className="font-mono text-[10px] text-mute">{m}</span>
                  </div>
                );
              })}
            </div>
            <span className="sr-only" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress * 100)} aria-label="generation progress" />
          </>
        )}
      </div>
    </div>
  );
}
