import { useState } from 'react';
import type { PairResult } from '../../api/types';
import { useClock } from '../../hooks/useClock';
import { DownloadIcon } from '../ui/Icons';
import { Sticker } from '../ui/Sticker';
import { Toggle, Transport } from '../ui/Transport';
import { MetricsRow } from '../metrics/MetricSticker';
import { downloadClip } from '../video/ClipSurface';
import { FrameStrip } from '../video/FrameStrip';
import { SyncedVideoPair } from '../video/SyncedVideoPair';

export function ResultCard({ result, example }: { result: PairResult; example?: boolean }) {
  const clock = useClock(result.teacher.num_frames, result.teacher.fps);
  const [showReversed, setShowReversed] = useState(false);
  const [showFlow, setShowFlow] = useState(false);
  const [showFrames, setShowFrames] = useState(false);
  const r = result.request;
  const v = result.teacher;

  return (
    <article className="w-full bg-white border-2 border-ink shadow-hard" aria-label={`Result for prompt: ${r.prompt}`}>
      <header className="flex items-center gap-3 px-4 py-3 border-b-2 border-ink">
        <p className="flex-1 min-w-0 font-semibold leading-snug truncate" title={r.prompt}>
          {r.prompt}
        </p>
        {example && <Sticker tone="outline">example</Sticker>}
        <Sticker tone={r.student_method === 'conv_lora' ? 'accent' : 'ink'}>{r.student_method}</Sticker>
      </header>

      <div className="p-4 grid gap-3">
        <SyncedVideoPair
          teacher={result.teacher}
          student={result.student}
          teacherReversed={result.teacher_reversed}
          showReversed={showReversed}
          showFlow={showFlow}
          clock={clock}
        />
        <Transport clock={clock}>
          <span className="flex gap-1.5">
            {result.teacher_reversed && (
              <Toggle on={showReversed} onChange={setShowReversed}>
                reversed
              </Toggle>
            )}
            <Toggle on={showFlow} onChange={setShowFlow}>
              flow
            </Toggle>
            <Toggle on={showFrames} onChange={setShowFrames}>
              frames
            </Toggle>
          </span>
        </Transport>
        {showFrames && (
          <div className="grid sm:grid-cols-2 gap-3">
            <FrameStrip video={result.teacher} label="teacher" />
            <FrameStrip video={result.student} label="student" />
          </div>
        )}
      </div>

      <div className="px-4 pb-4">
        <MetricsRow m={result.metrics} />
      </div>

      <footer className="flex flex-wrap items-center gap-x-4 gap-y-1 px-4 py-2 border-t-2 border-ink font-mono text-[11px] text-mute">
        <span>seed {r.seed}</span>
        <span>
          {v.width}×{v.height}×{v.num_frames}
        </span>
        <span>{r.steps} steps</span>
        <span>cfg {r.cfg}</span>
        <span>{result.time_taken_s}s</span>
        <span className="ml-auto flex gap-3 text-ink">
          {(['teacher', 'student'] as const).map((k) => (
            <button key={k} type="button" onClick={() => downloadClip(result[k].video_url, `${result.job_id}_${k}`)} className="flex items-center gap-1 hover:text-accent">
              <DownloadIcon size={12} /> {k}
            </button>
          ))}
        </span>
      </footer>
    </article>
  );
}
