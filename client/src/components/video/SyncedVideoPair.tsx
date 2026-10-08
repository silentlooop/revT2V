import type { VideoOut } from '../../api/types';
import type { Clock } from '../../hooks/useClock';
import { cx } from '../../lib/format';
import { ClipSurface, FlowOverlay } from './ClipSurface';

export function VideoPanel({
  video,
  clock,
  title,
  sub,
  tone,
  showFlow,
  className,
}: {
  video: VideoOut;
  clock: Clock;
  title: string;
  sub: string;
  tone: 'ink' | 'accent' | 'soft';
  showFlow: boolean;
  className?: string;
}) {
  const head = { ink: 'bg-ink text-white', accent: 'bg-accent text-on-accent', soft: 'bg-accent-wash text-accent-ink' }[tone];
  return (
    <figure className={cx('border-2 border-ink bg-ink', className)}>
      <figcaption className={cx('flex items-center justify-between px-2 py-1', head)}>
        <span className="font-pixel text-[10px]">{title}</span>
        <span className="font-mono text-[10px] opacity-80">{sub}</span>
      </figcaption>
      <div className="relative">
        <ClipSurface url={video.video_url} frame={clock.frame} fps={video.fps} label={`${title}, ${sub}, frame ${clock.frame}`} />
        {showFlow && <FlowOverlay flow={video.flow} frame={clock.frame} />}
      </div>
    </figure>
  );
}

export function SyncedVideoPair({
  teacher,
  student,
  teacherReversed,
  showReversed,
  showFlow,
  clock,
}: {
  teacher: VideoOut;
  student: VideoOut;
  teacherReversed?: VideoOut;
  showReversed: boolean;
  showFlow: boolean;
  clock: Clock;
}) {
  const three = showReversed && teacherReversed;
  return (
    <div className={cx('grid gap-3', three ? 'grid-cols-2 sm:grid-cols-3' : 'grid-cols-2')}>
      <VideoPanel video={teacher} clock={clock} title="TEACHER" sub="forward →" tone="ink" showFlow={showFlow} />
      {three && <VideoPanel video={teacherReversed} clock={clock} title="TEACHER REV" sub="reference" tone="soft" showFlow={showFlow} className="max-sm:col-span-2 max-sm:order-last" />}
      <VideoPanel video={student} clock={clock} title="STUDENT" sub="← reverse" tone="accent" showFlow={showFlow} />
    </div>
  );
}
