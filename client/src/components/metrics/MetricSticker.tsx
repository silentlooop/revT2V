import type { PairMetrics } from '../../api/types';
import { METRIC_HELP } from '../../content/generate';
import { cx, fmtNum } from '../../lib/format';
import { InfoTip } from '../ui/Tooltip';

type Help = (typeof METRIC_HELP)[keyof typeof METRIC_HELP];
type Tone = 'good' | 'mid' | 'bad';

function Tile({ help, value, sub, tone, verdict, note }: { help: Help; value: string; sub: React.ReactNode; tone: Tone; verdict: string; note?: string }) {
  return (
    <section className="border-2 border-ink p-3 flex flex-col gap-1 min-w-0" aria-label={`${help.title} metric`}>
      <header className="flex items-center gap-1.5">
        <span className="font-pixel text-[10px]">{help.title}</span>
        <span className="font-mono text-[10px] text-mute">{help.better === 'lower' ? '↓' : '↑'}</span>
        <span className="ml-auto text-mute">
          <InfoTip label={`About ${help.title}`} align="right">
            {help.plain}
            {note && <span className="block mt-1 opacity-70">{note}</span>}
          </InfoTip>
        </span>
      </header>
      <span className="condensed text-4xl leading-none tabular-nums">{value}</span>
      <div className="font-mono text-[11px] text-mute">{sub}</div>
      <span
        className={cx(
          'mt-1 self-start font-pixel text-[9px] px-1.5 py-0.5 leading-none',
          tone === 'good' && 'bg-accent text-on-accent',
          tone === 'mid' && 'bg-accent-wash text-accent-ink',
          tone === 'bad' && 'bg-ink text-white',
        )}
      >
        {verdict}
      </span>
    </section>
  );
}

export function MetricsRow({ m }: { m: PairMetrics }) {
  const clipD = m.clip.student - m.clip.teacher;
  const dynR = m.dynamic_degree.teacher > 0 ? m.dynamic_degree.student / m.dynamic_degree.teacher : 0;
  const cos = m.optical_flow.direction_cosine;
  const fvdTone: Tone = m.fvd < 150 ? 'good' : m.fvd < 450 ? 'mid' : 'bad';
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-2">
      <Tile help={METRIC_HELP.fvd} value={fmtNum(m.fvd, 0)} sub="vs reversed teacher" tone={fvdTone} verdict={fvdTone === 'good' ? 'close' : fvdTone === 'mid' ? 'ok' : 'far'} note={m.notes.fvd} />
      <Tile
        help={METRIC_HELP.clip}
        value={fmtNum(m.clip.student, 1)}
        sub={`teacher ${fmtNum(m.clip.teacher, 1)}`}
        tone={clipD > -0.75 ? 'good' : clipD > -1.5 ? 'mid' : 'bad'}
        verdict={`Δ ${clipD >= 0 ? '+' : ''}${clipD.toFixed(1)}`}
        note={m.notes.clip}
      />
      <Tile
        help={METRIC_HELP.dynamic_degree}
        value={fmtNum(m.dynamic_degree.student, 2)}
        sub={`teacher ${fmtNum(m.dynamic_degree.teacher, 2)}`}
        tone={dynR > 0.8 ? 'good' : dynR > 0.4 ? 'mid' : 'bad'}
        verdict={`${Math.round(dynR * 100)}%`}
        note={m.notes.dynamic_degree}
      />
      <Tile
        help={METRIC_HELP.optical_flow}
        value={cos.toFixed(2)}
        sub={`mag ${fmtNum(m.optical_flow.student_mag, 1)} / ${fmtNum(m.optical_flow.teacher_mag, 1)}`}
        tone={m.optical_flow.frozen ? 'bad' : cos > 0.6 ? 'good' : cos > 0.1 ? 'mid' : 'bad'}
        verdict={m.optical_flow.frozen ? 'frozen' : cos > 0.6 ? 'reversed' : cos > 0.1 ? 'partial' : 'not reversed'}
        note={m.notes.optical_flow}
      />
    </div>
  );
}
