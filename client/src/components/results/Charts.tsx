import { motion } from 'framer-motion';
import type { LossCurve } from '../../api/types';
import { TodoSticker } from '../ui/Sticker';

const FONT_MONO = { fontFamily: 'var(--font-mono)' };

/** Training loss curve for one student (total + components). */
export function LossChart({ curve }: { curve: LossCurve }) {
  const W = 320;
  const H = 170;
  const P = { l: 34, r: 8, t: 10, b: 24 };
  const all = [curve.total, ...(curve.components?.map((c) => c.values) ?? [])].flat();
  const ymax = Math.max(...all) * 1.08;
  const xmax = Math.max(...curve.steps);
  const sx = (s: number) => P.l + ((W - P.l - P.r) * s) / xmax;
  const sy = (v: number) => H - P.b - ((H - P.t - P.b) * v) / ymax;
  const path = (vals: number[]) => vals.map((v, i) => `${i ? 'L' : 'M'}${sx(curve.steps[i]).toFixed(1)} ${sy(v).toFixed(1)}`).join(' ');
  const dashes = ['5 3', '2 3'];
  return (
    <figure className="bg-white border-2 border-ink p-3">
      <figcaption className="flex items-center gap-2 mb-1">
        <span className="font-pixel text-xs">{curve.label}</span>
        {curve.todo && <TodoSticker className="ml-auto" />}
      </figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label={`Training loss of ${curve.label}, from ${curve.total[0]} to ${curve.total[curve.total.length - 1]}`}>
        {[0, 0.25, 0.5, 0.75, 1].map((k) => (
          <g key={k}>
            <line x1={P.l} x2={W - P.r} y1={sy(ymax * k)} y2={sy(ymax * k)} stroke="var(--color-rule)" />
            <text x={P.l - 4} y={sy(ymax * k) + 3} textAnchor="end" fontSize="8" style={FONT_MONO} fill="var(--color-mute)">
              {(ymax * k).toFixed(2)}
            </text>
          </g>
        ))}
        {[0, 0.5, 1].map((k) => (
          <text key={k} x={sx(xmax * k)} y={H - 8} textAnchor={k === 1 ? 'end' : k === 0 ? 'start' : 'middle'} fontSize="8" style={FONT_MONO} fill="var(--color-mute)">
            {k === 1 ? `${Math.round(xmax)} steps` : Math.round(xmax * k)}
          </text>
        ))}
        <line x1={P.l} x2={W - P.r} y1={H - P.b} y2={H - P.b} stroke="var(--color-ink)" strokeWidth="1.5" />
        <line x1={P.l} x2={P.l} y1={P.t} y2={H - P.b} stroke="var(--color-ink)" strokeWidth="1.5" />
        {curve.components?.map((c, i) => (
          <path key={c.name} d={path(c.values)} fill="none" stroke="var(--color-ink)" strokeWidth="1.3" strokeDasharray={dashes[i % 2]} />
        ))}
        <motion.path d={path(curve.total)} fill="none" stroke="var(--color-accent)" strokeWidth="2.4" initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }} viewport={{ once: true }} transition={{ duration: 1.2 }} />
      </svg>
      <div className="flex flex-wrap gap-3 text-[11px] font-mono mt-1">
        <span className="flex items-center gap-1">
          <span className="w-4 h-0.5 bg-accent" /> total
        </span>
        {curve.components?.map((c, i) => (
          <span key={c.name} className="flex items-center gap-1">
            <svg width="16" height="4" aria-hidden>
              <line x1="0" y1="2" x2="16" y2="2" stroke="var(--color-ink)" strokeWidth="1.5" strokeDasharray={dashes[i % 2]} />
            </svg>
            {c.name}
          </span>
        ))}
        <span className="ml-auto text-mute">final {curve.total[curve.total.length - 1].toFixed(3)}</span>
      </div>
    </figure>
  );
}
