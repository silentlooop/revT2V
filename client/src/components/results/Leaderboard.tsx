import type { LeaderboardRow, Method, MetricColumn } from '../../api/types';
import { cx, fmtNum } from '../../lib/format';
import { KindTag, TodoSticker } from '../ui/Sticker';
import { InfoTip } from '../ui/Tooltip';

export function bestPerColumn(rows: LeaderboardRow[], cols: MetricColumn[]) {
  const best: Record<string, string | null> = {};
  for (const c of cols) {
    let bm: string | null = null;
    let bv = 0;
    for (const r of rows) {
      const v = r[c.key];
      if (v == null) continue;
      const score = c.better === 'higher' ? v : c.better === 'lower' ? -v : -Math.abs(v - 1);
      if (bm == null || score > bv) {
        bm = r.method;
        bv = score;
      }
    }
    best[c.key] = bm;
  }
  return best;
}

export function Leaderboard({ rows, cols, methods }: { rows: LeaderboardRow[]; cols: MetricColumn[]; methods: Method[] }) {
  const best = bestPerColumn(rows, cols);
  const kindOf = (id: string) => methods.find((m) => m.id === id)?.kind ?? 'trained';
  return (
    <div className="overflow-x-auto scroll-thin border-2 border-ink bg-white shadow-hard">
      <table className="w-full min-w-[560px] text-sm">
        <caption className="sr-only">Set-level results. Best value per column highlighted.</caption>
        <thead>
          <tr className="border-b-2 border-ink bg-paper">
            <th scope="col" className="text-left px-4 py-2 font-mono text-xs font-normal text-mute">
              method
            </th>
            {cols.map((c) => (
              <th key={c.key} scope="col" className="text-right px-4 py-2">
                <span className="inline-flex items-center gap-1.5">
                  <span className="font-pixel text-[10px]">{c.label}</span>
                  <span className="font-mono text-[10px] text-mute">{c.better === 'lower' ? '↓' : '↑'}</span>
                  <InfoTip label={`About ${c.label}`} align="right">
                    {c.hint}
                  </InfoTip>
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const main = kindOf(r.method) === 'main';
            return (
              <tr key={r.method} className={cx('border-b border-ink/15 last:border-0', main && 'bg-accent-wash')}>
                <th scope="row" className="text-left px-4 py-3">
                  <span className="flex items-center gap-2">
                    <span className="font-mono font-bold">{r.method}</span>
                    <KindTag kind={kindOf(r.method)} />
                    {r.todo && <TodoSticker />}
                  </span>
                </th>
                {cols.map((c) => {
                  const isBest = best[c.key] === r.method;
                  return (
                    <td key={c.key} className="px-4 py-3 text-right">
                      <span className={cx('condensed text-2xl tabular-nums', isBest && 'text-accent underline decoration-2 underline-offset-4')}>{fmtNum(r[c.key], c.key === 'fvd' ? 0 : 2)}</span>
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
