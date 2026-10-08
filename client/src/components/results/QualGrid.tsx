import { useState } from 'react';
import type { QualitativeEntry } from '../../api/types';
import { useClock } from '../../hooks/useClock';
import { cx } from '../../lib/format';
import { isMockUrl } from '../../lib/mockClips';
import { Transport } from '../ui/Transport';
import { ClipSurface, FrameImage } from '../video/ClipSurface';

const FRAME_COLS = [0, 5, 10, 15];

/** Rows = methods, columns = frames; everything locked to one clock. */
export function QualGrid({ entries }: { entries: QualitativeEntry[] }) {
  const [tab, setTab] = useState(0);
  const e = entries[tab];
  const clock = useClock(16, 8);
  const methods = Object.keys(e.videos);

  return (
    <div className="bg-white border-2 border-ink shadow-hard">
      <div className="flex overflow-x-auto scroll-thin border-b-2 border-ink" role="tablist" aria-label="Test prompts">
        {entries.map((en, i) => (
          <button
            key={en.prompt}
            role="tab"
            type="button"
            aria-selected={tab === i}
            onClick={() => setTab(i)}
            className={cx('shrink-0 px-4 py-2 text-xs border-r border-ink/20', tab === i ? 'bg-ink text-white' : 'hover:bg-accent-wash')}
          >
            {en.prompt.split(' ').slice(0, 3).join(' ')}
          </button>
        ))}
      </div>

      <div className="overflow-x-auto scroll-thin" role="tabpanel">
        <table className="w-full min-w-[620px] table-fixed border-collapse">
          <thead>
            <tr>
              <th scope="col" className="w-32 text-left px-3 py-2 font-mono text-[10px] font-normal text-mute">
                seed {e.seed}
              </th>
              <th scope="col" className="px-1 py-2 font-mono text-[10px] text-accent">
                ▶
              </th>
              {FRAME_COLS.map((f) => (
                <th key={f} scope="col" className="px-1 py-2 font-mono text-[10px] font-normal text-mute">
                  f{f}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {methods.map((m) => (
              <tr key={m} className={cx('border-t border-ink/15', m === 'conv_lora' && 'bg-accent-wash')}>
                <th scope="row" className={cx('text-left px-3 font-mono text-xs', m === 'conv_lora' ? 'text-accent' : '', m === 'teacher' && 'text-mute')}>
                  {m}
                </th>
                <td className="p-1">
                  <div className={cx('border-2', m === 'conv_lora' ? 'border-accent' : 'border-ink')}>
                    <ClipSurface url={e.videos[m]} frame={clock.frame} label={`${m}, live`} />
                  </div>
                </td>
                {FRAME_COLS.map((f) => (
                  <td key={f} className="p-1">
                    <button type="button" onClick={() => clock.seek(f)} className={cx('block w-full border', clock.frame === f ? 'border-accent outline-2 outline-accent' : 'border-ink/40')} aria-label={`${m} frame ${f}`}>
                      {isMockUrl(e.videos[m]) ? <FrameImage src={`${e.videos[m]}&frame=${f}`} alt="" /> : <ClipSurface url={e.videos[m]} frame={f} label="" />}
                    </button>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <Transport clock={clock} className="border-t-2 border-ink px-3 py-2" />
    </div>
  );
}
