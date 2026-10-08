import { useMemo, useState } from 'react';
import { buildPair } from '../../api/mock';
import { GALLERY } from '../../content/generate';
import { FolderIcon } from '../ui/Icons';
import { Win98Button, Win98Window } from '../ui/Win98Window';
import { ResultCard } from './ResultCard';

export function OfflineGallery({ onRetry, onClose }: { onRetry: () => void; onClose?: () => void }) {
  const [open, setOpen] = useState<number | null>(0);
  const pairs = useMemo(() => GALLERY.map((req, i) => buildPair(`example_${i + 1}`, req, 180 + i * 17)), []);
  return (
    <div className="grid gap-4">
      <Win98Window title="Backend offline" onClose={onClose}>
        <div className="flex flex-wrap items-center gap-3">
          <p className="text-[13px] flex-1 min-w-48">The GPU server is offline. Browse saved examples:</p>
          <Win98Button onClick={onRetry}>Retry</Win98Button>
        </div>
        <div className="mt-2 grid grid-cols-3 sm:grid-cols-5 gap-1 bg-win-light shadow-win-in p-1.5">
          {pairs.map((p, i) => (
            <button
              key={p.job_id}
              type="button"
              onClick={() => setOpen(i)}
              aria-pressed={open === i}
              className="flex flex-col items-center gap-1 p-1 text-[11px] text-center hover:bg-accent-wash aria-pressed:bg-accent aria-pressed:text-on-accent"
            >
              <FolderIcon size={32} />
              <span className="line-clamp-1">{p.request.prompt.split(' ').slice(0, 3).join(' ')}</span>
            </button>
          ))}
        </div>
      </Win98Window>
      {open != null && <ResultCard key={pairs[open].job_id} result={pairs[open]} example />}
    </div>
  );
}
