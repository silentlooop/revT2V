import { useEffect, useState } from 'react';
import type { HealthStatus } from '../api/types';
import { StatusBadge } from '../components/chat/StatusBadge';
import { cx } from '../lib/format';

const LINKS = [
  { id: 'generate', n: '01', label: 'Generate' },
  { id: 'findings', n: '02', label: 'Findings' },
  { id: 'theory', n: '03', label: 'Theory' },
];

export function Nav({ health }: { health: HealthStatus | null }) {
  const [active, setActive] = useState('generate');
  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) => {
        const vis = entries.filter((e) => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (vis[0]) setActive(vis[0].target.id);
      },
      { rootMargin: '-40% 0px -55% 0px' },
    );
    LINKS.forEach((l) => {
      const el = document.getElementById(l.id);
      if (el) obs.observe(el);
    });
    return () => obs.disconnect();
  }, []);

  return (
    <nav className="sticky top-0 z-50 bg-white/95 backdrop-blur-[2px] border-b-2 border-ink" aria-label="Sections">
      <div className="mx-auto max-w-6xl px-4 md:px-8 h-14 flex items-center gap-2 sm:gap-3">
        <a href="#top" className="font-pixel text-sm md:text-base flex items-center gap-1.5 shrink-0">
          <span className="bg-accent text-on-accent px-1.5">rev</span>
          <span className="hidden sm:inline">T2V</span>
        </a>
        <ul className="flex items-stretch h-full ml-1 md:ml-6 overflow-x-auto min-w-0">
          {LINKS.map((l) => (
            <li key={l.id}>
              <a
                href={`#${l.id}`}
                aria-current={active === l.id ? 'location' : undefined}
                className={cx('h-full flex items-center gap-1.5 px-1.5 sm:px-2 md:px-4 border-x border-transparent text-[13px] sm:text-sm whitespace-nowrap', active === l.id ? 'bg-accent text-on-accent' : 'hover:bg-accent-wash')}
              >
                <span className="font-mono text-[10px] opacity-70 hidden sm:inline">{l.n}</span>
                <span className="font-semibold">{l.label}</span>
              </a>
            </li>
          ))}
        </ul>
        <div className="ml-auto flex items-center gap-2">
          <span className="hidden md:inline-flex">
            <StatusBadge health={health} />
          </span>
          <span className="md:hidden">
            <StatusBadge health={health} compact />
          </span>
        </div>
      </div>
    </nav>
  );
}
