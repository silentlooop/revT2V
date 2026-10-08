import type { ReactNode } from 'react';

export function SectionHeader({ num, id, title, right }: { num: string; id: string; title: string; right?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-end gap-x-4 gap-y-2 mb-6">
      <span className="font-mono text-sm text-accent pb-2" aria-hidden>
        {num}
      </span>
      <h2 id={id} className="condensed text-6xl md:text-7xl uppercase leading-[0.85]">
        {title}
      </h2>
      {right && <div className="ml-auto pb-1">{right}</div>}
    </div>
  );
}
