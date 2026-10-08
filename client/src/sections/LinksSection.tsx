import { LINKS, REFERENCES } from '../content/links';
import { cx } from '../lib/format';

function Logo({ kind }: { kind: string }) {
  if (kind === 'GitHub')
    return (
      <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden>
        <path
          fill="currentColor"
          d="M12 .5a12 12 0 0 0-3.8 23.4c.6.1.8-.3.8-.6v-2.1c-3.3.7-4-1.6-4-1.6-.6-1.4-1.4-1.8-1.4-1.8-1.1-.7.1-.7.1-.7 1.2.1 1.9 1.2 1.9 1.2 1.1 1.9 2.9 1.3 3.6 1 .1-.8.4-1.3.8-1.6-2.7-.3-5.5-1.3-5.5-5.9 0-1.3.5-2.4 1.2-3.2-.1-.3-.5-1.5.1-3.2 0 0 1-.3 3.3 1.2a11.5 11.5 0 0 1 6 0C17.3 4.7 18.3 5 18.3 5c.7 1.7.2 2.9.1 3.2.8.8 1.2 1.9 1.2 3.2 0 4.6-2.8 5.6-5.5 5.9.4.4.8 1.1.8 2.2v3.3c0 .3.2.7.8.6A12 12 0 0 0 12 .5z"
        />
      </svg>
    );
  return null;
}

export function LinksSection() {
  return (
    <section id="links" className="border-b-2 border-ink" aria-labelledby="links-title">
      <div className="mx-auto max-w-6xl px-4 md:px-8 py-10">
        <h2 id="links-title" className="font-pixel text-xs uppercase mb-4">
          Links
        </h2>
        <div className="grid sm:grid-cols-3 gap-4">
          {LINKS.map((l) => {
            const live = !!l.url;
            const body = (
              <>
                <div className="flex items-center gap-2">
                  <Logo kind={l.kind} />
                  <span className="font-mono text-xs text-mute">{l.kind}</span>
                  <span className="ml-auto text-lg" aria-hidden>
                    {live ? '↗' : ''}
                  </span>
                </div>
                <p className="condensed text-2xl uppercase leading-none mt-3">{l.label}</p>
                <p className={cx('font-mono text-xs mt-1 break-all', live ? 'text-accent' : 'text-bad')}>{live ? l.repo : 'link coming soon'}</p>
              </>
            );
            return live ? (
              <a key={l.repo} href={l.url} target="_blank" rel="noopener noreferrer" className="block bg-white border-2 border-ink p-4 shadow-hard hover:-translate-y-0.5 hover:shadow-hard-accent transition-[transform,box-shadow]">
                {body}
              </a>
            ) : (
              <div key={l.repo} className="bg-white border-2 border-dashed border-ink/40 p-4">
                {body}
              </div>
            );
          })}
        </div>

        <h2 className="font-pixel text-xs uppercase mt-10 mb-4">References</h2>
        <ol className="grid md:grid-cols-2 gap-x-8 gap-y-2 text-sm">
          {REFERENCES.map((r, i) => (
            <li key={r.title} className="flex gap-3 min-w-0">
              <span className="font-mono text-xs text-accent pt-0.5 w-6 shrink-0">[{i + 1}]</span>
              <span className="min-w-0">
                {r.url ? (
                  <a href={r.url} target="_blank" rel="noopener noreferrer" className="font-medium hover:text-accent hover:underline underline-offset-2">
                    {r.title} <span aria-hidden>↗</span>
                  </a>
                ) : (
                  <span className="font-medium">{r.title}</span>
                )}
                <span className="block font-mono text-[11px] text-mute">
                  {r.note}
                  {r.note && !r.url && ' · '}
                  {!r.url && <span className="text-bad">link needed</span>}
                </span>
              </span>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
