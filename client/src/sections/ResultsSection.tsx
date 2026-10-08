import type { Method } from '../api/types';
import { LossChart } from '../components/results/Charts';
import { FindingCard } from '../components/results/Findings';
import { Leaderboard } from '../components/results/Leaderboard';
import { QualGrid } from '../components/results/QualGrid';
import { TodoSticker } from '../components/ui/Sticker';
import { useResults } from '../hooks/useApi';
import { SectionHeader } from './SectionHeader';

function Sub({ title }: { title: string }) {
  return <h3 className="font-pixel text-xs uppercase mb-3">{title}</h3>;
}

export function ResultsSection({ methods }: { methods: Method[] }) {
  const { results, error } = useResults();

  return (
    <section id="findings" className="border-b-2 border-ink scroll-mt-14" aria-labelledby="findings-title">
      <div className="mx-auto max-w-6xl px-4 md:px-8 py-12">
        <SectionHeader
          num="02"
          id="findings-title"
          title="Findings"
          right={
            results && (
              <span className="flex items-center gap-2 font-mono text-xs text-mute">
                {results.test_set.num_prompts} prompts × {results.test_set.seeds_per_prompt} seeds
                {results.test_set.todo && <TodoSticker />}
              </span>
            )
          }
        />
        {!results ? (
          <p className="font-mono text-sm text-mute">{error ? `Could not load results: ${error}` : 'Loading…'}</p>
        ) : (
          <div className="grid gap-12 [&>*]:min-w-0">
            <div className="grid md:grid-cols-3 gap-4">
              {results.findings.map((f, i) => (
                <FindingCard key={f.id} f={f} i={i} />
              ))}
            </div>

            <div>
              <Sub title="Results" />
              <Leaderboard rows={results.leaderboard} cols={results.columns} methods={methods} />
            </div>

            <div>
              <Sub title="Training loss" />
              <div className="grid sm:grid-cols-2 gap-4">
                {results.loss_curves.map((c) => (
                  <LossChart key={c.method} curve={c} />
                ))}
              </div>
            </div>

            <div>
              <Sub title="Side by side · same noise" />
              <QualGrid entries={results.qualitative} />
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
