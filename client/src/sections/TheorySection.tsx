import { TheoryBlock } from '../components/theory/TheoryBlock';
import { THEORY } from '../content/theory';
import { SectionHeader } from './SectionHeader';

export function TheorySection() {
  return (
    <section id="theory" className="border-b-2 border-ink scroll-mt-14" aria-labelledby="theory-title">
      <div className="mx-auto max-w-6xl px-4 md:px-8 py-12">
        <SectionHeader num="03" id="theory-title" title="Theory" />
        <div className="grid gap-8 [&>*]:min-w-0">
          {THEORY.map((b) => (
            <TheoryBlock key={b.id} block={b} />
          ))}
        </div>
      </div>
    </section>
  );
}
