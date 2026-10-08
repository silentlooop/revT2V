import katex from 'katex';
import 'katex/dist/katex.min.css';
import { motion } from 'framer-motion';
import { useMemo, type ComponentType } from 'react';
import type { DiagramKey, TheoryBlock as Block } from '../../content/theory';
import { cx } from '../../lib/format';
import { Sticker } from '../ui/Sticker';
import { AttnLoraDiagram } from './diagrams/AttnLora';
import { ConvLoraDiagram } from './diagrams/ConvLora';
import { FoundationsDiagram } from './diagrams/Foundations';
import { MetricsDiagram } from './diagrams/Metrics';
import { ConvRotationDiagram } from './diagrams/ConvRotation';

const DIAGRAMS: Record<DiagramKey, ComponentType> = {
  foundations: FoundationsDiagram,
  attnLora: AttnLoraDiagram,
  convRotation: ConvRotationDiagram,
  convLora: ConvLoraDiagram,
  metrics: MetricsDiagram,
};

export function Equation({ tex }: { tex: string }) {
  const html = useMemo(() => katex.renderToString(tex, { displayMode: true, throwOnError: false, strict: false }), [tex]);
  return <div dangerouslySetInnerHTML={{ __html: html }} />;
}

export function TheoryBlock({ block }: { block: Block }) {
  const Diagram = DIAGRAMS[block.diagram];
  const main = block.tag === 'main';
  const wide = block.diagram === 'metrics';

  return (
    <motion.article
      id={`theory-${block.id}`}
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.15 }}
      transition={{ duration: 0.45 }}
      className={cx('min-w-0 bg-white border-2 border-ink scroll-mt-20 grid', main ? 'shadow-hard-accent' : 'shadow-hard', !wide && 'lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]')}
      aria-labelledby={`theory-${block.id}-title`}
    >
      <div className={cx('min-w-0 p-5 gap-4 [&>*]:min-w-0', wide ? 'grid md:grid-cols-[1fr_1fr_1fr] items-start' : 'flex flex-col')}>
        <div className="flex items-center gap-2">
          <span className="font-mono text-xs text-accent">{block.num}</span>
          <h3 id={`theory-${block.id}-title`} className="condensed text-4xl uppercase leading-none">
            {block.title}
          </h3>
          {block.tag && (
            <Sticker tone={main ? 'accent' : 'outline'} className="ml-auto">
              {block.tag}
            </Sticker>
          )}
        </div>
        <p className="text-[15px] leading-relaxed text-ink-2">{block.summary}</p>
        <div className={cx('grid gap-1 [&>*]:min-w-0', wide ? 'md:border-l md:pl-4 border-ink/15' : 'border-t border-ink/15 pt-3')}>
          {block.equations.map((e) => (
            <div key={e.label}>
              <p className="font-mono text-[10px] text-mute">{e.label}</p>
              <Equation tex={e.tex} />
            </div>
          ))}
        </div>
      </div>
      <figure className={cx('min-w-0 p-5 bg-grid-fine border-ink flex flex-col justify-center', wide ? 'border-t-2' : 'border-t-2 lg:border-t-0 lg:border-l-2')}>
        <Diagram />
      </figure>
    </motion.article>
  );
}
