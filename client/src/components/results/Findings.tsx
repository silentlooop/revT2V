import { motion } from 'framer-motion';
import type { Finding } from '../../api/types';
import { cx } from '../../lib/format';
import { Sticker, TodoSticker } from '../ui/Sticker';

const MARK = { negative: '✕', positive: '✓', mixed: '~', todo: '…' } as const;

export function FindingCard({ f, i }: { f: Finding; i: number }) {
  return (
    <motion.article
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.3 }}
      transition={{ duration: 0.4, delay: i * 0.08 }}
      className={cx('bg-white border-2 border-ink p-4 flex flex-col gap-3', f.verdict === 'positive' ? 'shadow-hard-accent' : 'shadow-hard')}
    >
      <div className="flex items-center gap-2">
        <Sticker tone={f.method === 'conv_lora' ? 'accent' : 'ink'}>{f.method}</Sticker>
        {f.todo && <TodoSticker />}
        <span className={cx('ml-auto grid place-items-center size-7 font-bold', f.verdict === 'positive' ? 'bg-accent text-on-accent' : 'border-2 border-ink')} aria-label={f.verdict}>
          {MARK[f.verdict]}
        </span>
      </div>
      <h3 className="condensed text-2xl uppercase leading-[0.95]">{f.title}</h3>
      <p className="text-sm text-ink-2 leading-snug">{f.body}</p>
    </motion.article>
  );
}
