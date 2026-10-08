import { useState } from 'react';
import type { GenerateRequest, Method } from '../../api/types';
import { DEFAULT_REQUEST, STEP_PRESETS } from '../../content/generate';
import { cx, randomSeed } from '../../lib/format';
import { RefreshIcon, SendIcon } from '../ui/Icons';

const inputCls = 'w-full bg-white border border-ink/30 px-2 py-1 font-mono text-sm focus:border-accent focus:outline-none';
const seg = (on: boolean) => cx('px-2.5 py-1 font-mono text-xs border-r border-ink last:border-r-0', on ? 'bg-ink text-white' : 'bg-white hover:bg-accent-wash');

export function Composer({ methods, disabled, onSend, prompt, setPrompt }: { methods: Method[]; disabled?: boolean; onSend: (r: GenerateRequest) => void; prompt: string; setPrompt: (p: string) => void }) {
  const [method, setMethod] = useState<string>(DEFAULT_REQUEST.student_method);
  const [seed, setSeed] = useState(42);
  const [steps, setSteps] = useState(DEFAULT_REQUEST.steps);
  const [cfg, setCfg] = useState(DEFAULT_REQUEST.cfg);
  const [frames, setFrames] = useState(DEFAULT_REQUEST.num_frames);
  const [neg, setNeg] = useState(DEFAULT_REQUEST.negative_prompt);
  const [adv, setAdv] = useState(false);

  const canSend = prompt.trim().length > 2 && !disabled;
  const send = () => {
    if (!canSend) return;
    onSend({ prompt: prompt.trim(), student_method: method, seed, steps, cfg, num_frames: frames, negative_prompt: neg });
    setPrompt('');
  };

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        send();
      }}
      className="bg-white border-2 border-ink shadow-hard"
      aria-label="Generate a reverse-time video"
    >
      <div className="flex items-stretch">
        <label htmlFor="prompt" className="sr-only">
          Prompt
        </label>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          rows={2}
          placeholder="Describe a scene…"
          className="flex-1 min-w-0 resize-none px-4 py-3 text-base leading-snug focus:outline-none bg-transparent"
        />
        <button
          type="submit"
          disabled={!canSend}
          className="shrink-0 bg-accent text-on-accent px-5 font-bold text-sm flex items-center gap-2 disabled:bg-mute hover:bg-accent-ink"
        >
          Generate <SendIcon size={16} />
        </button>
      </div>

      <div className="border-t-2 border-ink flex flex-wrap items-center gap-x-4 gap-y-2 px-3 py-2">
        <div className="flex border border-ink" role="radiogroup" aria-label="Student method">
          {(methods.length ? methods : [{ id: method, label: method, available: true } as Method]).map((m) => (
            <button
              key={m.id}
              type="button"
              role="radio"
              aria-checked={method === m.id}
              disabled={!m.available}
              onClick={() => setMethod(m.id)}
              title={m.description}
              className={cx(seg(method === m.id), method === m.id && m.kind === 'main' && 'bg-accent! text-on-accent!', 'disabled:opacity-40')}
            >
              {m.label}
            </button>
          ))}
        </div>

        <div className="flex items-center">
          <label htmlFor="seed" className="font-mono text-xs text-mute mr-1.5">
            seed
          </label>
          <input id="seed" type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value) || 0)} className="w-24 border border-ink px-2 py-1 font-mono text-xs focus:outline-none focus:border-accent" />
          <button type="button" onClick={() => setSeed(randomSeed())} className="border border-l-0 border-ink px-1.5 py-1 hover:bg-accent hover:text-on-accent" aria-label="Randomize seed">
            <RefreshIcon size={14} />
          </button>
        </div>

        <div className="flex items-center">
          <span className="font-mono text-xs text-mute mr-1.5">steps</span>
          <div className="flex border border-ink" role="group" aria-label="Step presets">
            {STEP_PRESETS.map((p) => (
              <button key={p.label} type="button" onClick={() => setSteps(p.steps)} aria-pressed={steps === p.steps} title={`${p.steps} DDIM steps`} className={seg(steps === p.steps)}>
                {p.label}
              </button>
            ))}
          </div>
        </div>

        <button type="button" aria-expanded={adv} onClick={() => setAdv(!adv)} className="ml-auto font-mono text-xs text-mute hover:text-ink">
          {adv ? '− advanced' : '+ advanced'}
        </button>
      </div>

      {adv && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 border-t border-ink/20 px-3 py-3">
          <label className="text-xs">
            <span className="font-mono text-mute block mb-1">steps</span>
            <input type="number" min={1} max={100} value={steps} onChange={(e) => setSteps(Number(e.target.value))} className={inputCls} />
          </label>
          <label className="text-xs">
            <span className="font-mono text-mute block mb-1">cfg</span>
            <input type="number" step={0.5} min={1} max={20} value={cfg} onChange={(e) => setCfg(Number(e.target.value))} className={inputCls} />
          </label>
          <label className="text-xs">
            <span className="font-mono text-mute block mb-1">frames</span>
            <select value={frames} onChange={(e) => setFrames(Number(e.target.value))} className={inputCls}>
              {[8, 16, 24].map((n) => (
                <option key={n}>{n}</option>
              ))}
            </select>
          </label>
          <label className="text-xs">
            <span className="font-mono text-mute block mb-1">negative</span>
            <input value={neg} onChange={(e) => setNeg(e.target.value)} className={inputCls} />
          </label>
        </div>
      )}
    </form>
  );
}
