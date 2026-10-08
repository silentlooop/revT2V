import { useClock } from '../hooks/useClock';
import { mockClipUrl } from '../lib/mockClips';
import { SelectionBox } from '../components/ui/SelectionBox';
import { ClipSurface } from '../components/video/ClipSurface';

const T_URL = mockClipUrl({ scene: 'smoke', seed: 42, variant: 'teacher', method: 'conv_lora' });
const S_URL = mockClipUrl({ scene: 'smoke', seed: 42, variant: 'student', method: 'conv_lora' });

export function Hero() {
  const clock = useClock(16, 8);
  return (
    <header className="border-b-2 border-ink">
      <div className="mx-auto max-w-6xl px-4 md:px-8 py-12 md:py-16 grid md:grid-cols-[1.1fr_1fr] gap-10 items-center">
        <div>
          <p className="font-mono text-xs text-accent mb-4">IE 643 · IIT Bombay</p>
          <h1 className="condensed text-[clamp(4rem,11vw,9rem)] uppercase leading-[0.82]">
            rev<span className="text-accent">T2V</span>
          </h1>
          <p className="mt-5 text-lg max-w-md leading-snug">Text-to-video that plays backward in time.</p>
          <div className="mt-7 flex gap-2">
            <a href="#generate" className="bg-accent text-on-accent border-2 border-ink px-5 py-2.5 font-bold shadow-hard hover:-translate-y-0.5 transition-transform">
              Generate
            </a>
            <a href="#findings" className="bg-white border-2 border-ink px-5 py-2.5 font-semibold hover:bg-accent-wash">
              Findings
            </a>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-5">
          <figure>
            <div className="border-2 border-ink shadow-hard">
              <ClipSurface url={T_URL} frame={clock.frame} label="Teacher: smoke rising" />
            </div>
            <figcaption className="mt-2 font-mono text-xs">teacher →</figcaption>
          </figure>
          <figure className="mt-10">
            <SelectionBox>
              <ClipSurface url={S_URL} frame={clock.frame} label="Student: smoke sinking back" />
            </SelectionBox>
            <figcaption className="mt-2 font-mono text-xs text-accent">← student</figcaption>
          </figure>
        </div>
      </div>
    </header>
  );
}
