import { MotionConfig } from 'framer-motion';
import { useHealth, useMethods } from './hooks/useApi';
import { GenerateSection } from './sections/GenerateSection';
import { Hero } from './sections/Hero';
import { Nav } from './sections/Nav';
import { ResultsSection } from './sections/ResultsSection';
import { TheorySection } from './sections/TheorySection';
import { LinksSection } from './sections/LinksSection';

export default function App() {
  const { health, refresh } = useHealth();
  const methods = useMethods();

  return (
    <MotionConfig reducedMotion="user">
      <div id="top">
        <a href="#generate" className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-[200] bg-accent text-on-accent px-3 py-2">
          Skip to generator
        </a>
        <Nav health={health} />
        <Hero />
        <main>
          <GenerateSection health={health} refreshHealth={refresh} methods={methods} />
          <ResultsSection methods={methods} />
          <TheorySection />
          <LinksSection />
        </main>
        <footer className="bg-ink text-white">
          <div className="mx-auto max-w-6xl px-4 md:px-8 py-5 flex flex-wrap gap-x-6 gap-y-1 items-center font-mono text-xs">
            <span className="font-pixel text-sm">
              <span className="bg-accent text-on-accent px-1">rev</span>T2V
            </span>
            <span className="opacity-70">IE 643 · IIT Bombay</span>
            <span className="ml-auto opacity-70">
              © {new Date().getFullYear()}{' '}
              <a href="https://github.com/silentlooop" target="_blank" rel="noopener noreferrer" className="underline underline-offset-2 hover:opacity-100 hover:text-accent-soft">
                silentlooop
              </a>
            </span>
          </div>
        </footer>
      </div>
    </MotionConfig>
  );
}
