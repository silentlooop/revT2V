import { useEffect, useState } from 'react';
import { USE_MOCK } from '../api/client';
import { isMockOffline, setMockOffline } from '../api/mock';
import type { GenerateRequest, HealthStatus, Method } from '../api/types';
import { ChatThread, useThread } from '../components/chat/ChatThread';
import { Composer } from '../components/chat/Composer';
import { OfflineGallery } from '../components/chat/OfflineGallery';
import { RecycleBinIcon } from '../components/ui/Icons';
import { EXAMPLE_PROMPTS } from '../content/generate';
import { SectionHeader } from './SectionHeader';

export function GenerateSection({ health, refreshHealth, methods }: { health: HealthStatus | null; refreshHealth: () => void; methods: Method[] }) {
  const thread = useThread();
  const [prompt, setPrompt] = useState('');
  const [galleryDismissed, setGalleryDismissed] = useState(false);
  const offline = health != null && !health.online;

  useEffect(() => {
    thread.setItems((xs) => xs.map((x) => (x.status === 'submitting' ? { ...x, status: 'error', error: 'Interrupted by reload.' } : x)));
  }, []);

  useEffect(() => {
    if (!offline) setGalleryDismissed(false);
  }, [offline]);

  const send = (req: GenerateRequest) => thread.submit(req);
  const empty = thread.items.length === 0;

  return (
    <section id="generate" className="border-b-2 border-ink scroll-mt-14" aria-labelledby="generate-title">
      <div className="mx-auto max-w-6xl px-4 md:px-8 py-12">
        <SectionHeader
          num="01"
          id="generate-title"
          title="Generate"
          right={
            <div className="flex items-center gap-3">
              {USE_MOCK && (
                <button
                  type="button"
                  onClick={() => {
                    setMockOffline(!isMockOffline());
                    refreshHealth();
                  }}
                  className="font-mono text-[11px] text-mute hover:text-ink"
                  title="Mock only: simulate the backend going offline"
                >
                  sim {offline ? 'online' : 'offline'}
                </button>
              )}
              {!empty && (
                <button
                  type="button"
                  onClick={() => confirm('Clear the chat?') && thread.clear()}
                  className="flex items-center gap-1.5 border-2 border-ink bg-white px-2 py-1 text-xs font-semibold hover:bg-accent hover:text-on-accent"
                >
                  <RecycleBinIcon size={16} /> New chat
                </button>
              )}
            </div>
          }
        />

        <div className="grid gap-6">
          {offline && !galleryDismissed && <OfflineGallery onRetry={refreshHealth} onClose={() => setGalleryDismissed(true)} />}

          {!empty && <ChatThread thread={thread} />}

          <div className={empty ? '' : 'sticky bottom-3 z-30'}>
            {empty && (
              <div className="flex flex-wrap gap-1.5 mb-3" aria-label="Example prompts">
                {EXAMPLE_PROMPTS.map((p) => (
                  <button key={p} type="button" onClick={() => setPrompt(p)} className="border border-ink bg-white px-2.5 py-1 text-xs hover:bg-accent hover:text-on-accent hover:border-accent">
                    {p.split(',')[0]}
                  </button>
                ))}
              </div>
            )}
            <Composer methods={methods} onSend={send} disabled={offline} prompt={prompt} setPrompt={setPrompt} />
          </div>
        </div>
      </div>
    </section>
  );
}
