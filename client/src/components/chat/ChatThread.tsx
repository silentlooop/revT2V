import { useCallback, useEffect, useRef } from 'react';
import { api } from '../../api/client';
import type { GenerateRequest, PairResult } from '../../api/types';
import { useJob, useLocalStorage } from '../../hooks/useApi';
import { PendingCard } from './PendingCard';
import { ResultCard } from './ResultCard';
import { UserBubble } from './UserBubble';

export type ThreadItem = {
  id: string;
  req: GenerateRequest;
  jobId: string | null;
  status: 'submitting' | 'pending' | 'done' | 'error';
  result?: PairResult;
  error?: string;
};

export const THREAD_KEY = 'revt2v.thread.v1';

export function useThread() {
  const [items, setItems] = useLocalStorage<ThreadItem[]>(THREAD_KEY, []);
  const patch = useCallback((id: string, p: Partial<ThreadItem>) => setItems((xs) => xs.map((x) => (x.id === id ? { ...x, ...p } : x))), [setItems]);

  const submit = useCallback(
    async (req: GenerateRequest, existingId?: string) => {
      const id = existingId ?? `msg_${Date.now().toString(36)}`;
      if (existingId) patch(id, { status: 'submitting', error: undefined, jobId: null });
      else setItems((xs) => [...xs, { id, req, jobId: null, status: 'submitting' }]);
      try {
        const { job_id } = await api.generate(req);
        patch(id, { jobId: job_id, status: 'pending' });
      } catch (e) {
        patch(id, { status: 'error', error: (e as Error).message });
      }
    },
    [patch, setItems],
  );

  return { items, setItems, patch, submit, clear: () => setItems([]) };
}

function JobEntry({ item, patch, retry, remove }: { item: ThreadItem; patch: (id: string, p: Partial<ThreadItem>) => void; retry: () => void; remove: () => void }) {
  const job = useJob(
    item.status === 'pending' ? item.jobId : null,
    (result) => patch(item.id, { status: 'done', result }),
    (error) => patch(item.id, { status: 'error', error }),
  );
  if (item.status === 'done' && item.result) return <ResultCard result={item.result} />;
  return <PendingCard req={item.req} job={job} error={item.status === 'error' ? item.error : undefined} onRetry={retry} onDismiss={remove} />;
}

export function ChatThread({ thread }: { thread: ReturnType<typeof useThread> }) {
  const { items, patch, submit, setItems } = thread;
  const endRef = useRef<HTMLDivElement>(null);
  const count = useRef(items.length);
  useEffect(() => {
    if (items.length > count.current) endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
    count.current = items.length;
  }, [items.length]);

  return (
    <ol className="grid gap-6" aria-label="Generation thread">
      {items.map((it) => (
        <li key={it.id} className="grid gap-4">
          <UserBubble req={it.req} />
          <div className="flex justify-start">
            <JobEntry item={it} patch={patch} retry={() => submit(it.req, it.id)} remove={() => setItems((xs) => xs.filter((x) => x.id !== it.id))} />
          </div>
        </li>
      ))}
      <div ref={endRef} />
    </ol>
  );
}
