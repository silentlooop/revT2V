import { useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import type { HealthStatus, Job, Method, PairResult, Results } from '../api/types';

export function useLocalStorage<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key);
      return raw ? (JSON.parse(raw) as T) : initial;
    } catch {
      return initial;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* storage blocked — state stays in memory */
    }
  }, [key, value]);
  return [value, setValue] as const;
}

export function useHealth(intervalMs = 8000) {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    let alive = true;
    const run = () =>
      api
        .getHealth()
        .then((h) => {
          if (!alive) return;
          setHealth(h);
          setError(null);
        })
        .catch((e: Error) => {
          if (!alive) return;
          setHealth({ online: false, gpu: null, model_loaded: false, queue_length: 0, version: '?' });
          setError(e.message);
        });
    run();
    const id = setInterval(run, intervalMs);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [intervalMs, nonce]);
  return { health, error, refresh: () => setNonce((n) => n + 1) };
}

export function useMethods() {
  const [methods, setMethods] = useState<Method[]>([]);
  useEffect(() => {
    api.getMethods().then(setMethods).catch(() => setMethods([]));
  }, []);
  return methods;
}

export function useResults() {
  const [results, setResults] = useState<Results | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.getResults().then(setResults).catch((e: Error) => setError(e.message));
  }, []);
  return { results, error };
}

/** Polls a job until done, then fetches the result. Safe to remount (resumes by id). */
export function useJob(jobId: string | null, onDone: (r: PairResult) => void, onError: (msg: string) => void) {
  const [job, setJob] = useState<Job | null>(null);
  const cb = useRef({ onDone, onError });
  cb.current = { onDone, onError };

  useEffect(() => {
    if (!jobId) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    let failures = 0;
    const poll = async () => {
      try {
        const j = await api.getJob(jobId);
        if (!alive) return;
        failures = 0;
        setJob(j);
        if (j.status === 'done') {
          const r = await api.getResult(jobId);
          if (alive) cb.current.onDone(r);
          return;
        }
        if (j.status === 'error') {
          cb.current.onError(j.error ?? 'generation failed');
          return;
        }
      } catch (e) {
        if (!alive) return;
        if (++failures > 5) {
          cb.current.onError((e as Error).message);
          return;
        }
      }
      timer = setTimeout(poll, 700);
    };
    poll();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [jobId]);

  return job;
}
