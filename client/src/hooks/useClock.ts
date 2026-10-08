import { useCallback, useEffect, useRef, useState } from 'react';

export interface Clock {
  frame: number;
  frames: number;
  playing: boolean;
  loop: boolean;
  speed: number;
  play(): void;
  pause(): void;
  toggle(): void;
  seek(f: number): void;
  step(d: number): void;
  setLoop(v: boolean): void;
  setSpeed(v: number): void;
}

/** One master clock → every video bound to it stays frame-locked. */
export function useClock(frames: number, fps = 8, autoplay = true): Clock {
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(autoplay);
  const [loop, setLoop] = useState(true);
  const [speed, setSpeed] = useState(1);
  const t = useRef(0);
  const last = useRef<number | null>(null);

  useEffect(() => {
    if (!playing) {
      last.current = null;
      return;
    }
    let raf = 0;
    const tick = (now: number) => {
      if (last.current != null) {
        t.current += ((now - last.current) / 1000) * fps * speed;
        if (t.current >= frames) {
          if (loop) t.current %= frames;
          else {
            t.current = frames - 1;
            setFrame(frames - 1);
            setPlaying(false);
            return;
          }
        }
        const f = Math.floor(t.current);
        setFrame((prev) => (prev === f ? prev : f));
      }
      last.current = now;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, fps, speed, loop, frames]);

  const seek = useCallback(
    (f: number) => {
      const n = ((Math.round(f) % frames) + frames) % frames;
      t.current = n;
      setFrame(n);
    },
    [frames],
  );

  return {
    frame,
    frames,
    playing,
    loop,
    speed,
    play: () => {
      if (!loop && t.current >= frames - 1) seek(0);
      setPlaying(true);
    },
    pause: () => setPlaying(false),
    toggle: () => setPlaying((p) => !p),
    seek,
    step: (d: number) => {
      setPlaying(false);
      seek(Math.floor(t.current) + d);
    },
    setLoop,
    setSpeed,
  };
}
