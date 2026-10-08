export const fmtTime = (s: number) => {
  const m = Math.floor(s / 60);
  const ss = Math.floor(s % 60);
  return `${m}:${ss.toString().padStart(2, '0')}`;
};

export const fmtNum = (v: number | null | undefined, digits = 2) =>
  v == null || Number.isNaN(v) ? '—' : Number.isInteger(v) && digits === 0 ? String(v) : v.toFixed(digits);

export const randomSeed = () => Math.floor(Math.random() * 2 ** 31);

export const cx = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(' ');
