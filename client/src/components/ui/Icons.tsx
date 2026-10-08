import type { SVGProps } from 'react';

type P = SVGProps<SVGSVGElement> & { size?: number };

export const FolderIcon = ({ size = 40, ...p }: P) => (
  <svg width={size} height={size * 0.8} viewBox="0 0 50 40" aria-hidden {...p}>
    <path d="M2 6h17l4 4h25v28H2z" fill="var(--color-accent-soft)" stroke="var(--color-ink)" strokeWidth="1.5" />
    <path d="M2 13h46v25H2z" fill="color-mix(in srgb, var(--color-accent-soft) 70%, white)" stroke="var(--color-ink)" strokeWidth="1.5" />
  </svg>
);

export const RecycleBinIcon = ({ size = 34, ...p }: P) => (
  <svg width={size} height={size * 1.15} viewBox="0 0 40 46" aria-hidden {...p}>
    <ellipse cx="20" cy="7" rx="16" ry="4" fill="#e8e8e8" stroke="var(--color-ink)" strokeWidth="1.5" />
    <path d="M4 7l3 35c.3 2 6 3.5 13 3.5S32.7 44 33 42l3-35" fill="color-mix(in srgb, var(--color-accent-soft) 40%, white)" stroke="var(--color-ink)" strokeWidth="1.5" />
    {[10, 15, 20, 25, 30].map((x) => (
      <line key={x} x1={x} y1="12" x2={x + (x - 20) * 0.08} y2="41" stroke="var(--color-ink)" strokeOpacity=".35" />
    ))}
    <path d="M15 24l5-6 5 6m-8 4l3 4 3-4" fill="none" stroke="var(--color-accent)" strokeWidth="2" />
  </svg>
);

export const HourglassIcon = ({ size = 18, ...p }: P) => (
  <svg width={size} height={size} viewBox="0 0 20 20" aria-hidden {...p}>
    <path d="M4 2h12M4 18h12M5 2c0 5 10 5 10 8S5 13 5 18M15 2c0 5-10 5-10 8s10 3 10 8" fill="none" stroke="currentColor" strokeWidth="2" />
    <path d="M7 16h6l-3-3z" fill="currentColor" />
  </svg>
);

export const CursorIcon = ({ size = 22, ...p }: P) => (
  <svg width={size} height={size} viewBox="0 0 22 22" aria-hidden {...p}>
    <path d="M2 2l7 18 2.5-7.5L19 10z" fill="var(--color-white)" stroke="var(--color-ink)" strokeWidth="1.5" strokeLinejoin="round" />
  </svg>
);

export const MoveIcon = ({ size = 24, ...p }: P) => (
  <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden {...p}>
    <path d="M12 2v20M2 12h20M12 2l-3 3m3-3l3 3M12 22l-3-3m3 3l3-3M2 12l3-3m-3 3l3 3m17-3l-3-3m3 3l-3 3" fill="none" stroke="currentColor" strokeWidth="1.8" />
  </svg>
);

export const RefreshIcon = ({ size = 20, ...p }: P) => (
  <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden {...p}>
    <path d="M20 12a8 8 0 1 1-2.4-5.7M20 4v5h-5" fill="none" stroke="currentColor" strokeWidth="2.2" />
  </svg>
);

export const HeadphonesIcon = ({ size = 20, ...p }: P) => (
  <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden {...p}>
    <path d="M4 15v-3a8 8 0 0 1 16 0v3" fill="none" stroke="currentColor" strokeWidth="2.2" />
    <rect x="3" y="14" width="5" height="7" rx="1.5" fill="currentColor" />
    <rect x="16" y="14" width="5" height="7" rx="1.5" fill="currentColor" />
  </svg>
);

export const DownloadIcon = ({ size = 16, ...p }: P) => (
  <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden {...p}>
    <path d="M12 3v12m-5-5l5 5 5-5M4 20h16" fill="none" stroke="currentColor" strokeWidth="2.4" />
  </svg>
);

export const StarIcon = ({ size = 18, filled = true, ...p }: P & { filled?: boolean }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden {...p}>
    <path
      d="M12 2l2.9 6.6 7.1.6-5.4 4.7 1.6 7L12 17.3 5.8 21l1.6-7L2 9.2l7.1-.6z"
      fill={filled ? 'currentColor' : 'none'}
      stroke="currentColor"
      strokeWidth={filled ? 0 : 1.3}
      strokeLinejoin="round"
    />
  </svg>
);

export const SendIcon = ({ size = 20, ...p }: P) => (
  <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden {...p}>
    <path d="M3 12h15m-6-7l7 7-7 7" fill="none" stroke="currentColor" strokeWidth="2.6" />
  </svg>
);
