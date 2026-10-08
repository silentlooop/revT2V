# revT2V · frontend

Interactive frontend for **revT2V: Distilling Video Diffusion Models for Reverse-Time Video Generation** (IE 643 Deep Learning, IIT Bombay).

A frozen ModelScope T2V 1.7B **teacher** generates normal forward-time video. A **student** generates the same scenes with time running backwards. The page has three sections, in order: **Generate** (prompt teacher + student side by side), **Findings** (results), **Theory** (how each method works).

Methods:

| method | idea |
|---|---|
| `attn_injection` | rotate the teacher's temporal attention scores 180° + LoRA on value/output |
| `conv_oracle` | flip temporal conv kernels in time, no training (exact mirror) |
| `conv_lora` | LoRA on temporal convs + attention, ε-MSE + mirror loss (main) |

Stack: React 19 + Vite + TypeScript, Tailwind CSS v4 (custom tokens only), Framer Motion, KaTeX. No UI kits.

---

## Setup

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # type-check + production build into dist/
npm run preview    # serve the production build
```

Requires Node 20+.

## Mock → real API

All data goes through `src/api/client.ts`. By default it uses an in-browser mock (`src/api/mock.ts`). The mock simulates the queue, step-by-step teacher/student progress, refresh-safe jobs, and procedurally rendered video clips, so every feature works without a GPU.

To use the real FastAPI backend:

```bash
cp .env.example .env.local
# then edit:
VITE_USE_MOCK=false
VITE_API_BASE_URL=https://your-tunnel.trycloudflare.com
```

Restart `npm run dev`. Nothing else changes: `client.ts` is the only file that knows which backend is in use. The backend has to implement **`src/api/CONTRACT.md`**, which documents every endpoint, request/response shape and metric definition. The TypeScript types are in `src/api/types.ts`.

Mock-only extras:
- **sim offline** button (Generate section header) toggles the backend into "offline" to show the Win98 dialog and the example gallery. You can also open the app with `?offline`.
- Mock "videos" are `mock://` URLs rendered to canvas. Download gives a PNG contact sheet instead of an MP4.

## Editing content (no component changes needed)

| what | file |
|---|---|
| Theory text, equations (KaTeX), tags | `src/content/theory.ts` |
| Findings, results table, loss curves, side-by-side grid | `src/content/results.ts` |
| Hugging Face + GitHub links and references at the bottom of the page | `src/content/links.ts` |
| Example prompts, defaults, step presets, metric tooltips, offline gallery | `src/content/generate.ts` |
| Colours, fonts, shadows, textures | `src/styles/index.css` (`@theme` block) |

**Results placeholders.** Every value in `results.ts` that is a placeholder has `todo: true`, which renders a red **TODO** sticker in the UI. Replace the numbers, then delete the `todo` flags. Use `null` for a metric you don't have yet. The qualitative grid takes a video URL per method per prompt. Swap the `mock://` URLs for real MP4 URLs from the same seed.

**Theory.** Each block has a one-sentence `summary`, `equations` (`{label, tex}`, backslashes escaped as `\\`) and a `diagram` key. Don't rename diagram keys; they map to the SVG components in `src/components/theory/diagrams/`.

## Project structure

```
src/
  api/          types.ts · client.ts (mock/real switch) · mock.ts · CONTRACT.md
  content/      theory.ts · results.ts · generate.ts   ← edit these
  hooks/        useClock (synced playback) · useApi (health, methods, job polling, localStorage)
  lib/          mockClips (procedural clips + flow) · rng · format
  components/
    ui/         Sticker, SelectionBox, Win98Window, Transport (+Toggle), InfoTip, Icons
    video/      ClipSurface (+FlowOverlay, FrameImage), SyncedVideoPair, FrameStrip, MosaicProgress
    metrics/    MetricSticker (MetricsRow)
    chat/       ChatThread, UserBubble, PendingCard, ResultCard, Composer,
                StatusBadge, OfflineGallery
    theory/     TheoryBlock (+Equation) · diagrams/ (kit + one SVG per method)
    results/    Leaderboard, Charts (MetricBarChart, LossChart), QualGrid, Findings (FindingCard)
  sections/     Hero, Nav, GenerateSection, TheorySection, ResultsSection, SectionHeader
  styles/       index.css (Tailwind v4 theme + texture utilities)
references/     the moodboards the visual design is based on
```

---

## Interface methodology

*(Written to be adapted for the course report.)*

### Goals

The interface has three jobs: (1) let anyone **generate** a teacher/student pair from a text prompt and judge the reversal by eye and by metric; (2) **explain** how each method works, from diffusion basics to our losses; (3) **present** the experimental findings honestly, including negative results. A non-functional requirement drove most design decisions: generation takes minutes on a free T4 GPU, so the UI must stay informative while waiting and must survive refreshes and backend outages.

### Architecture

```
 React UI ──▶ src/api/client.ts ──▶ USE_MOCK ? mock.ts (in-browser) : FastAPI (Colab GPU via tunnel)
    ▲                                         │
    └──────── typed data (src/api/types.ts) ◀──┘
 src/content/*.ts ──▶ static, editable text + results
```

- **Single API seam.** Components never call `fetch`. Everything goes through six functions (`getHealth`, `getMethods`, `generate`, `getJob`, `getResult`, `getResults`) with shared TypeScript types. A written contract (`CONTRACT.md`) let the frontend be built and tested before the backend existed, and keeps the two halves decoupled.
- **Realistic mock.** The mock reproduces the backend's *temporal behaviour*, not just its shapes: queue position, teacher then student phases with per-step progress, and jobs whose progress is derived from their start time (stored in `localStorage`), so a refresh resumes correctly. Mock videos are procedural canvas scenes (smoke, ink, pouring, a bouncing ball, a walking figure), each with an analytic velocity field. That lets the optical-flow overlay and the flow metrics (magnitude, direction cosine against the reversed teacher, FROZEN flag) be computed from actual motion rather than invented. Each method maps to its expected behaviour (`attn_injection` → damped motion with a direction wobble, `conv_oracle` → exact mirror, `conv_lora` → near-exact reversal).
- **Content as data.** All theory text and every result number lives in `src/content/`. Placeholder results are flagged (`todo: true`) and rendered with a visible TODO sticker, so unfinished numbers cannot be mistaken for real ones.

### Data flow of a generation

1. The composer builds a `GenerateRequest` (prompt, method, seed, steps, CFG, frames, negative prompt) and `POST /generate` returns a `job_id`.
2. The thread entry is saved to `localStorage` immediately with its `job_id`.
3. `useJob` polls `GET /jobs/{id}` (~0.7 s). The pending card shows the active model, step x/N, elapsed time and queue position. The preview frame is a pixel mosaic that resolves block by block as steps complete.
4. On `done`, `GET /jobs/{id}/result` returns teacher/student `VideoOut`s plus `PairMetrics`. The result card replaces the pending card and is persisted.
5. If `/health` fails, the UI switches to an offline state (Win98 dialog) and offers precomputed examples rendered in the same card component.

### Fair-comparison design

- Teacher and student always share **seed and initial noise**. The UI states this in the pending card, the result card and the qualitative grid.
- **One master clock** (`useClock`) drives all videos in a card or grid. Every player renders the clock's current frame (canvas draw for mock clips, frame-accurate `currentTime` seek for MP4s), so teacher, student and reversed teacher are frame-locked rather than "roughly in sync". A single transport bar (play, frame scrubber, loop, speed) controls them all.
- **All four metrics are always visible** on each result card, each with a ↑/↓ hint, a plain-English tooltip, a teacher-vs-student comparison and the backend's optional caveat note (e.g. single-pair FVD is approximate; set-level FVD is in Findings).

### Visual design

The look borrows from a collage / UI-parody moodboard (`references/`), deliberately toned down so the page reads as a clean research demo: graph-paper background, solid white panels with hard (blur-free) shadows, an electric-blue accent, and minimal text. A few parodies remain where they map to a real function:

| element | function |
|---|---|
| Figma-style selection box | marks the student output |
| S · M · XXL size picker | DDIM step presets (15 / 25 / 50) |
| Windows 98 dialog + folders | offline mode and example gallery |
| recycle bin | new chat |
| pixel mosaic | diffusion progress |

Explanations live in tooltips rather than on the page; cards show numbers first.

**Tokens.** All colours, fonts and shadows are defined once in the Tailwind v4 `@theme` block. Tailwind's default palette and shadows are explicitly reset, so off-palette values can't sneak in. Textures (grid, grain, halftone, stripes) are small CSS utilities.

**Typography.** Archivo (condensed 900) for headlines, Silkscreen for small tags, Space Mono for data, and Inter Tight for UI text.

### Usability and accessibility

The collage is decoration. Content stays legible:

- Videos, numbers and controls sit on solid white panels with high-contrast ink. The accent is never the only way information is conveyed (metric verdicts are textual: reversed, partial, frozen).
- All controls are keyboard reachable with a visible focus ring. The frame scrubber is a native range input. Tooltips open on focus and close on Escape. Toggles use `role="switch"`. Tabs and progress bars carry ARIA roles. Decorative SVG is `aria-hidden` and diagrams have text labels.
- `prefers-reduced-motion` disables animation.
- Responsive: the collage collapses to a single column on phones, wide tables scroll inside their own container, and the page has no horizontal overflow at 390 px.
- Theory diagrams are custom SVG, animated gently on scroll with Framer Motion (a rotating attention matrix, kernel taps swapping on flip). Equations are rendered with KaTeX.

### Limitations of the interface

- Mock clips are abstract stand-ins; their metrics are self-consistent but not real model outputs.
- The pending card's mosaic preview is schematic: the backend doesn't stream intermediate frames.
- Findings numbers are placeholders until replaced in `results.ts`.
