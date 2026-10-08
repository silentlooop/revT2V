import type { GenerateRequest } from '../api/types';

export const EXAMPLE_PROMPTS = [
  'white smoke rising from a burning incense stick, black background',
  'red ink dropping into a clear glass of water, close-up',
  'orange juice being poured into a glass, close-up',
  'a red ball falling onto a wooden floor, static camera',
  'a man walking away from the camera down a long corridor, static camera',
];

export const DEFAULT_REQUEST: Omit<GenerateRequest, 'prompt' | 'seed'> = {
  student_method: 'conv_lora',
  steps: 25,
  cfg: 9.0,
  num_frames: 16,
  negative_prompt: 'watermark, text',
};

/** small / med / XXL step presets (the moodboard size selector) */
export const STEP_PRESETS = [
  { label: 'S', steps: 15 },
  { label: 'M', steps: 25 },
  { label: 'XXL', steps: 50 },
] as const;

/** Precomputed examples shown when the backend is offline. */
export const GALLERY: GenerateRequest[] = EXAMPLE_PROMPTS.map((prompt, i) => ({
  ...DEFAULT_REQUEST,
  prompt,
  seed: [42, 7, 1234, 99, 2024][i],
}));

export const METRIC_HELP = {
  fvd: { title: 'FVD', better: 'lower' as const, plain: 'Distance between student and reversed teacher. Lower is closer.' },
  clip: { title: 'CLIP', better: 'higher' as const, plain: 'How well frames match the prompt.' },
  dynamic_degree: { title: 'Dynamic', better: 'higher' as const, plain: 'How much the video moves (VBench).' },
  optical_flow: { title: 'Flow', better: 'higher' as const, plain: 'Motion direction vs. reversed teacher. +1 = correctly reversed.' },
};
