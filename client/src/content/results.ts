import type { Results } from '../api/types';
import { mockClipUrl, sceneForPrompt } from '../lib/mockClips';

/**
 * EDIT ME — every number in the Findings section comes from this file.
 * Rows with `todo: true` show a TODO tag until you replace them.
 */

const QUAL_METHODS = ['teacher', 'attn_injection', 'conv_oracle', 'conv_lora'];

// TODO: replace mock:// URLs with real mp4 URLs (same seed for every method in a row)
function qual(prompt: string, seed: number) {
  const scene = sceneForPrompt(prompt);
  const videos: Record<string, string> = {};
  for (const m of QUAL_METHODS) {
    videos[m] =
      m === 'teacher'
        ? mockClipUrl({ scene, seed, variant: 'teacher', method: 'conv_lora' })
        : mockClipUrl({ scene, seed, variant: 'student', method: m });
  }
  return { prompt, seed, videos };
}

const steps = Array.from({ length: 41 }, (_, i) => i * 50);
const curve = (start: number, end: number, k: number, noise: number, seed: number) =>
  steps.map((s, i) => +(end + (start - end) * Math.exp(-s / k) + Math.sin(i * 1.7 + seed) * noise).toFixed(4));

export const RESULTS: Results = {
  test_set: { name: 'revT2V-test', num_prompts: 20, seeds_per_prompt: 3, todo: true },

  columns: [
    { key: 'fvd', label: 'FVD', better: 'lower', hint: 'Distance to the reversed teacher (I3D features).' },
    { key: 'clip', label: 'CLIP', better: 'higher', hint: 'Text–video alignment.' },
    { key: 'dynamic_degree', label: 'Dynamic', better: 'higher', hint: 'VBench dynamic degree: how much the video moves.' },
    { key: 'flow_cosine', label: 'Flow cos', better: 'higher', hint: 'Student flow vs. reversed teacher flow. +1 = perfectly reversed.' },
  ],

  // TODO: placeholders — replace with set-level numbers
  leaderboard: [
    { method: 'attn_injection', fvd: 452, clip: 29.9, dynamic_degree: 0.38, flow_cosine: 0.41, todo: true },
    { method: 'conv_oracle', fvd: 21, clip: 31.1, dynamic_degree: 0.7, flow_cosine: 0.99, todo: true, note: 'training-free' },
    { method: 'conv_lora', fvd: 246, clip: 30.6, dynamic_degree: 0.64, flow_cosine: 0.81, todo: true },
  ],

  // TODO: paste real loss logs
  loss_curves: [
    { method: 'attn_injection', label: 'attn_injection', steps, total: curve(0.4, 0.17, 520, 0.007, 4), todo: true },
    {
      method: 'conv_lora',
      label: 'conv_lora',
      steps,
      total: curve(0.42, 0.11, 420, 0.006, 1),
      components: [
        { name: 'ε-MSE', values: curve(0.3, 0.085, 380, 0.005, 2) },
        { name: 'mirror', values: curve(0.12, 0.025, 520, 0.003, 3) },
      ],
      todo: true,
    },
  ],

  qualitative: [
    qual('white smoke rising from a burning incense stick, black background', 42),
    qual('red ink dropping into a clear glass of water, close-up', 7),
    qual('orange juice being poured into a glass, close-up', 1234),
    qual('a red ball falling onto a wooden floor, static camera', 99),
    qual('a man walking away from the camera down a long corridor, static camera', 2024),
  ],

  findings: [
    {
      id: 'f1',
      method: 'attn_injection',
      verdict: 'todo',
      title: 'TODO: attn_injection result',
      body: 'TODO: one line on what captured-and-rotated attention injection achieves at up_attn1 (zero-shot and/or with a trained LoRA).',
      todo: true,
    },
    {
      id: 'f2',
      method: 'conv_oracle',
      verdict: 'positive',
      title: 'Flipping temporal convs gives an exact mirror',
      body: 'No training. Output matches the reversed teacher on smoke, ink, juice and walking.',
    },
    {
      id: 'f3',
      method: 'conv_lora',
      verdict: 'todo',
      title: 'TODO: conv_lora result',
      body: 'TODO: one line on what the trained student achieves.',
      todo: true,
    },
  ],
};
