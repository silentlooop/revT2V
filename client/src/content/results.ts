import type { Results } from '../api/types';
import { mockClipUrl, sceneForPrompt } from '../lib/mockClips';

/**
 * EDIT ME — every number in the Findings section comes from this file.
 * Rows with `todo: true` show a TODO tag until you replace them.
 */

const QUAL_METHODS = ['teacher', 'attn_injection', 'conv_mirror', 'conv_lora', 'attn_rotation'];

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

// Real step=N loss=X lines from logs/*.log (same format scripts/make_report_assets.py
// parses), log_every_steps=10. attn_rotation's log only covers a resumed
// segment (step 800 onward was already trained before this log started).
const CONV_LORA_STEPS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990];
const CONV_LORA_LOSS = [0.1075, 0.1514, 0.068, 0.0475, 0.0845, 0.0896, 0.0761, 0.0923, 0.1118, 0.0719, 0.0639, 0.0428, 0.0428, 0.1382, 0.0495, 0.0527, 0.0974, 0.0456, 0.0485, 0.098, 0.0216, 0.1335, 0.037, 0.1657, 0.0828, 0.0658, 0.1162, 0.0612, 0.049, 0.0989, 0.0821, 0.0791, 0.0959, 0.0873, 0.0872, 0.0645, 0.0648, 0.0461, 0.1303, 0.0603, 0.0562, 0.1339, 0.047, 0.0582, 0.0775, 0.035, 0.0883, 0.0311, 0.1563, 0.1013, 0.054, 0.0713, 0.0671, 0.1179, 0.0887, 0.1156, 0.113, 0.1069, 0.0713, 0.0851, 0.0477, 0.0727, 0.1061, 0.0633, 0.0679, 0.0667, 0.048, 0.0491, 0.0762, 0.0299, 0.1153, 0.0308, 0.1592, 0.0827, 0.0737, 0.1204, 0.0782, 0.0491, 0.1483, 0.0988, 0.0832, 0.1057, 0.1254, 0.0672, 0.0895, 0.0499, 0.0502, 0.1117, 0.0539, 0.0683, 0.1289, 0.0565, 0.0609, 0.0658, 0.0301, 0.1462, 0.0464, 0.1212, 0.1006];

const ATTN_INJECTION_STEPS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 790, 800, 810, 820, 830, 840, 850, 860, 870, 880, 890, 900, 910, 920, 930, 940, 950, 960, 970, 980, 990];
const ATTN_INJECTION_LOSS = [0.1489, 0.1003, 0.0668, 0.0637, 0.0757, 0.085, 0.0934, 0.0941, 0.117, 0.0693, 0.0511, 0.0414, 0.0458, 0.1218, 0.0445, 0.063, 0.1092, 0.058, 0.0468, 0.0584, 0.0225, 0.1374, 0.0248, 0.0957, 0.0691, 0.0487, 0.1499, 0.0629, 0.0457, 0.1181, 0.0726, 0.0894, 0.1122, 0.1147, 0.0956, 0.0698, 0.0531, 0.0458, 0.1295, 0.048, 0.0564, 0.0705, 0.0565, 0.057, 0.0649, 0.0207, 0.0948, 0.0384, 0.1175, 0.0796, 0.044, 0.1171, 0.0539, 0.0436, 0.112, 0.0903, 0.0757, 0.1336, 0.1101, 0.063, 0.0814, 0.0718, 0.0437, 0.0867, 0.0528, 0.0595, 0.1331, 0.0787, 0.0679, 0.0621, 0.0311, 0.1232, 0.0276, 0.1624, 0.0681, 0.0527, 0.0862, 0.0679, 0.0617, 0.0862, 0.0834, 0.1059, 0.0956, 0.1415, 0.0778, 0.0718, 0.066, 0.0794, 0.1174, 0.0501, 0.0607, 0.1427, 0.0562, 0.0709, 0.0942, 0.029, 0.1423, 0.0336, 0.1016, 0.1248];

const ATTN_ROTATION_STEPS = [820, 840, 860, 880, 900, 920, 940, 960, 980];
const ATTN_ROTATION_LOSS = [0.1504, 0.106, 0.0972, 0.1042, 0.1055, 0.1123, 0.1273, 0.1358, 0.059];

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
    { method: 'conv_mirror', fvd: 21, clip: 31.1, dynamic_degree: 0.7, flow_cosine: 0.99, todo: true, note: 'training-free' },
    { method: 'conv_lora', fvd: 246, clip: 30.6, dynamic_degree: 0.64, flow_cosine: 0.81, todo: true },
    { method: 'attn_rotation', fvd: 410, clip: 30.1, dynamic_degree: 0.42, flow_cosine: 0.37, todo: true },
  ],

  // Real training runs from logs/*.log -- step=N loss=X lines, see
  // scripts/make_report_assets.py's loss_curves() for the same parse.
  loss_curves: [
    { method: 'attn_injection', label: 'attn_injection', steps: ATTN_INJECTION_STEPS, total: ATTN_INJECTION_LOSS },
    { method: 'conv_lora', label: 'conv_lora', steps: CONV_LORA_STEPS, total: CONV_LORA_LOSS },
    { method: 'attn_rotation', label: 'attn_rotation (resumed, steps 820–980)', steps: ATTN_ROTATION_STEPS, total: ATTN_ROTATION_LOSS },
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
      method: 'conv_mirror',
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
    {
      id: 'f4',
      method: 'attn_rotation',
      verdict: 'todo',
      title: 'TODO: attn_rotation result',
      body: 'TODO: one line on what self-rotation + attention-only LoRA achieves without touching the convs.',
      todo: true,
    },
  ],
};
