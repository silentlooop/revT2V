/**
 * EDIT ME — all wording in the Theory section.
 * `tex` is rendered with KaTeX (escape backslashes). Don't rename `diagram` keys.
 */

export type DiagramKey = 'foundations' | 'attnLora' | 'convRotation' | 'convLora' | 'metrics';

export interface TheoryBlock {
  id: string;
  num: string;
  title: string;
  tag?: string;
  summary: string;
  equations: { label: string; tex: string }[];
  diagram: DiagramKey;
}

export const THEORY: TheoryBlock[] = [
  {
    id: 'foundations',
    num: '00',
    title: 'Foundations',
    summary:
      'A diffusion model turns noise into video over many denoising steps. Only the temporal layers of the U-Net know frame order, so that is where we intervene.',
    equations: [
      { label: 'denoising objective', tex: '\\mathcal L = \\big\\| \\varepsilon - \\varepsilon_\\theta(z_t, t, c)\\big\\|^2' },
      { label: 'time reversal', tex: '(\\mathcal R x)_f = x_{F-1-f}' },
    ],
    diagram: 'foundations',
  },
  {
    id: 'attn_injection',
    num: '01',
    title: 'attn_injection',
    summary: 'Capture the base U-Net’s own attention on the time-flipped latent (stock weights, no LoRA), rotate the map 180°, and apply it to the real trajectory’s own value/output projections — LoRA sits only on to_v and to_out.',
    equations: [
      { label: 'rotated attention', tex: 'A^{\\text{rot}} = J\\,A\\,J,\\quad A = \\mathrm{softmax}(QK^\\top/\\sqrt d)' },
      { label: 'LoRA', tex: "W' = W + BA,\\; r \\ll d" },
    ],
    diagram: 'attnLora',
  },
  {
    id: 'conv_oracle',
    num: '02',
    title: 'conv_oracle',
    tag: 'training-free',
    summary: 'Flip every temporal convolution kernel along time. The network becomes an exact time mirror of the teacher.',
    equations: [
      { label: 'kernel flip', tex: "k'_\\tau = k_{-\\tau}" },
      { label: 'result', tex: "G_{k'}(\\varepsilon) = \\mathcal R\\, G_{k}(\\mathcal R\\varepsilon)" },
    ],
    diagram: 'convRotation',
  },
  {
    id: 'conv_lora',
    num: '03',
    title: 'conv_lora',
    tag: 'main',
    summary: 'LoRA on temporal convs and attention, trained on reversed teacher videos with a mirror loss.',
    equations: [
      { label: 'loss', tex: '\\mathcal L = \\mathcal L_{\\varepsilon} + \\lambda\\,\\mathcal L_{\\text{mir}}' },
      { label: 'ε-MSE on reversed latents', tex: '\\mathcal L_{\\varepsilon} = \\|\\varepsilon - \\varepsilon_\\theta(z_t^{\\mathcal R})\\|^2' },
      { label: 'mirror', tex: '\\mathcal L_{\\text{mir}} = \\|\\varepsilon_\\theta(z_t) - \\mathcal R\\,\\varepsilon_\\phi(\\mathcal R z_t)\\|^2' },
    ],
    diagram: 'convLora',
  },
  {
    id: 'metrics',
    num: '04',
    title: 'Metrics',
    summary: 'FVD for distance, CLIP for prompt match, dynamic degree for motion, optical flow for direction.',
    equations: [{ label: 'flow direction', tex: "\\cos = \\frac{\\langle u^{s},\\, -\\mathcal R u^{t}\\rangle}{\\|u^s\\|\\,\\|u^t\\|}" }],
    diagram: 'metrics',
  },
];
