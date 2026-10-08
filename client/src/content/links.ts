/** EDIT ME — project links + references shown at the bottom of the page. Leave `url` empty to show "coming soon". */
export const LINKS = [
  { kind: 'Hugging Face', label: 'Teacher model', repo: 'ali-vilab/text-to-video-ms-1.7b', url: 'https://huggingface.co/ali-vilab/text-to-video-ms-1.7b' },
  { kind: 'Hugging Face', label: 'Student checkpoints', repo: 'silentlooop/revt2v-ckpt', url: 'https://huggingface.co/silentlooop/revt2v-ckpt' },
  { kind: 'GitHub', label: 'Code', repo: 'silentlooop/revT2V', url: 'https://github.com/silentlooop/revT2V' },
];

export interface Reference {
  title: string;
  note?: string;
  /** leave empty until you have the link — shows "link needed" */
  url?: string;
}

export const REFERENCES: Reference[] = [
  { title: 'Complete guide for distillation', note: 'TODO: source' },
  { title: 'Time reversal sampling (MPD)', note: 'TODO: source' },
  { title: 'Generative Inbetweening: Adapting Image-to-Video Models for Keyframe Interpolation' },
  { title: 'Can Video Diffusion Models Predict Past Frames? Bidirectional Cycle Consistency for Reversible Interpolation' },
  { title: 'VBench', note: 'evaluation suite / prompts', url: 'https://github.com/Vchitect/VBench' },
  { title: 'How to turn Google Colab and Kaggle into a live server' },
  { title: 'What are Diffusion Models?', note: 'Lilian Weng', url: 'https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#nice' },
  { title: 'ModelScope Text-to-Video', note: 'Wang et al., 2023', url: 'https://arxiv.org/abs/2308.06571' },
  { title: 'High-Resolution Image Synthesis with Latent Diffusion Models (LDM)', note: 'Rombach et al., CVPR 2022', url: 'https://arxiv.org/abs/2112.10752' },
];
