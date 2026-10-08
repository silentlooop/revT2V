import { motion } from 'framer-motion';
import { C, Card, F, FrameThumb, Num, Svg, Tag, Wire, ease, inView } from './kit';

/** conv_lora: LoRA on temporal Conv3d + attention, and the two-part loss. */
export function ConvLoraDiagram() {
  const taps = [0.35, 0.8, 0.55];
  const delta = [0.25, -0.1, -0.22];
  return (
    <Svg w={640} h={380} label="Diagram: LoRA on a temporal convolution kernel and temporal attention; the loss is epsilon MSE on reversed latents plus a weighted mirror loss">
      {/* kernel + lora */}
      <Num x={14} y={18} n={1} />
      <text x={32} y={22} fontSize="11" style={F.pixel} fill={C.ink}>
        TEMPORAL CONV3D · kernel (3,1,1)
      </text>
      <g transform="translate(30 50)">
        {taps.map((v, i) => (
          <g key={i}>
            <rect x={i * 44} y={100 - v * 100} width={34} height={v * 100} fill="url(#ht-ink)" stroke={C.ink} strokeWidth="1.5" />
            <motion.rect
              x={i * 44}
              width={34}
              fill={C.accent}
              opacity={0.85}
              initial={{ y: 100 - v * 100, height: 0 }}
              whileInView={{ y: 100 - Math.max(v, v + delta[i]) * 100, height: Math.abs(delta[i]) * 100 }}
              viewport={inView}
              transition={{ ...ease, delay: 0.4 + i * 0.1 }}
              style={delta[i] < 0 ? { mixBlendMode: 'multiply' } : undefined}
            />
            <text x={i * 44 + 17} y={118} textAnchor="middle" fontSize="10" style={F.mono} fill={C.ink}>
              {['f−1', 'f', 'f+1'][i]}
            </text>
          </g>
        ))}
        <line x1={-6} y1={100} x2={136} y2={100} stroke={C.ink} strokeWidth="1.5" />
        <Tag x={150} y={30}>
          + ΔW = BA
        </Tag>
      </g>

      {/* where lora goes */}
      <g transform="translate(330 30)">
        <Num x={-16} y={-12} n={2} />
        <text x={2} y={-8} fontSize="11" style={F.pixel} fill={C.ink}>
          TRAINABLE: BLUE ONLY
        </text>
        {[
          ['spatial conv', false],
          ['spatial attn', false],
          ['cross-attn (text)', false],
          ['temporal conv ×4', true],
          ['temporal attn', true],
        ].map(([n, on], i) => (
          <g key={String(n)} transform={`translate(0 ${10 + i * 28})`}>
            <rect x={0} y={0} width={200} height={22} fill={on ? C.accent : C.white} stroke={C.ink} strokeWidth="1.5" />
            {!on && <rect x={0} y={0} width={200} height={22} fill="url(#stripe)" opacity=".12" />}
            <text x={10} y={15} fontSize="11" style={F.mono} fill={on ? C.onAccent : C.ink}>
              {String(n)}
            </text>
            <text x={190} y={15} textAnchor="end" fontSize="10" style={F.pixel} fill={on ? C.onAccent : C.mute}>
              {on ? 'LoRA' : '❄'}
            </text>
          </g>
        ))}
      </g>

      {/* losses */}
      <Num x={14} y={206} n={3} />
      <text x={32} y={210} fontSize="11" style={F.pixel} fill={C.ink}>
        LOSS = ε-MSE + λ · MIRROR
      </text>
      <Card x={20} y={226} w={270} h={120} fill={C.white}>
        <text x={32} y={246} fontSize="11" style={F.pixel} fill={C.ink}>
          A · ε-MSE on noisy 𝓡z₀
        </text>
        {[0, 1, 2, 3].map((i) => (
          <FrameThumb key={i} x={32 + i * 30} y={258} s={26} t={1 - i / 3} dot={C.accent} noise={0.45} />
        ))}
        <text x={160} y={276} fontSize="12" style={F.mono} fill={C.ink}>
          → ε_θ ≈ ε
        </text>
      </Card>
      <Card x={310} y={226} w={310} h={120} fill={C.wash}>
        <text x={322} y={246} fontSize="11" style={F.pixel} fill={C.ink}>
          B · mirror vs flipped teacher
        </text>
        <text x={322} y={272} fontSize="12" style={F.mono} fill={C.ink}>
          ε_θ(z_t)
        </text>
        <text x={402} y={272} fontSize="14" style={F.cond} fill={C.accent}>
          ≈
        </text>
        <text x={422} y={272} fontSize="12" style={F.mono} fill={C.ink}>
          𝓡 ε_φ(𝓡 z_t)
        </text>
      </Card>
      {/* animated loss composition bar */}
      <g transform="translate(20 358)">
        <motion.rect x={0} y={0} height={14} fill={C.ink} initial={{ width: 0 }} whileInView={{ width: 390 }} viewport={inView} transition={{ ...ease, duration: 1, delay: 0.6 }} />
        <motion.rect y={0} height={14} fill={C.accent} initial={{ x: 390, width: 0 }} whileInView={{ x: 390, width: 210 }} viewport={inView} transition={{ ...ease, duration: 0.8, delay: 1.4 }} />
        <text x={8} y={11} fontSize="9" style={F.pixel} fill={C.white}>
          ε-MSE
        </text>
        <text x={398} y={11} fontSize="9" style={F.pixel} fill={C.onAccent}>
          λ·MIRROR
        </text>
      </g>
      <Wire d="M290 286 L306 286" accent={false} arrow delay={0.8} />
    </Svg>
  );
}
