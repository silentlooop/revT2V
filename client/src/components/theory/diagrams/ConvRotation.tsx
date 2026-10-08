import { motion } from 'framer-motion';
import { C, F, FrameThumb, Num, Svg, Tag, Wire, ease, inView } from './kit';

const K = [0.18, 0.5, 0.92];

export function ConvRotationDiagram() {
  const bw = 40;
  return (
    <Svg w={640} h={300} label="Diagram: a 1D temporal kernel before and after flipping along time; the flipped network outputs the exact time mirror of the teacher">
      <Num x={14} y={18} n={1} />
      <text x={32} y={22} fontSize="11" style={F.pixel} fill={C.ink}>
        1D TEMPORAL KERNEL k
      </text>
      {/* before */}
      <g transform="translate(30 50)">
        {K.map((v, i) => (
          <g key={i}>
            <rect x={i * (bw + 8)} y={110 - v * 100} width={bw} height={v * 100} fill={C.ink} />
            <text x={i * (bw + 8) + bw / 2} y={128} textAnchor="middle" fontSize="11" style={F.mono} fill={C.ink}>
              w{['₋₁', '₀', '₊₁'][i]}
            </text>
          </g>
        ))}
        <line x1={-6} y1={110} x2={150} y2={110} stroke={C.ink} strokeWidth="1.5" />
        <text x={72} y={150} textAnchor="middle" fontSize="11" style={F.mono} fill={C.mute}>
          teacher k
        </text>
      </g>
      <Wire d="M190 110 C 215 80, 245 80, 270 110" arrow delay={0.3} />
      <text x={230} y={74} textAnchor="middle" fontSize="12" style={F.pixel} fill={C.accent}>
        flip τ→−τ
      </text>
      {/* after: bars animate to swapped positions */}
      <g transform="translate(290 50)">
        {K.map((v, i) => (
          <motion.rect
            key={i}
            y={110 - v * 100}
            width={bw}
            height={v * 100}
            fill={C.accent}
            initial={{ x: i * (bw + 8) }}
            whileInView={{ x: (2 - i) * (bw + 8) }}
            viewport={inView}
            transition={{ ...ease, duration: 1.1, delay: 0.6 }}
          />
        ))}
        {[0, 1, 2].map((i) => (
          <text key={i} x={i * (bw + 8) + bw / 2} y={128} textAnchor="middle" fontSize="11" style={F.mono} fill={C.accent}>
            w{['₊₁', '₀', '₋₁'][i]}
          </text>
        ))}
        <line x1={-6} y1={110} x2={150} y2={110} stroke={C.ink} strokeWidth="1.5" />
        <text x={72} y={150} textAnchor="middle" fontSize="11" style={F.mono} fill={C.accent}>
          k′ (flipped)
        </text>
      </g>

      {/* ORACLE stamp */}
      <motion.g initial={{ scale: 1.6, opacity: 0, rotate: -14 }} whileInView={{ scale: 1, opacity: 1, rotate: -8 }} viewport={inView} transition={{ duration: 0.35, delay: 1.5 }} style={{ transformOrigin: '548px 96px' }}>
        <rect x={478} y={70} width={140} height={52} fill="none" stroke={C.accent} strokeWidth="3" />
        <rect x={483} y={75} width={130} height={42} fill="none" stroke={C.accent} strokeWidth="1" />
        <text x={548} y={103} textAnchor="middle" fontSize="15" style={F.pixel} fill={C.accent}>
          NO TRAINING
        </text>
      </motion.g>

      {/* consequence */}
      <Num x={14} y={222} n={2} />
      <text x={32} y={226} fontSize="11" style={F.pixel} fill={C.ink}>
        SAME NOISE IN → MIRROR OUT
      </text>
      {Array.from({ length: 6 }, (_, i) => (
        <FrameThumb key={`t${i}`} x={30 + i * 34} y={244} s={30} t={i / 5} />
      ))}
      <text x={130} y={292} textAnchor="middle" fontSize="10" style={F.mono} fill={C.ink}>
        G_k(ε)
      </text>
      <text x={268} y={266} textAnchor="middle" fontSize="18" style={F.cond} fill={C.ink}>
        ≡ 𝓡 ·
      </text>
      {Array.from({ length: 6 }, (_, i) => (
        <FrameThumb key={`o${i}`} x={310 + i * 34} y={244} s={30} t={1 - i / 5} dot={C.accent} />
      ))}
      <text x={410} y={292} textAnchor="middle" fontSize="10" style={F.mono} fill={C.accent}>
        G_k′(𝓡ε)
      </text>
      <Tag x={530} y={262}>
        Δ ≈ 0 ✓
      </Tag>
    </Svg>
  );
}
