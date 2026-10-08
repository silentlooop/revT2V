import { motion } from 'framer-motion';
import { C, F, Num, Sel, Svg, Tag, Wire, ease, inView } from './kit';

const N = 8;
// a plausible temporal-attention pattern: strong diagonal, leaning to the previous frames
const A = Array.from({ length: N }, (_, i) =>
  Array.from({ length: N }, (_, j) => {
    const d = i - j;
    return Math.exp(-((d - 0.8) ** 2) / 2.2) + (j === 0 ? 0.25 : 0);
  }),
).map((row) => {
  const s = row.reduce((a, b) => a + b, 0);
  return row.map((v) => v / s);
});

function Matrix({ x, y, cell, title, rotate }: { x: number; y: number; cell: number; title: string; rotate?: boolean }) {
  const size = cell * N;
  const max = Math.max(...A.flat());
  return (
    <g>
      <text x={x} y={y - 12} fontSize="12" style={F.pixel} fill={C.ink}>
        {title}
      </text>
      <motion.g
        initial={rotate ? { rotate: 0 } : undefined}
        whileInView={rotate ? { rotate: 180 } : undefined}
        viewport={inView}
        transition={{ ...ease, duration: 1.4, delay: 0.5 }}
        style={{ transformOrigin: `${x + size / 2}px ${y + size / 2}px` }}
      >
        {A.map((row, i) =>
          row.map((v, j) => (
            <rect key={`${i}-${j}`} x={x + j * cell} y={y + i * cell} width={cell - 1} height={cell - 1} fill={C.accent} opacity={0.06 + (v / max) * 0.94} />
          )),
        )}
        <rect x={x} y={y} width={size} height={size} fill="none" stroke={C.ink} strokeWidth="2" />
        <circle cx={x + cell / 2} cy={y + cell / 2} r="3" fill={C.white} stroke={C.ink} />
      </motion.g>
      <text x={x + size / 2} y={y + size + 16} textAnchor="middle" fontSize="10" style={F.mono} fill={C.ink}>
        key frame j →
      </text>
      <text x={x - 8} y={y + size / 2} textAnchor="middle" fontSize="10" style={F.mono} fill={C.ink} transform={`rotate(-90 ${x - 8} ${y + size / 2})`}>
        query frame i →
      </text>
    </g>
  );
}

export function AttnLoraDiagram() {
  const cell = 20;
  return (
    <Svg w={640} h={280} label="Diagram: an 8 by 8 temporal attention matrix and the same matrix rotated by 180 degrees">
      <Num x={20} y={20} n={1} />
      <Matrix x={40} y={56} cell={cell} title="A = softmax(QKᵀ/√d)" />
      <Wire d="M218 136 C 250 110, 280 110, 312 136" arrow delay={0.3} />
      <text x={265} y={104} textAnchor="middle" fontSize="20" style={F.cond} fill={C.accent}>
        180°
      </text>
      <text x={265} y={160} textAnchor="middle" fontSize="11" style={F.mono} fill={C.ink}>
        J A J
      </text>
      <Num x={330} y={20} n={2} />
      <Matrix x={330} y={56} cell={cell} title="A_rot" rotate />
      <Sel x={326} y={52} w={cell * N + 8} h={cell * N + 8} />

      {/* V / O projections with LoRA */}
      <g transform="translate(520 56)">
        <Num x={0} y={-36} n={3} />
        {['to_q', 'to_k', 'to_v', 'to_out'].map((p, i) => {
          const lora = i >= 2;
          return (
            <g key={p} transform={`translate(0 ${i * 40})`}>
              <rect x={0} y={0} width={70} height={28} fill={lora ? C.white : 'url(#ht-ink)'} stroke={C.ink} strokeWidth="1.5" />
              {!lora && <rect x={0} y={0} width={70} height={28} fill={C.white} opacity=".6" />}
              <text x={35} y={18} textAnchor="middle" fontSize="11" style={F.mono} fill={C.ink}>
                {p}
              </text>
              {lora && (
                <motion.g initial={{ x: 10, opacity: 0 }} whileInView={{ x: 0, opacity: 1 }} viewport={inView} transition={{ ...ease, delay: 1 + i * 0.1 }}>
                  <rect x={74} y={3} width={30} height={22} fill={C.accent} />
                  <text x={89} y={17} textAnchor="middle" fontSize="8" style={F.pixel} fill={C.onAccent}>
                    LoRA
                  </text>
                </motion.g>
              )}
            </g>
          );
        })}
        <text x={35} y={176} textAnchor="middle" fontSize="10" style={F.mono} fill={C.mute}>
          q,k frozen
        </text>
      </g>

      <Tag x={40} y={262} tone="soft">
        frame f ↔ F−1−f
      </Tag>
    </Svg>
  );
}
