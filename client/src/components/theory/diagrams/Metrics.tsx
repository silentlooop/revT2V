import { motion } from 'framer-motion';
import { mulberry32 } from '../../../lib/rng';
import { C, F, Svg, ease, inView } from './kit';

const r = mulberry32(7);
const cloudA = Array.from({ length: 26 }, () => [40 + (r() + r() - 1) * 26, 70 + (r() + r() - 1) * 26]);
const cloudB = Array.from({ length: 26 }, () => [100 + (r() + r() - 1) * 26, 80 + (r() + r() - 1) * 26]);

export function MetricsDiagram() {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      <Panel n="01" title="FVD ↓">
        <Svg w={150} h={140} label="Two clouds of I3D features; FVD is the distance between them">
          {cloudA.map(([x, y], i) => (
            <circle key={`a${i}`} cx={x} cy={y} r="3" fill={C.ink} />
          ))}
          {cloudB.map(([x, y], i) => (
            <motion.circle key={`b${i}`} cx={x} cy={y} r="3" fill={C.accent} initial={{ cx: x + 30 }} whileInView={{ cx: x }} viewport={inView} transition={{ ...ease, duration: 1 }} />
          ))}
          <ellipse cx={40} cy={70} rx={30} ry={28} fill="none" stroke={C.ink} strokeDasharray="3 3" />
          <ellipse cx={100} cy={80} rx={30} ry={28} fill="none" stroke={C.accent} strokeDasharray="3 3" />
          <line x1={40} y1={70} x2={100} y2={80} stroke={C.ink} strokeWidth="1.5" markerEnd="url(#arr)" />
          <text x={75} y={130} textAnchor="middle" fontSize="10" style={F.mono} fill={C.ink}>
            μ, Σ distance
          </text>
        </Svg>
      </Panel>
      <Panel n="02" title="CLIP ↑">
        <Svg w={150} h={140} label="Frames and prompt embedded by CLIP, compared by cosine">
          {[0, 1, 2].map((i) => (
            <rect key={i} x={10 + i * 8} y={20 + i * 8} width={50} height={50} fill={C.white} stroke={C.ink} strokeWidth="1.5" />
          ))}
          <rect x={92} y={36} width={52} height={26} fill={C.accent} />
          <text x={118} y={53} textAnchor="middle" fontSize="9" style={F.pixel} fill={C.onAccent}>
            “text”
          </text>
          <motion.path d="M50 110 Q 75 70 118 110" fill="none" stroke={C.accent} strokeWidth="2" initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }} viewport={inView} transition={{ ...ease, delay: 0.3 }} />
          <line x1={50} y1={110} x2={50} y2={86} stroke={C.ink} />
          <line x1={118} y1={110} x2={118} y2={62} stroke={C.ink} />
          <text x={84} y={132} textAnchor="middle" fontSize="10" style={F.mono} fill={C.ink}>
            cos(E_I, E_T)
          </text>
        </Svg>
      </Panel>
      <Panel n="03" title="DYNAMIC ↑">
        <Svg w={150} h={140} label="Motion meter: how much the video moves">
          {Array.from({ length: 10 }, (_, i) => (
            <motion.rect
              key={i}
              x={10 + i * 13}
              width={10}
              fill={i < 7 ? C.accent : C.white}
              stroke={C.ink}
              strokeWidth="1"
              initial={{ y: 100, height: 0 }}
              whileInView={{ y: 100 - (i + 2) * 7, height: (i + 2) * 7 }}
              viewport={inView}
              transition={{ ...ease, delay: i * 0.05 }}
            />
          ))}
          <line x1={6} y1={56} x2={144} y2={56} stroke={C.ink} strokeDasharray="4 3" />
          <text x={8} y={50} fontSize="9" style={F.mono} fill={C.ink}>
            threshold
          </text>
          <text x={75} y={124} textAnchor="middle" fontSize="10" style={F.mono} fill={C.ink}>
            RAFT flow / frame
          </text>
        </Svg>
      </Panel>
      <Panel n="04" title="FLOW cos ↑">
        <Svg w={150} h={140} label="Student motion vectors compared with the reversed teacher vectors">
          {Array.from({ length: 9 }, (_, k) => {
            const x = 20 + (k % 3) * 22;
            const y = 30 + Math.floor(k / 3) * 26;
            return <line key={k} x1={x} y1={y} x2={x} y2={y - 14} stroke={C.ink} strokeWidth="1.5" markerEnd="url(#arr)" />;
          })}
          {Array.from({ length: 9 }, (_, k) => {
            const x = 96 + (k % 3) * 22;
            const y = 18 + Math.floor(k / 3) * 26;
            return <line key={k} x1={x} y1={y} x2={x} y2={y + 14} stroke={C.accent} strokeWidth="1.5" markerEnd="url(#arr-a)" />;
          })}
          <text x={42} y={112} textAnchor="middle" fontSize="9" style={F.mono} fill={C.ink}>
            teacher
          </text>
          <text x={118} y={112} textAnchor="middle" fontSize="9" style={F.mono} fill={C.accent}>
            student
          </text>
          <text x={75} y={132} textAnchor="middle" fontSize="10" style={F.mono} fill={C.ink}>
            cos → +1 ✓
          </text>
        </Svg>
      </Panel>
      <div className="col-span-2 lg:col-span-4 flex flex-wrap gap-2 items-center">
        <span className="font-mono text-[11px] bg-ink text-white px-1.5">FROZEN</span>
        <span className="font-mono text-xs text-mute">student motion &lt; 0.3 × teacher</span>
      </div>
    </div>
  );
}

function Panel({ n, title, children }: { n: string; title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white border-2 border-ink shadow-hard-xs p-2">
      <div className="flex items-baseline gap-1.5 border-b border-dashed border-ink/40 pb-1 mb-1">
        <span className="outline-text-accent text-xl leading-none">{n}</span>
        <span className="font-pixel text-[11px]">{title}</span>
      </div>
      {children}
    </div>
  );
}
