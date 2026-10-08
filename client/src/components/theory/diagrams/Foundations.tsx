import { motion } from 'framer-motion';
import { C, Card, F, FrameThumb, Num, Sel, Svg, Tag, Wire, ease, inView } from './kit';

function Stack({ x, y, s, noise, t, i }: { x: number; y: number; s: number; noise: number; t: number; i: number }) {
  return (
    <motion.g initial={{ opacity: 0, y: 8 }} whileInView={{ opacity: 1, y: 0 }} viewport={inView} transition={{ ...ease, delay: i * 0.12 }}>
      {[2, 1, 0].map((k) => (
        <FrameThumb key={k} x={x + k * 6} y={y - k * 6} s={s} t={t} noise={noise} fill={C.white} />
      ))}
    </motion.g>
  );
}

function Block({ x, y, label, i }: { x: number; y: number; label: string; i: number }) {
  return (
    <motion.g initial={{ opacity: 0, scale: 0.9 }} whileInView={{ opacity: 1, scale: 1 }} viewport={inView} transition={{ ...ease, delay: 0.3 + i * 0.08 }} style={{ transformOrigin: `${x + 35}px ${y + 22}px` }}>
      <rect x={x + 3} y={y + 3} width="72" height="44" fill={C.ink} />
      <rect x={x} y={y} width="50" height="44" fill="url(#ht-ink)" stroke={C.ink} strokeWidth="2" />
      <rect x={x} y={y} width="50" height="44" fill={C.white} opacity=".55" />
      <motion.rect
        x={x + 50}
        y={y}
        width="22"
        height="44"
        fill={C.accent}
        stroke={C.ink}
        strokeWidth="2"
        animate={{ opacity: [1, 0.55, 1] }}
        transition={{ duration: 2.4, repeat: Infinity, delay: i * 0.2 }}
      />
      <text x={x + 25} y={y + 26} textAnchor="middle" fontSize="10" style={F.mono} fill={C.ink}>
        {label}
      </text>
      <text x={x + 61} y={y + 27} textAnchor="middle" fontSize="9" style={F.pixel} fill={C.onAccent}>
        T
      </text>
    </motion.g>
  );
}

export function FoundationsDiagram() {
  const noise = [1, 0.72, 0.45, 0.2, 0];
  const down = [
    [40, 168],
    [120, 214],
    [200, 260],
  ];
  const up = [
    [380, 260],
    [460, 214],
    [540, 168],
  ];
  return (
    <Svg w={640} h={470} label="Diagram: diffusion denoises a noisy latent video into a clean one through a factorised 3D U-Net whose temporal layers are highlighted">
      {/* row 1: noise → video */}
      <Num x={12} y={14} n={1} />
      <text x={28} y={18} fontSize="11" style={F.pixel} fill={C.ink}>
        DENOISE · DDIM ×25
      </text>
      {noise.map((n, i) => (
        <g key={i}>
          <Stack x={14 + i * 104} y={44} s={60} noise={n} t={0.5} i={i} />
          <text x={44 + i * 104 + 6} y={130} textAnchor="middle" fontSize="11" style={F.mono} fill={C.ink}>
            {['z_T', 'z_¾T', 'z_½T', 'z_¼T', 'z_0'][i]}
          </text>
          {i < 4 && <Wire d={`M${88 + i * 104} 70 L${110 + i * 104} 70`} arrow delay={0.2 + i * 0.12} />}
        </g>
      ))}
      <Card x={540} y={36} w={92} h={70} fill={C.wash}>
        <text x={586} y={62} textAnchor="middle" fontSize="11" style={F.pixel} fill={C.ink}>
          VAE dec
        </text>
        <text x={586} y={80} textAnchor="middle" fontSize="10" style={F.mono} fill={C.ink}>
          32²×4 → 256²
        </text>
        <text x={586} y={95} textAnchor="middle" fontSize="10" style={F.mono} fill={C.mute}>
          ×16 frames
        </text>
      </Card>
      <Wire d="M516 70 L536 70" arrow delay={0.8} />

      {/* row 2: factorised U-Net */}
      <Num x={270} y={212} n={2} />
      <text x={286} y={216} fontSize="11" style={F.pixel} fill={C.ink}>
        3D U-NET
      </text>
      {down.map(([x, y], i) => (
        <Block key={`d${i}`} x={x} y={y} label={`down${i + 1}`} i={i} />
      ))}
      <Block x={290} y={316} label="mid" i={3} />
      {up.map(([x, y], i) => (
        <Block key={`u${i}`} x={x} y={y} label={`up${3 - i}`} i={4 + i} />
      ))}
      <Wire d="M112 190 L120 214" accent={false} arrow delay={0.5} />
      <Wire d="M192 236 L200 260" accent={false} arrow delay={0.55} />
      <Wire d="M272 282 L290 316" accent={false} arrow delay={0.6} />
      <Wire d="M362 338 L380 282" accent={false} arrow delay={0.65} />
      <Wire d="M452 260 L460 236" accent={false} arrow delay={0.7} />
      <Wire d="M532 214 L540 190" accent={false} arrow delay={0.75} />
      {[0, 1, 2].map((i) => (
        <Wire key={i} d={`M${down[i][0] + 72} ${down[i][1] + 22} L${up[2 - i][0]} ${up[2 - i][1] + 22}`} accent={false} dashed delay={0.9 + i * 0.1} />
      ))}
      <text x={320} y={180} textAnchor="middle" fontSize="9" style={F.mono} fill={C.mute}>
        skip connections
      </text>
      <Sel x={36} y={164} w={58} h={52} dashed />
      <Tag x={36} y={150} tone="white">
        SPATIAL · per frame
      </Tag>
      <Sel x={586} y={164} w={30} h={52} />
      <Tag x={636} y={150} anchor="end">
        TEMPORAL · across frames
      </Tag>
      {/* temporal layer zoom */}
      <g>
        <rect x={20} y={318} width={262} height={56} fill={C.white} stroke={C.ink} strokeWidth="1.5" />
        <text x={28} y={334} fontSize="10" style={F.pixel} fill={C.accent}>
          T = temporal conv + temporal attn
        </text>
        <text x={28} y={352} fontSize="10" style={F.mono} fill={C.ink}>
          conv: kernel 3×1×1 over (f-1, f, f+1)
        </text>
        <text x={28} y={367} fontSize="10" style={F.mono} fill={C.ink}>
          attn: F×F map per pixel position
        </text>
      </g>

      {/* row 3: what reverse time means */}
      <Num x={12} y={402} n={3} />
      <text x={28} y={406} fontSize="11" style={F.pixel} fill={C.ink}>
        REVERSE TIME
      </text>
      {Array.from({ length: 7 }, (_, i) => (
        <FrameThumb key={`f${i}`} x={20 + i * 36} y={418} s={30} t={i / 6} />
      ))}
      <Wire d="M20 458 L272 458" accent={false} arrow />
      <text x={146} y={470} textAnchor="middle" fontSize="10" style={F.mono} fill={C.ink}>
        teacher: smoke rises
      </text>
      {Array.from({ length: 7 }, (_, i) => (
        <FrameThumb key={`r${i}`} x={370 + i * 36} y={418} s={30} t={1 - i / 6} dot={C.accent} />
      ))}
      <Wire d="M370 458 L622 458" arrow />
      <text x={496} y={470} textAnchor="middle" fontSize="10" style={F.mono} fill={C.accent}>
        goal: smoke sinks back in
      </text>
      <text x={321} y={442} textAnchor="middle" fontSize="20" style={F.cond} fill={C.ink}>
        ⇄
      </text>
    </Svg>
  );
}
