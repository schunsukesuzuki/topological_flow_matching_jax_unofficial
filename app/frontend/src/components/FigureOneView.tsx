import type { FigureOnePayload } from "../types";

type Props = { payload: FigureOnePayload };

function scalePoint(p: number[], size: number) {
  const margin = 16;
  const r = (size - 2 * margin) / 2;
  return [size / 2 + p[0] * r, size / 2 + p[1] * r];
}

function quantile(sorted: number[], q: number) {
  if (sorted.length === 0) return 0;
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

function normalizedSignal(signal: number[]) {
  const absVals = signal.map((v) => Math.abs(v)).sort((a, b) => a - b);
  const mean = signal.reduce((a, b) => a + b, 0) / Math.max(signal.length, 1);
  const variance = signal.reduce((acc, v) => acc + (v - mean) ** 2, 0) / Math.max(signal.length, 1);
  const std = Math.sqrt(variance);

  // Robust denominator prevents a few values from saturating the whole panel.
  const robustAbs = quantile(absVals, 0.9);
  const denom = robustAbs > 1e-6 ? robustAbs : Math.max(Math.abs(mean), 1e-6);

  return signal.map((v) => {
    if (std < 1e-5) {
      const s = mean >= 0 ? 0.55 : -0.55;
      return s;
    }
    const x = Math.max(-1, Math.min(1, v / denom));
    return Math.tanh(1.15 * x);
  });
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t;
}

function signalColor(norm: number) {
  const x = Math.max(-1, Math.min(1, norm));
  const ax = Math.abs(x);
  if (x >= 0) {
    // white -> soft red
    return `rgb(${Math.round(lerp(246, 193, ax))}, ${Math.round(lerp(247, 68, ax))}, ${Math.round(lerp(250, 68, ax))})`;
  }
  // white -> soft blue
  return `rgb(${Math.round(lerp(246, 49, ax))}, ${Math.round(lerp(247, 115, ax))}, ${Math.round(lerp(250, 194, ax))})`;
}

function GraphPanel({ title, nodes, edges, signal }: { title: string; nodes: number[][]; edges: number[][]; signal: number[] }) {
  const size = 170;
  const norm = normalizedSignal(signal);
  return (
    <div className="figure-panel">
      <div className="figure-title">{title}</div>
      <svg viewBox={`0 0 ${size} ${size}`} className="figure-svg">
        {edges.map(([a, b], i) => {
          const [x1, y1] = scalePoint(nodes[a], size);
          const [x2, y2] = scalePoint(nodes[b], size);
          return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} className="figure-edge" />;
        })}
        {nodes.map((p, i) => {
          const [x, y] = scalePoint(p, size);
          const mag = Math.abs(norm[i] ?? 0);
          const radius = 3.7 + 1.8 * mag;
          return <circle key={i} cx={x} cy={y} r={radius} fill={signalColor(norm[i] ?? 0)} stroke="#1f2937" strokeWidth={0.45} />;
        })}
      </svg>
    </div>
  );
}

function ArrowEdge({ a, b, nodes, value, size }: { a: number; b: number; nodes: number[][]; value: number; size: number }) {
  const [ax, ay] = scalePoint(nodes[a], size);
  const [bx, by] = scalePoint(nodes[b], size);
  const dx = bx - ax;
  const dy = by - ay;
  const len = Math.sqrt(dx * dx + dy * dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  const nx = -uy;
  const ny = ux;
  const mag = Math.abs(value);

  // Center the arrow on the edge; length varies with magnitude like the paper.
  const arrowLen = Math.max(6, len * (0.18 + 0.55 * mag));
  const cx = (ax + bx) / 2;
  const cy = (ay + by) / 2;
  const dir = value >= 0 ? 1 : -1;
  const sx = cx - dir * ux * arrowLen * 0.5;
  const sy = cy - dir * uy * arrowLen * 0.5;
  const ex = cx + dir * ux * arrowLen * 0.5;
  const ey = cy + dir * uy * arrowLen * 0.5;
  const head = 3.8 + 1.5 * mag;
  const hx1 = ex - ux * head - nx * head * 0.55;
  const hy1 = ey - uy * head - ny * head * 0.55;
  const hx2 = ex - ux * head + nx * head * 0.55;
  const hy2 = ey - uy * head + ny * head * 0.55;
  const stroke = signalColor(value);
  const strokeWidth = 1.1 + 1.2 * mag;

  return (
    <g>
      <line x1={sx} y1={sy} x2={ex} y2={ey} stroke={stroke} strokeWidth={strokeWidth} strokeLinecap="round" opacity={0.95} />
      <line x1={ex} y1={ey} x2={hx1} y2={hy1} stroke={stroke} strokeWidth={strokeWidth} strokeLinecap="round" opacity={0.95} />
      <line x1={ex} y1={ey} x2={hx2} y2={hy2} stroke={stroke} strokeWidth={strokeWidth} strokeLinecap="round" opacity={0.95} />
    </g>
  );
}

function EdgePanel({ title, nodes, edges, faces, signal }: { title: string; nodes: number[][]; edges: number[][]; faces: number[][]; signal: number[] }) {
  const size = 170;
  const norm = normalizedSignal(signal);
  return (
    <div className="figure-panel">
      <div className="figure-title">{title}</div>
      <svg viewBox={`0 0 ${size} ${size}`} className="figure-svg">
        {faces.map((face, i) => {
          const points = face.map((idx) => scalePoint(nodes[idx], size).join(",")).join(" ");
          return <polygon key={i} points={points} className="figure-face" />;
        })}
        {edges.map(([a, b], i) => {
          const [x1, y1] = scalePoint(nodes[a], size);
          const [x2, y2] = scalePoint(nodes[b], size);
          return <line key={`mesh-${i}`} x1={x1} y1={y1} x2={x2} y2={y2} className="figure-edge" />;
        })}
        {edges.map(([a, b], i) => <ArrowEdge key={`arrow-${i}`} a={a} b={b} nodes={nodes} value={norm[i] ?? 0} size={size} />)}
      </svg>
    </div>
  );
}

export function FigureOneView({ payload }: Props) {
  return (
    <section className="card wide figure-card">
      <h3>Figure-1-style Hodge spectrum / heat GP view</h3>
      <p className="muted">
        Top row: graph node signals. Bottom row: 2-simplicial annulus edge signals. The display has been revised to reduce color saturation and use robust per-panel scaling, while edge signals are now rendered as direction-aware arrows with length proportional to magnitude, closer to Figure 1.
      </p>
      <div className="figure-grid">
        {payload.labels.map((label, i) => <GraphPanel key={`g-${i}`} title={label} nodes={payload.nodes} edges={payload.edges} signal={payload.graph_signals[i]} />)}
        {payload.labels.map((label, i) => <EdgePanel key={`e-${i}`} title={label} nodes={payload.nodes} edges={payload.edges} faces={payload.faces} signal={payload.edge_signals[i]} />)}
      </div>
    </section>
  );
}
